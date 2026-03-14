"""
Drifting Model with SimCLR Encoder for CIFAR-10.

SimCLR (contrastive learning) is better suited for drifting because:
- Trained to make similar samples close in feature space
- Explicitly optimizes for distance/similarity measurement
- This is exactly what drifting field computation needs!

Two modes:
1. Use ImageNet-pretrained encoder (quick, may transfer well)
2. Train SimCLR on CIFAR-10 first (better alignment, takes longer)
"""

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============== SimCLR Training Components ==============

class SimCLRAugmentation:
    """SimCLR-style augmentation for contrastive learning."""

    def __init__(self, size=32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class SimCLREncoder(nn.Module):
    """ResNet18 encoder for SimCLR, modified for CIFAR-10 (32x32)."""

    def __init__(self, feature_dim=128):
        super().__init__()

        # Use ResNet18 but modify for CIFAR-10
        resnet = models.resnet18(pretrained=False)

        # Modify first conv for 32x32 images (no aggressive downsampling)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()  # Remove maxpool

        # Encoder (everything except final fc)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )

        # Projection head (MLP)
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
        )

        self.feature_dim = 512  # Encoder output dim
        self.proj_dim = feature_dim

    def forward(self, x):
        """Return both encoder features and projected features."""
        h = self.encoder(x)
        h = h.flatten(start_dim=1)  # [B, 512]
        z = self.projector(h)  # [B, feature_dim]
        return h, z

    def encode(self, x):
        """Return only encoder features (for drifting)."""
        h = self.encoder(x)
        return h.flatten(start_dim=1)


def nt_xent_loss(z1, z2, temperature=0.5):
    """NT-Xent loss (Normalized Temperature-scaled Cross Entropy)."""
    batch_size = z1.shape[0]

    # Normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate
    z = torch.cat([z1, z2], dim=0)  # [2B, D]

    # Similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # [2B, 2B]

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i+B) and (i+B, i)
    pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, device=z.device)
    pos_mask[:batch_size, batch_size:] = torch.eye(batch_size, device=z.device)
    pos_mask[batch_size:, :batch_size] = torch.eye(batch_size, device=z.device)

    # Loss
    log_softmax = F.log_softmax(sim, dim=1)
    loss = -(log_softmax * pos_mask).sum() / (2 * batch_size)

    return loss


def train_simclr(config, num_epochs=100):
    """Pre-train SimCLR encoder on CIFAR-10."""

    device = torch.device(config.device)
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training SimCLR encoder on CIFAR-10...")

    # Dataset with SimCLR augmentation
    train_dataset = datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        download=True,
        transform=SimCLRAugmentation(size=32),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    # Model
    encoder = SimCLREncoder(feature_dim=128).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training
    encoder.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for (x1, x2), _ in tqdm(train_loader, desc=f"SimCLR Epoch {epoch+1}/{num_epochs}"):
            x1, x2 = x1.to(device), x2.to(device)

            _, z1 = encoder(x1)
            _, z2 = encoder(x2)

            loss = nt_xent_loss(z1, z2, temperature=0.5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    # Save encoder
    torch.save(encoder.state_dict(), save_dir / "simclr_encoder.pt")
    logger.info(f"SimCLR encoder saved to {save_dir / 'simclr_encoder.pt'}")

    return encoder


# ============== Drifting Training Components ==============

@dataclass
class Config:
    data_dir: str = "./data"
    num_classes: int = 10
    image_size: int = 32

    noise_dim: int = 128
    hidden_dim: int = 256
    base_ch: int = 128

    drift_steps: int = 100000
    batch_size: int = 128
    n_pos: int = 256
    lr: float = 2e-4
    beta1: float = 0.0
    beta2: float = 0.99
    warmup_steps: int = 1000
    ema_decay: float = 0.999

    temperatures: Tuple[float, ...] = (0.02, 0.05, 0.2)

    save_dir: str = "./cifar10_simclr"
    log_every: int = 100
    save_every: int = 2000
    seed: int = 42
    device: str = "cuda"

    # SimCLR specific
    simclr_epochs: int = 100
    use_pretrained_simclr: bool = False  # If True, load from file


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch or upsample else nn.Identity()

    def forward(self, x, emb):
        h = F.silu(self.norm1(x))
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        scale, shift = self.emb_proj(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class ImageGenerator(nn.Module):
    def __init__(self, noise_dim=128, hidden_dim=256, num_classes=10, base_ch=128):
        super().__init__()
        self.base_ch = base_ch
        self.class_emb = nn.Embedding(num_classes, hidden_dim)
        self.proj = nn.Linear(noise_dim + hidden_dim, base_ch * 4 * 4 * 4)
        self.block1 = ResBlock(base_ch * 4, base_ch * 4, hidden_dim, upsample=True)
        self.block2 = ResBlock(base_ch * 4, base_ch * 2, hidden_dim, upsample=True)
        self.block3 = ResBlock(base_ch * 2, base_ch, hidden_dim, upsample=True)
        self.out_norm = nn.BatchNorm2d(base_ch)
        self.out_conv = nn.Conv2d(base_ch, 3, 3, padding=1)

    def forward(self, z, labels):
        c_emb = self.class_emb(labels)
        h = self.proj(torch.cat([z, c_emb], dim=1)).view(-1, self.base_ch * 4, 4, 4)
        h = self.block1(h, c_emb)
        h = self.block2(h, c_emb)
        h = self.block3(h, c_emb)
        return torch.tanh(self.out_conv(F.silu(self.out_norm(h))))


def compute_drift(gen_feat, pos_feat, temperatures):
    """Compute drifting field."""
    V_total = torch.zeros_like(gen_feat)
    D = gen_feat.shape[1]

    for temp in temperatures:
        targets = torch.cat([gen_feat, pos_feat], dim=0)
        G = gen_feat.shape[0]

        dist = torch.cdist(gen_feat, targets) / math.sqrt(D)
        dist[:, :G].fill_diagonal_(1e6)

        kernel = (-dist / temp).exp()
        row_sum = kernel.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        col_sum = kernel.sum(dim=-2, keepdim=True).clamp_min(1e-12)
        normalized_kernel = kernel / (row_sum * col_sum).sqrt()

        pos_weight = normalized_kernel[:, G:]
        neg_weight = normalized_kernel[:, :G]

        V = (pos_weight * neg_weight.sum(dim=-1, keepdim=True)) @ targets[G:] - \
            (neg_weight * pos_weight.sum(dim=-1, keepdim=True)) @ targets[:G]

        V = V / (torch.sqrt(torch.mean(V ** 2) + 1e-8) + 1e-8)
        V_total = V_total + V

    return V_total


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data

    def apply(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data = self.shadow[n]

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[n]


def train_drifting(config, encoder):
    """Train generator with SimCLR encoder for drifting."""

    device = torch.device(config.device)
    save_dir = Path(config.save_dir)
    torch.manual_seed(config.seed)

    # Freeze encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Load CIFAR-10
    logger.info("Loading CIFAR-10...")
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(root=config.data_dir, train=True, download=True, transform=transform)

    all_images = torch.stack([dataset[i][0] for i in range(len(dataset))])
    all_labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    logger.info(f"Loaded {len(all_images)} images")

    # Pre-extract features
    logger.info("Pre-extracting SimCLR features...")
    all_features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_images), 256), desc="Extracting"):
            batch = all_images[i:i+256].to(device)
            feat = encoder.encode(batch)
            all_features.append(feat.cpu())
    all_features = torch.cat(all_features, dim=0)
    logger.info(f"Feature shape: {all_features.shape}")

    # Generator
    generator = ImageGenerator(
        noise_dim=config.noise_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        base_ch=config.base_ch,
    ).to(device)

    logger.info(f"Generator params: {sum(p.numel() for p in generator.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(generator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - config.warmup_steps) / (config.drift_steps - config.warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = EMA(generator, config.ema_decay)

    losses = []

    logger.info(f"Starting drifting training with SimCLR encoder for {config.drift_steps} steps...")

    pbar = tqdm(range(1, config.drift_steps + 1), desc="Training")
    for step in pbar:
        generator.train()

        classes = torch.randint(0, config.num_classes, (config.batch_size,), device=device)
        z = torch.randn(config.batch_size, config.noise_dim, device=device)
        gen_images = generator(z, classes)

        # Extract features
        gen_feat = encoder.encode(gen_images)

        # Sample positive features
        pos_per_class = config.n_pos // config.num_classes
        pos_feat_list = []
        for c in range(config.num_classes):
            class_idx = torch.where(all_labels == c)[0]
            sample_idx = class_idx[torch.randint(0, len(class_idx), (pos_per_class,))]
            pos_feat_list.append(all_features[sample_idx])
        pos_feat = torch.cat(pos_feat_list, dim=0).to(device)

        # Class-conditional loss
        total_loss = 0.0
        count = 0

        for c in range(config.num_classes):
            gen_mask = classes == c
            if gen_mask.sum() == 0:
                continue

            gen_c = gen_feat[gen_mask]
            pos_c = pos_feat[c * pos_per_class:(c + 1) * pos_per_class]

            V = compute_drift(gen_c, pos_c, list(config.temperatures))
            target = (gen_c + V).detach()
            loss_c = F.mse_loss(gen_c, target)

            total_loss = total_loss + loss_c * gen_mask.sum().item()
            count += gen_mask.sum().item()

        loss = total_loss / count if count > 0 else torch.tensor(0.0, device=device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        ema.update(generator)

        losses.append(loss.item())

        if step % config.log_every == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4e}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        if step % config.save_every == 0 or step == 1:
            save_samples(generator, ema, all_images, all_labels, step, save_dir, losses, config, device)
            torch.save({
                "step": step,
                "generator": generator.state_dict(),
                "ema_shadow": ema.shadow,
            }, save_dir / "checkpoint.pt")

    logger.info(f"Training complete! Final loss: {losses[-1]:.4e}")


@torch.no_grad()
def save_samples(generator, ema, all_images, all_labels, step, save_dir, losses, config, device):
    generator.eval()
    ema.apply(generator)

    samples = []
    for c in range(config.num_classes):
        z = torch.randn(10, config.noise_dim, device=device)
        labels = torch.full((10,), c, dtype=torch.long, device=device)
        s = (generator(z, labels).clamp(-1, 1) + 1) / 2
        samples.append(s.cpu())
    samples = torch.cat(samples, dim=0)
    ema.restore(generator)

    fig, axes = plt.subplots(10, 10, figsize=(12, 12))
    for i in range(10):
        for j in range(10):
            axes[i, j].imshow(samples[i * 10 + j].permute(1, 2, 0).numpy())
            axes[i, j].axis("off")
    plt.suptitle(f"SimCLR Encoder - Step {step}")
    plt.tight_layout()
    plt.savefig(save_dir / f"samples_step{step}.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    plt.savefig(save_dir / f"losses_step{step}.png", dpi=150)
    plt.close()

    logger.info(f"Saved samples at step {step}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./cifar10_simclr")
    parser.add_argument("--drift_steps", type=int, default=100000)
    parser.add_argument("--simclr_epochs", type=int, default=100)
    parser.add_argument("--skip_simclr_training", action="store_true", help="Load pretrained SimCLR")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = Config(
        save_dir=args.save_dir,
        drift_steps=args.drift_steps,
        simclr_epochs=args.simclr_epochs,
        device=args.device,
    )

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)

    # Step 1: Train or load SimCLR encoder
    encoder_path = save_dir / "simclr_encoder.pt"

    if args.skip_simclr_training and encoder_path.exists():
        logger.info(f"Loading pretrained SimCLR encoder from {encoder_path}")
        encoder = SimCLREncoder(feature_dim=128).to(device)
        encoder.load_state_dict(torch.load(encoder_path))
    else:
        logger.info("Training SimCLR encoder from scratch...")
        encoder = train_simclr(config, num_epochs=config.simclr_epochs)

    # Step 2: Train drifting with SimCLR encoder
    train_drifting(config, encoder)


if __name__ == "__main__":
    main()
