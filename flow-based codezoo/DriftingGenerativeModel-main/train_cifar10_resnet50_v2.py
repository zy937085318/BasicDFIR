"""
Drifting Model Training for CIFAR-10 with ResNet50.
Exact same setup as working ResNet18 version, just with ResNet50 encoder.
"""

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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


@dataclass
class Config:
    # Data
    data_dir: str = "./data"
    num_classes: int = 10
    image_size: int = 32

    # Generator
    noise_dim: int = 128
    hidden_dim: int = 256
    base_ch: int = 128

    # Training
    drift_steps: int = 100000
    batch_size: int = 128
    n_pos: int = 256
    lr: float = 2e-4
    beta1: float = 0.0
    beta2: float = 0.99
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    ema_decay: float = 0.999

    # Drifting - Adjusted temperatures for ResNet50's feature space
    # ResNet50 has 2048-dim features vs ResNet18's 512-dim
    # sqrt(2048)/sqrt(512) = 2, so distances are ~2x larger after sqrt(D) normalization
    # We need ~2x larger temperatures
    temperatures: Tuple[float, ...] = (0.04, 0.1, 0.4)

    # Logging
    save_dir: str = "./cifar10_resnet50_v2"
    log_every: int = 100
    save_every: int = 2000
    seed: int = 42
    device: str = "cuda"


class ResBlock(nn.Module):
    """Residual block with conditional batch norm."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, upsample: bool = False):
        super().__init__()
        self.upsample = upsample

        self.norm1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)

        if in_ch != out_ch or upsample:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)

        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        h = self.conv1(h)

        # Condition on class embedding
        emb_out = self.emb_proj(emb)[:, :, None, None]
        scale, shift = emb_out.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift

        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)


class ImageGenerator(nn.Module):
    """Generator for 32x32 CIFAR-10 images."""

    def __init__(
        self,
        noise_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 10,
        base_ch: int = 128,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        ch = base_ch

        self.class_emb = nn.Embedding(num_classes, hidden_dim)
        self.noise_proj = nn.Linear(noise_dim, hidden_dim)

        # 4x4 -> 8x8 -> 16x16 -> 32x32
        self.initial = nn.Linear(hidden_dim, ch * 8 * 4 * 4)

        self.blocks = nn.ModuleList([
            ResBlock(ch * 8, ch * 4, hidden_dim, upsample=True),  # 4->8
            ResBlock(ch * 4, ch * 2, hidden_dim, upsample=True),  # 8->16
            ResBlock(ch * 2, ch * 1, hidden_dim, upsample=True),  # 16->32
        ])

        self.final_norm = nn.BatchNorm2d(ch)
        self.final_conv = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        class_emb = self.class_emb(labels)
        noise_emb = self.noise_proj(z)
        emb = class_emb + noise_emb

        h = self.initial(emb)
        h = h.view(h.size(0), -1, 4, 4)

        for block in self.blocks:
            h = block(h, emb)

        h = self.final_norm(h)
        h = F.silu(h)
        h = self.final_conv(h)
        h = torch.tanh(h)

        return h


class FeatureEncoder(nn.Module):
    """Frozen ResNet50 feature encoder."""

    def __init__(self, freeze: bool = True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        # Apply ImageNet normalization
        x = (x - self.mean) / self.std
        # Resize to 224x224
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        # Extract features
        feat = self.features(x)
        feat = feat.flatten(start_dim=1)
        return feat

    def train(self, mode: bool = True):
        super().train(mode)
        self.features.eval()
        return self


def compute_drift_features(
    gen_feat: torch.Tensor,
    pos_feat: torch.Tensor,
    temperatures: List[float],
) -> torch.Tensor:
    """Compute drifting field in feature space."""
    V_total = torch.zeros_like(gen_feat)
    D = gen_feat.shape[1]

    for temp in temperatures:
        targets = torch.cat([gen_feat, pos_feat], dim=0)
        G = gen_feat.shape[0]

        dist = torch.cdist(gen_feat, targets)
        dist = dist / math.sqrt(D)

        dist[:, :G].fill_diagonal_(1e6)

        kernel = (-dist / temp).exp()

        row_sum = kernel.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        col_sum = kernel.sum(dim=-2, keepdim=True).clamp_min(1e-12)
        normalizer = (row_sum * col_sum).sqrt()
        normalized_kernel = kernel / normalizer

        pos_weight = normalized_kernel[:, G:]
        neg_weight = normalized_kernel[:, :G]

        neg_weight_sum = neg_weight.sum(dim=-1, keepdim=True)
        pos_weight_sum = pos_weight.sum(dim=-1, keepdim=True)

        pos_coeff = pos_weight * neg_weight_sum
        neg_coeff = neg_weight * pos_weight_sum

        V_pos = pos_coeff @ targets[G:]
        V_neg = neg_coeff @ targets[:G]

        V = V_pos - V_neg

        V_norm = torch.sqrt(torch.mean(V ** 2) + 1e-8)
        V = V / (V_norm + 1e-8)

        V_total = V_total + V

    return V_total


class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


def train(config: Config):
    """Train generator using drifting in ResNet50 feature space."""

    device = torch.device(config.device)
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)

    # Load CIFAR-10
    logger.info("Loading CIFAR-10...")
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(
        root=config.data_dir, train=True, download=True, transform=transform
    )

    all_images = torch.stack([dataset[i][0] for i in range(len(dataset))])
    all_labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    logger.info(f"Loaded {len(all_images)} images")

    # Create frozen feature encoder
    logger.info("Loading pretrained ResNet50 feature encoder...")
    feature_encoder = FeatureEncoder(freeze=True).to(device)
    feature_encoder.eval()
    logger.info(f"Feature encoder output dim: {feature_encoder.feature_dim}")

    # Pre-extract features
    logger.info("Pre-extracting features for all training images...")
    all_features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_images), 256), desc="Extracting features"):
            batch = all_images[i:i+256].to(device)
            feat = feature_encoder(batch)
            all_features.append(feat.cpu())
    all_features = torch.cat(all_features, dim=0)
    logger.info(f"Feature cache shape: {all_features.shape}")
    logger.info(f"Feature stats: mean={all_features.mean():.4f}, std={all_features.std():.4f}")

    # Create generator
    generator = ImageGenerator(
        noise_dim=config.noise_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        base_ch=config.base_ch,
    ).to(device)

    n_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    logger.info(f"Generator parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    # LR scheduler
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        progress = (step - config.warmup_steps) / (config.drift_steps - config.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = EMA(generator, config.ema_decay)

    # Training loop
    drift_losses = []
    drift_norms = []

    logger.info(f"Starting drifting model training for {config.drift_steps} steps...")
    logger.info(f"Temperatures: {config.temperatures}")

    pbar = tqdm(range(1, config.drift_steps + 1), desc="Training")
    for step in pbar:
        generator.train()

        classes = torch.randint(0, config.num_classes, (config.batch_size,), device=device)
        z = torch.randn(config.batch_size, config.noise_dim, device=device)
        gen_images = generator(z, classes)

        gen_feat = feature_encoder(gen_images)

        # Sample positive features
        pos_per_class = config.n_pos // config.num_classes
        pos_feat_list = []
        for c in range(config.num_classes):
            class_mask = all_labels == c
            class_idx = torch.where(class_mask)[0]
            sample_idx = class_idx[torch.randint(0, len(class_idx), (pos_per_class,))]
            pos_feat_list.append(all_features[sample_idx])
        pos_feat = torch.cat(pos_feat_list, dim=0).to(device)

        # Compute class-conditional drifting loss
        total_loss = 0.0
        total_drift_norm = 0.0
        count = 0

        for c in range(config.num_classes):
            gen_mask = classes == c
            pos_start = c * pos_per_class
            pos_end = (c + 1) * pos_per_class

            if gen_mask.sum() == 0:
                continue

            gen_c = gen_feat[gen_mask]
            pos_c = pos_feat[pos_start:pos_end]

            V = compute_drift_features(gen_c, pos_c, list(config.temperatures))

            target = (gen_c + V).detach()
            loss_c = F.mse_loss(gen_c, target)

            n_c = gen_mask.sum().item()
            total_loss = total_loss + loss_c * n_c
            total_drift_norm = total_drift_norm + V.norm(dim=-1).mean().item() * n_c
            count += n_c

        if count > 0:
            loss = total_loss / count
            avg_drift_norm = total_drift_norm / count
        else:
            loss = torch.tensor(0.0, device=device)
            avg_drift_norm = 0.0

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        ema.update()

        drift_losses.append(loss.item())
        drift_norms.append(avg_drift_norm)

        if step % config.log_every == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4e}",
                "drift": f"{avg_drift_norm:.4f}",
                "grad": f"{grad_norm:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        if step % config.save_every == 0 or step == 1:
            save_samples(
                generator, ema, feature_encoder, all_images, all_labels,
                step, save_dir, drift_losses, drift_norms, config, device
            )

            torch.save({
                "step": step,
                "generator": generator.state_dict(),
                "ema_shadow": ema.shadow,
                "optimizer": optimizer.state_dict(),
                "config": config,
            }, save_dir / "checkpoint.pt")

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Final loss: {drift_losses[-1]:.4e}")


@torch.no_grad()
def save_samples(
    generator, ema, feature_encoder, all_images, all_labels,
    step, save_dir, drift_losses, drift_norms, config, device
):
    """Generate and save sample images."""

    generator.eval()
    ema.apply()

    all_samples = []
    for c in range(config.num_classes):
        z = torch.randn(10, config.noise_dim, device=device)
        labels = torch.full((10,), c, dtype=torch.long, device=device)
        samples = generator(z, labels)
        samples = (samples.clamp(-1, 1) + 1) / 2
        all_samples.append(samples.cpu())

    all_samples = torch.cat(all_samples, dim=0)
    ema.restore()

    fig, axes = plt.subplots(10, 10, figsize=(12, 12))
    for i in range(10):
        for j in range(10):
            idx = i * 10 + j
            img = all_samples[idx].permute(1, 2, 0).numpy()
            axes[i, j].imshow(img)
            axes[i, j].axis("off")

    class_names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    for i, name in enumerate(class_names):
        axes[i, 0].set_ylabel(name, fontsize=10)

    plt.suptitle(f"Generated Samples (Step {step})")
    plt.tight_layout()
    plt.savefig(save_dir / f"samples_step{step}.png", dpi=100)
    plt.close()

    # Loss plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(drift_losses)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Drifting Loss")
    ax2.plot(drift_norms)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Drift Norm")
    ax2.set_title("Drift Magnitude")
    plt.tight_layout()
    plt.savefig(save_dir / f"losses_step{step}.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./cifar10_resnet50_v2")
    parser.add_argument("--drift_steps", type=int, default=100000)
    args = parser.parse_args()

    config = Config(save_dir=args.save_dir, drift_steps=args.drift_steps)
    train(config)


if __name__ == "__main__":
    main()
