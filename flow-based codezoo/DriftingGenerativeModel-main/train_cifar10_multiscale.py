"""
Drifting Model Training for CIFAR-10 with Multi-Scale Features.

Based on "Generative Modeling via Drifting" (arXiv:2602.04770)

Key: Multi-scale feature extraction from pretrained ResNet18.
- Extract features from layer1, layer2, layer3, layer4
- Compute drifting loss at each scale
- Combine losses

This matches the original paper's approach more closely than single-scale.
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

    # Drifting
    temperatures: Tuple[float, ...] = (0.02, 0.05, 0.2)

    # Logging
    save_dir: str = "./cifar10_multiscale"
    log_every: int = 100
    save_every: int = 2000
    seed: int = 42
    device: str = "cuda"


class ResBlock(nn.Module):
    """Residual block with class conditioning."""

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

        emb_out = self.emb_proj(emb)[:, :, None, None]
        scale, shift = emb_out.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift

        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)


class ImageGenerator(nn.Module):
    """Generator that outputs 32x32 RGB images."""

    def __init__(
        self,
        noise_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 10,
        base_ch: int = 128,
    ):
        super().__init__()

        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.base_ch = base_ch

        self.class_emb = nn.Embedding(num_classes, hidden_dim)
        self.proj = nn.Linear(noise_dim + hidden_dim, base_ch * 4 * 4 * 4)

        self.block1 = ResBlock(base_ch * 4, base_ch * 4, hidden_dim, upsample=True)
        self.block2 = ResBlock(base_ch * 4, base_ch * 2, hidden_dim, upsample=True)
        self.block3 = ResBlock(base_ch * 2, base_ch, hidden_dim, upsample=True)

        self.out_norm = nn.BatchNorm2d(base_ch)
        self.out_conv = nn.Conv2d(base_ch, 3, 3, padding=1)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        c_emb = self.class_emb(labels)
        h = torch.cat([z, c_emb], dim=1)
        h = self.proj(h)
        h = h.view(-1, self.base_ch * 4, 4, 4)

        h = self.block1(h, c_emb)
        h = self.block2(h, c_emb)
        h = self.block3(h, c_emb)

        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        h = torch.tanh(h)

        return h


class MultiScaleFeatureEncoder(nn.Module):
    """Multi-scale feature encoder using pretrained ResNet18.

    Extracts features from multiple stages:
    - layer1: 64 channels, 8x8 (for 32x32 input upscaled to 224)
    - layer2: 128 channels, 4x4
    - layer3: 256 channels, 2x2
    - layer4: 512 channels, 1x1

    This matches the original paper's multi-scale approach.
    """

    def __init__(self, freeze: bool = True):
        super().__init__()

        resnet = models.resnet18(pretrained=True)

        # Early layers
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        # Multi-scale feature layers
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

        # Feature dimensions at each scale
        self.feature_dims = [64, 128, 256, 512]
        self.scale_names = ['layer1', 'layer2', 'layer3', 'layer4']

        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()

        # ImageNet normalization
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Images in [-1, 1] range, shape [B, 3, H, W]

        Returns:
            Dict of features at each scale, each flattened to [B, D_i]
        """
        # Normalize
        x = (x + 1) / 2
        x = (x - self.mean) / self.std

        # Resize to 224x224
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # Extract multi-scale features
        x = self.stem(x)

        features = {}

        x = self.layer1(x)
        features['layer1'] = x.flatten(start_dim=1)  # [B, 64 * 56 * 56] -> too large, use avgpool

        x = self.layer2(x)
        features['layer2'] = x.flatten(start_dim=1)

        x = self.layer3(x)
        features['layer3'] = x.flatten(start_dim=1)

        x = self.layer4(x)
        features['layer4'] = x.flatten(start_dim=1)

        # Use adaptive avg pool to reduce spatial dimensions
        # This gives us fixed-size features regardless of input size
        return features

    def forward_pooled(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features with global average pooling at each scale.

        This reduces memory and makes features comparable across scales.
        """
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        x = self.stem(x)

        features = {}

        x = self.layer1(x)
        features['layer1'] = F.adaptive_avg_pool2d(x, 1).flatten(1)  # [B, 64]

        x = self.layer2(x)
        features['layer2'] = F.adaptive_avg_pool2d(x, 1).flatten(1)  # [B, 128]

        x = self.layer3(x)
        features['layer3'] = F.adaptive_avg_pool2d(x, 1).flatten(1)  # [B, 256]

        x = self.layer4(x)
        features['layer4'] = F.adaptive_avg_pool2d(x, 1).flatten(1)  # [B, 512]

        return features

    def train(self, mode: bool = True):
        super().train(mode)
        # Always keep in eval mode for frozen encoder
        self.stem.eval()
        self.layer1.eval()
        self.layer2.eval()
        self.layer3.eval()
        self.layer4.eval()
        return self


def compute_drift_single_scale(
    gen_feat: torch.Tensor,
    pos_feat: torch.Tensor,
    temperatures: List[float],
) -> torch.Tensor:
    """Compute drifting field for a single scale."""
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


def compute_multiscale_drift_loss(
    gen_features: Dict[str, torch.Tensor],
    pos_features: Dict[str, torch.Tensor],
    temperatures: List[float],
    scale_weights: Dict[str, float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute multi-scale drifting loss.

    Args:
        gen_features: Dict of generated features at each scale
        pos_features: Dict of positive features at each scale
        temperatures: Temperature values for kernel
        scale_weights: Optional weights for each scale (default: equal weights)

    Returns:
        Total loss and metrics dict
    """
    if scale_weights is None:
        scale_weights = {k: 1.0 for k in gen_features.keys()}

    total_loss = 0.0
    total_drift_norm = 0.0
    metrics = {}

    for scale_name in gen_features.keys():
        gen_feat = gen_features[scale_name]
        pos_feat = pos_features[scale_name]

        # Compute drift at this scale
        V = compute_drift_single_scale(gen_feat, pos_feat, temperatures)

        # Loss at this scale
        target = (gen_feat + V).detach()
        loss_scale = F.mse_loss(gen_feat, target)

        # Weighted contribution
        weight = scale_weights.get(scale_name, 1.0)
        total_loss = total_loss + weight * loss_scale

        drift_norm = V.norm(dim=-1).mean().item()
        total_drift_norm += drift_norm

        metrics[f'loss_{scale_name}'] = loss_scale.item()
        metrics[f'drift_{scale_name}'] = drift_norm

    metrics['total_drift_norm'] = total_drift_norm / len(gen_features)

    return total_loss, metrics


class EMA:
    """Exponential Moving Average."""

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
    """Train with multi-scale drifting loss."""

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

    # Multi-scale feature encoder
    logger.info("Loading multi-scale ResNet18 encoder...")
    feature_encoder = MultiScaleFeatureEncoder(freeze=True).to(device)
    feature_encoder.eval()
    logger.info(f"Feature dimensions: {feature_encoder.feature_dims}")

    # Pre-extract multi-scale features
    logger.info("Pre-extracting multi-scale features for all images...")
    all_features = {name: [] for name in feature_encoder.scale_names}

    with torch.no_grad():
        for i in tqdm(range(0, len(all_images), 256), desc="Extracting features"):
            batch = all_images[i:i+256].to(device)
            feats = feature_encoder.forward_pooled(batch)
            for name in feature_encoder.scale_names:
                all_features[name].append(feats[name].cpu())

    for name in feature_encoder.scale_names:
        all_features[name] = torch.cat(all_features[name], dim=0)
        logger.info(f"  {name}: {all_features[name].shape}")

    # Generator
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

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        progress = (step - config.warmup_steps) / (config.drift_steps - config.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = EMA(generator, config.ema_decay)

    # Training
    losses_history = []

    logger.info(f"Starting multi-scale training for {config.drift_steps} steps...")
    logger.info(f"Temperatures: {config.temperatures}")
    logger.info("Computing drifting loss at 4 scales: layer1(64D), layer2(128D), layer3(256D), layer4(512D)")

    pbar = tqdm(range(1, config.drift_steps + 1), desc="Training")
    for step in pbar:
        generator.train()

        # Sample classes
        classes = torch.randint(0, config.num_classes, (config.batch_size,), device=device)

        # Generate images
        z = torch.randn(config.batch_size, config.noise_dim, device=device)
        gen_images = generator(z, classes)

        # Extract multi-scale features from generated images
        gen_features = feature_encoder.forward_pooled(gen_images)

        # Sample positive features per class
        pos_per_class = config.n_pos // config.num_classes
        pos_features = {name: [] for name in feature_encoder.scale_names}

        for c in range(config.num_classes):
            class_mask = all_labels == c
            class_idx = torch.where(class_mask)[0]
            sample_idx = class_idx[torch.randint(0, len(class_idx), (pos_per_class,))]
            for name in feature_encoder.scale_names:
                pos_features[name].append(all_features[name][sample_idx])

        for name in feature_encoder.scale_names:
            pos_features[name] = torch.cat(pos_features[name], dim=0).to(device)

        # Compute multi-scale class-conditional loss
        total_loss = 0.0
        total_metrics = {}
        count = 0

        for c in range(config.num_classes):
            gen_mask = classes == c
            pos_start = c * pos_per_class
            pos_end = (c + 1) * pos_per_class

            if gen_mask.sum() == 0:
                continue

            gen_feats_c = {name: gen_features[name][gen_mask] for name in feature_encoder.scale_names}
            pos_feats_c = {name: pos_features[name][pos_start:pos_end] for name in feature_encoder.scale_names}

            loss_c, metrics_c = compute_multiscale_drift_loss(
                gen_feats_c, pos_feats_c, list(config.temperatures)
            )

            n_c = gen_mask.sum().item()
            total_loss = total_loss + loss_c * n_c
            count += n_c

            for k, v in metrics_c.items():
                total_metrics[k] = total_metrics.get(k, 0) + v * n_c

        if count > 0:
            loss = total_loss / count
            for k in total_metrics:
                total_metrics[k] /= count
        else:
            loss = torch.tensor(0.0, device=device)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        ema.update()

        losses_history.append(loss.item())

        if step % config.log_every == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4e}",
                "drift": f"{total_metrics.get('total_drift_norm', 0):.2f}",
                "grad": f"{grad_norm:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        if step % config.save_every == 0 or step == 1:
            save_samples(
                generator, ema, feature_encoder, all_images, all_labels,
                step, save_dir, losses_history, config, device
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
    logger.info(f"Final loss: {losses_history[-1]:.4e}")
    logger.info("=" * 60)


@torch.no_grad()
def save_samples(generator, ema, feature_encoder, all_images, all_labels,
                 step, save_dir, losses_history, config, device):
    """Save sample images."""

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

    # Sample grid
    fig, axes = plt.subplots(10, 10, figsize=(12, 12))
    for i in range(10):
        for j in range(10):
            axes[i, j].imshow(all_samples[i * 10 + j].permute(1, 2, 0).numpy())
            axes[i, j].axis("off")
    plt.suptitle(f"Generated Samples (Step {step}) - Multi-Scale")
    plt.tight_layout()
    plt.savefig(save_dir / f"samples_step{step}.png", dpi=150)
    plt.close()

    # Loss plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses_history, alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Multi-Scale Drifting Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"losses_step{step}.png", dpi=150)
    plt.close()

    # Comparison
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(10):
        class_mask = all_labels == i
        real_idx = torch.where(class_mask)[0][0]
        real_img = (all_images[real_idx] + 1) / 2
        axes[0, i].imshow(real_img.permute(1, 2, 0).numpy())
        axes[0, i].axis("off")
        axes[0, i].set_title(f"C{i}", fontsize=8)
        axes[1, i].imshow(all_samples[i * 10].permute(1, 2, 0).numpy())
        axes[1, i].axis("off")
    plt.suptitle(f"Real (top) vs Generated (bottom) - Step {step}")
    plt.tight_layout()
    plt.savefig(save_dir / f"comparison_step{step}.png", dpi=150)
    plt.close()

    logger.info(f"Saved samples at step {step}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./cifar10_multiscale")
    parser.add_argument("--drift_steps", type=int, default=100000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = Config(
        save_dir=args.save_dir,
        drift_steps=args.drift_steps,
        device=args.device,
    )

    train(config)


if __name__ == "__main__":
    main()
