#!/usr/bin/env python3
"""
MNIST training for Drifting Models with CNN Encoder.

Following the paper: compute drifting field V in feature space, not pixel space.
This should make the loss decrease properly as the distribution converges.

Architecture:
- CNN Encoder: 28x28 image → 64-dim feature
- CNN Decoder: 64-dim feature → 28x28 image
- MLP Generator: noise → 64-dim feature (then decode to image)
- V is computed in 64-dim feature space
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Core: Compute Drift V (official implementation)
# ============================================================

def compute_drift(gen: torch.Tensor, pos: torch.Tensor, temp: float = 0.05):
    """
    Compute drift field V in feature space.
    No distance normalization needed for low-dim features.
    """
    targets = torch.cat([gen, pos], dim=0)
    G = gen.shape[0]

    dist = torch.cdist(gen, targets)
    dist[:, :G].fill_diagonal_(1e6)  # mask self
    kernel = (-dist / temp).exp()

    normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True)
    normalizer = normalizer.clamp_min(1e-12).sqrt()
    normalized_kernel = kernel / normalizer

    pos_coeff = normalized_kernel[:, G:] * normalized_kernel[:, :G].sum(dim=-1, keepdim=True)
    pos_V = pos_coeff @ targets[G:]
    neg_coeff = normalized_kernel[:, :G] * normalized_kernel[:, G:].sum(dim=-1, keepdim=True)
    neg_V = neg_coeff @ targets[:G]

    return pos_V - neg_V


# ============================================================
# CNN Encoder/Decoder
# ============================================================

class CNNEncoder(nn.Module):
    """CNN encoder: 28x28 grayscale → feature_dim."""
    def __init__(self, feature_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            # 28x28 → 14x14
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            # 14x14 → 7x7
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            # 7x7 → 4x4
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            # 4x4 → 2x2
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
        )
        # 256 * 2 * 2 = 1024 → feature_dim
        self.fc = nn.Linear(256 * 2 * 2, feature_dim)

    def forward(self, x):
        # x: (B, 1, 28, 28)
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class CNNDecoder(nn.Module):
    """CNN decoder: feature_dim → 28x28 grayscale."""
    def __init__(self, feature_dim=64):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 128 * 7 * 7)
        self.deconv = nn.Sequential(
            # 7x7 → 14x14
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            # 14x14 → 28x28
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            # 28x28 → 28x28
            nn.Conv2d(32, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 128, 7, 7)
        return self.deconv(h)


# ============================================================
# MLP Generator (in feature space)
# ============================================================

class MLPGenerator(nn.Module):
    """MLP: noise → feature space."""
    def __init__(self, in_dim=32, hidden=256, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z):
        return self.net(z)


# ============================================================
# Training
# ============================================================

def train_mnist_with_encoder(
    data_dir: str = "/scratch/users/minghuix/data/mnist",
    steps: int = 8000,
    batch_size: int = 512,
    lr: float = 1e-3,
    temp: float = 0.05,
    in_dim: int = 32,
    hidden: int = 256,
    feature_dim: int = 64,
    ae_pretrain_steps: int = 2000,
    device: str = "cuda",
    save_dir: str = "./mnist_outputs_encoder",
    log_every: int = 100,
    seed: int = 42,
):
    """Train drifting model with CNN encoder in feature space."""
    device = torch.device(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)

    # Load MNIST
    logger.info(f"Loading MNIST from {data_dir}...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)

    # Preload to GPU
    logger.info("Preloading data to GPU...")
    all_images = torch.stack([dataset[i][0] for i in tqdm(range(len(dataset)), desc="Loading")])
    all_images = all_images.to(device)
    logger.info(f"Loaded {len(all_images)} images, shape: {all_images.shape}")

    # Create models
    encoder = CNNEncoder(feature_dim=feature_dim).to(device)
    decoder = CNNDecoder(feature_dim=feature_dim).to(device)
    generator = MLPGenerator(in_dim=in_dim, hidden=hidden, out_dim=feature_dim).to(device)

    logger.info(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    logger.info(f"Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")
    logger.info(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    logger.info(f"Feature dim: {feature_dim}, Temperature: {temp}")

    # ============================================================
    # Phase 1: Pretrain Autoencoder
    # ============================================================
    logger.info(f"Phase 1: Pretraining autoencoder for {ae_pretrain_steps} steps...")
    ae_opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

    ae_losses = []
    pbar = tqdm(range(1, ae_pretrain_steps + 1), desc="AE Pretrain")
    for step in pbar:
        idx = torch.randint(0, len(all_images), (batch_size,), device=device)
        x = all_images[idx]

        z = encoder(x)
        x_rec = decoder(z)
        ae_loss = F.mse_loss(x_rec, x)

        ae_opt.zero_grad()
        ae_loss.backward()
        ae_opt.step()

        ae_losses.append(ae_loss.item())
        pbar.set_postfix(loss=f"{ae_loss.item():.4e}")

    logger.info(f"AE pretrain done, final loss: {ae_losses[-1]:.4e}")

    # Freeze encoder (optional - can also fine-tune)
    # for p in encoder.parameters():
    #     p.requires_grad = False

    # Encode all images to feature space
    logger.info("Encoding all images to feature space...")
    with torch.no_grad():
        # Process in batches to avoid OOM
        all_features = []
        for i in tqdm(range(0, len(all_images), 1000), desc="Encoding"):
            batch = all_images[i:i+1000]
            features = encoder(batch)
            all_features.append(features)
        all_features = torch.cat(all_features, dim=0)

    # Normalize features: zero mean, then scale so pairwise distances are ~1
    # For temp=0.05 to work well, we want avg pairwise distance around 1
    feat_mean = all_features.mean(dim=0, keepdim=True)
    all_features = all_features - feat_mean

    # Compute current avg pairwise distance and scale
    with torch.no_grad():
        sample_idx = torch.randperm(len(all_features))[:500]
        sample_feats = all_features[sample_idx]
        dists = torch.cdist(sample_feats, sample_feats)
        mask = ~torch.eye(500, dtype=torch.bool, device=device)
        avg_dist = dists[mask].mean().item()

    # Scale to target distance (~1 for temp=0.05)
    target_dist = 1.0
    scale = target_dist / (avg_dist + 1e-8)
    all_features = all_features * scale
    feat_scale = scale

    logger.info(f"Feature shape: {all_features.shape}")
    logger.info(f"Original avg pairwise dist: {avg_dist:.4f}, scale: {scale:.4f}")

    # ============================================================
    # Phase 2: Train Generator with Drifting Loss in Feature Space
    # ============================================================
    logger.info(f"Phase 2: Training generator for {steps} steps...")
    gen_opt = torch.optim.Adam(generator.parameters(), lr=lr)

    loss_history = []
    drift_norm_history = []
    ema_loss = None

    pbar = tqdm(range(1, steps + 1), desc="Generator")
    for step in pbar:
        # Sample real features (already normalized)
        idx = torch.randint(0, len(all_features), (batch_size,), device=device)
        pos_features = all_features[idx]

        # Generate features and apply same normalization
        gen_features_raw = generator(torch.randn(batch_size, in_dim, device=device))
        gen_features = (gen_features_raw - feat_mean) * feat_scale

        # Compute drift in feature space
        with torch.no_grad():
            V = compute_drift(gen_features, pos_features, temp=temp)
            target = (gen_features + V).detach()
            drift_norm = V.norm(dim=-1).mean().item()

        # Drifting loss in feature space
        loss = F.mse_loss(gen_features, target)

        gen_opt.zero_grad()
        loss.backward()
        gen_opt.step()

        loss_history.append(loss.item())
        drift_norm_history.append(drift_norm)
        ema_loss = loss.item() if ema_loss is None else 0.96 * ema_loss + 0.04 * loss.item()
        pbar.set_postfix(loss=f"{ema_loss:.2e}", drift=f"{drift_norm:.4f}")

        if step % log_every == 0:
            logger.info(f"Step {step} | Loss: {loss.item():.4e} | EMA: {ema_loss:.4e} | Drift: {drift_norm:.4f}")

    # ============================================================
    # Generate and Visualize
    # ============================================================
    logger.info("Generating final samples...")
    generator.eval()
    decoder.eval()

    with torch.no_grad():
        # Generate features and decode to images
        gen_features = generator(torch.randn(100, in_dim, device=device))
        samples = decoder(gen_features)
        samples = samples.cpu()
        samples = (samples + 1) / 2
        samples = samples.clamp(0, 1)

    # Save results
    np.save(save_dir / "final_samples.npy", samples.numpy())
    np.save(save_dir / "losses.npy", {
        "loss": np.array(loss_history),
        "drift_norm": np.array(drift_norm_history),
        "ae_loss": np.array(ae_losses),
    })
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "generator": generator.state_dict(),
    }, save_dir / "models.pt")

    # Visualizations
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Samples grid
        fig, axes = plt.subplots(10, 10, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i, 0], cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_dir / "samples.png", dpi=150)
        plt.close()

        # Loss curves
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(ae_losses, alpha=0.7)
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('AE Pretrain Loss')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(loss_history, alpha=0.7)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Drifting Loss (Feature Space)')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(drift_norm_history, alpha=0.7, color='orange')
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('Drift Norm')
        axes[2].set_title('Drift Norm')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / "losses.png", dpi=150)
        plt.close()

        # Comparison: Real vs Generated
        fig, axes = plt.subplots(2, 10, figsize=(15, 3))

        # Real samples
        real_idx = torch.randint(0, len(all_images), (10,))
        real_samples = all_images[real_idx].cpu()
        real_samples = (real_samples + 1) / 2

        for i in range(10):
            axes[0, i].imshow(real_samples[i, 0], cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Real', fontsize=12)

        for i in range(10):
            axes[1, i].imshow(samples[i, 0], cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Generated', fontsize=12)

        plt.tight_layout()
        plt.savefig(save_dir / "comparison.png", dpi=150)
        plt.close()

        # AE reconstruction quality check
        with torch.no_grad():
            test_imgs = all_images[:10]
            test_features = encoder(test_imgs)
            test_rec = decoder(test_features)
            test_rec = (test_rec.cpu() + 1) / 2

        fig, axes = plt.subplots(2, 10, figsize=(15, 3))
        for i in range(10):
            axes[0, i].imshow((test_imgs[i, 0].cpu() + 1) / 2, cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=12)
            axes[1, i].imshow(test_rec[i, 0], cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('AE Recon', fontsize=12)

        plt.tight_layout()
        plt.savefig(save_dir / "ae_reconstruction.png", dpi=150)
        plt.close()

        logger.info(f"Visualizations saved to {save_dir}")

    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")

    # Summary
    logger.info("=" * 50)
    logger.info("Training Complete!")
    logger.info(f"AE final loss: {ae_losses[-1]:.4e}")
    logger.info(f"Drifting final loss: {loss_history[-1]:.4e}")
    logger.info(f"Final drift norm: {drift_norm_history[-1]:.4f}")
    logger.info(f"Results saved to: {save_dir}")
    logger.info("=" * 50)

    return generator, encoder, decoder, loss_history


def main():
    parser = argparse.ArgumentParser(description="Train Drifting Model on MNIST with CNN Encoder")
    parser.add_argument("--data_dir", type=str, default="/scratch/users/minghuix/data/mnist")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--in_dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--ae_pretrain_steps", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./mnist_outputs_encoder")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_mnist_with_encoder(
        data_dir=args.data_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        temp=args.temp,
        in_dim=args.in_dim,
        hidden=args.hidden,
        feature_dim=args.feature_dim,
        ae_pretrain_steps=args.ae_pretrain_steps,
        device=args.device,
        save_dir=args.save_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
