#!/usr/bin/env python3
"""
MNIST training for Drifting Models.
Based on the official Colab notebook implementation.

This script trains a drifting model on MNIST images (28x28 = 784 dims).
"""

import argparse
import os
import logging
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Core: Compute Drift V (EXACT official implementation)
# ============================================================

def compute_drift(gen: torch.Tensor, pos: torch.Tensor, temp: float = 0.05):
    """
    Compute drift field V with attention-based kernel.
    Official implementation from the paper.

    For high-dimensional data, we normalize distances by sqrt(D) to keep
    kernel values in a reasonable range.
    """
    targets = torch.cat([gen, pos], dim=0)
    G = gen.shape[0]
    D = gen.shape[1]

    dist = torch.cdist(gen, targets)
    # Normalize distance by sqrt(D) for high-dimensional stability
    dist = dist / (D ** 0.5)
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


def drifting_loss(gen: torch.Tensor, pos: torch.Tensor, temp: float = 0.05):
    """Drifting loss: MSE(gen, stopgrad(gen + V))."""
    with torch.no_grad():
        V = compute_drift(gen, pos, temp)
        target = (gen + V).detach()
    return F.mse_loss(gen, target)


# ============================================================
# MLP Generator for MNIST
# ============================================================

class MLP(nn.Module):
    """MLP: noise -> flattened image. Based on official Colab."""
    def __init__(self, in_dim=64, hidden=512, out_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z):
        return self.net(z)


# ============================================================
# Training
# ============================================================

def train_mnist(
    data_dir: str = "/scratch/users/minghuix/data/mnist",
    steps: int = 8000,
    batch_size: int = 512,
    lr: float = 1e-3,
    temp: float = 0.1,  # Higher temp for higher-dim data
    in_dim: int = 64,
    hidden: int = 512,
    device: str = "cuda",
    save_dir: str = "./mnist_outputs",
    log_every: int = 100,
    save_every: int = 2000,
    seed: int = 42,
):
    """Train drifting model on MNIST."""
    device = torch.device(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)

    # Load MNIST
    logger.info(f"Loading MNIST from {data_dir}...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Scale to [-1, 1]
    ])

    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    # Preload all data to GPU for faster training
    logger.info("Preloading data to GPU...")
    all_images = []
    for img, _ in tqdm(dataset, desc="Loading"):
        all_images.append(img.flatten())
    all_images = torch.stack(all_images).to(device)
    logger.info(f"Loaded {len(all_images)} images, shape: {all_images.shape}")

    # Create model
    out_dim = 28 * 28  # 784
    model = MLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Temperature: {temp}, LR: {lr}, Steps: {steps}")
    logger.info(f"Batch size: {batch_size}, Input dim: {in_dim}, Hidden: {hidden}")

    loss_history = []
    drift_norm_history = []
    ema = None

    pbar = tqdm(range(1, steps + 1))
    for step in pbar:
        # Sample random batch of real images
        idx = torch.randint(0, len(all_images), (batch_size,), device=device)
        pos = all_images[idx]

        # Generate samples
        gen = model(torch.randn(batch_size, in_dim, device=device))

        # Compute drift for logging
        with torch.no_grad():
            V = compute_drift(gen, pos, temp=temp)
            drift_norm = V.norm(dim=-1).mean().item()

        # Compute loss
        loss = drifting_loss(gen, pos, temp=temp)

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        drift_norm_history.append(drift_norm)
        ema = loss.item() if ema is None else 0.96 * ema + 0.04 * loss.item()
        pbar.set_postfix(loss=f"{ema:.2e}", drift=f"{drift_norm:.4f}")

        if step % log_every == 0:
            logger.info(f"Step {step} | Loss: {loss.item():.4e} | EMA: {ema:.4e} | Drift: {drift_norm:.4f}")

        # Save checkpoint
        if step % save_every == 0 or step == steps:
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimizer': opt.state_dict(),
                'loss_history': loss_history,
                'drift_norm_history': drift_norm_history,
            }, save_dir / f"checkpoint_{step}.pt")
            logger.info(f"Saved checkpoint at step {step}")

    # Generate final samples
    logger.info("Generating final samples...")
    model.eval()
    with torch.no_grad():
        # Generate 100 samples for visualization
        samples = model(torch.randn(100, in_dim, device=device))
        samples = samples.view(-1, 1, 28, 28).cpu()
        # Denormalize to [0, 1]
        samples = (samples + 1) / 2
        samples = samples.clamp(0, 1)

    # Save results
    np.save(save_dir / "final_samples.npy", samples.numpy())
    np.save(save_dir / "losses.npy", {
        "loss": np.array(loss_history),
        "drift_norm": np.array(drift_norm_history),
    })
    torch.save(model.state_dict(), save_dir / "generator.pt")

    # Create visualizations
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        logger.info("Creating visualizations...")

        # Sample grid
        fig, axes = plt.subplots(10, 10, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i, 0], cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_dir / "samples.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Loss curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(loss_history, alpha=0.7)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        ax2.plot(drift_norm_history, alpha=0.7, color='orange')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Drift Norm')
        ax2.set_title('Drift Norm')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / "losses.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Compare with real samples
        fig, axes = plt.subplots(2, 10, figsize=(15, 3))

        # Real samples
        real_idx = torch.randint(0, len(all_images), (10,))
        real_samples = all_images[real_idx].view(-1, 1, 28, 28).cpu()
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
        plt.savefig(save_dir / "comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Visualizations saved to {save_dir}")

    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")

    # Print summary
    logger.info("=" * 50)
    logger.info("Training Complete!")
    logger.info(f"Final loss: {loss_history[-1]:.4e}")
    logger.info(f"Final drift norm: {drift_norm_history[-1]:.4f}")
    logger.info(f"Results saved to: {save_dir}")
    logger.info("=" * 50)

    return model, loss_history


def main():
    parser = argparse.ArgumentParser(description="Train Drifting Model on MNIST")
    parser.add_argument("--data_dir", type=str, default="/scratch/users/minghuix/data/mnist")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--in_dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./mnist_outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_mnist(
        data_dir=args.data_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        temp=args.temp,
        in_dim=args.in_dim,
        hidden=args.hidden,
        device=args.device,
        save_dir=args.save_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
