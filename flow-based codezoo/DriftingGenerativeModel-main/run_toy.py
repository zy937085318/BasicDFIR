#!/usr/bin/env python3
"""
Official implementation of Drifting Models for 2D toy examples.
This is an EXACT port of the official Colab notebook from the paper authors.

Key differences from previous versions:
1. Uses exp(-dist/temp) kernel, not softmax
2. Bidirectional normalization: sqrt(row_sum * col_sum)
3. Single temperature (0.05), not multi-temperature
4. Noise dimension = 32 (not 2!)
5. MLP with 3 hidden layers and SiLU activation
"""

import argparse
import math
import logging
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    This is the EXACT implementation from the official Colab notebook.

    Args:
        gen: Generated samples [G, D]
        pos: Data samples [P, D]
        temp: Temperature for kernel

    Returns:
        V: Drift vectors [G, D]
    """
    targets = torch.cat([gen, pos], dim=0)
    G = gen.shape[0]

    dist = torch.cdist(gen, targets)
    dist[:, :G].fill_diagonal_(1e6)  # mask self
    kernel = (-dist / temp).exp()  # unnormalized kernel

    # Normalize along both dimensions
    normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True)
    normalizer = normalizer.clamp_min(1e-12).sqrt()
    normalized_kernel = kernel / normalizer

    # Cross-weighting
    pos_coeff = normalized_kernel[:, G:] * normalized_kernel[:, :G].sum(dim=-1, keepdim=True)
    pos_V = pos_coeff @ targets[G:]
    neg_coeff = normalized_kernel[:, :G] * normalized_kernel[:, G:].sum(dim=-1, keepdim=True)
    neg_V = neg_coeff @ targets[:G]

    return pos_V - neg_V


def drifting_loss(gen: torch.Tensor, pos: torch.Tensor, compute_drift_fn):
    """Drifting loss: MSE(gen, stopgrad(gen + V))."""
    with torch.no_grad():
        V = compute_drift_fn(gen, pos)
        target = (gen + V).detach()
    return F.mse_loss(gen, target)


# ============================================================
# Dataset Samplers (from official Colab)
# ============================================================

def sample_checkerboard(n: int, noise: float = 0.05, seed: int = None) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    b = torch.randint(0, 2, (n,), generator=g)
    i = torch.randint(0, 2, (n,), generator=g) * 2 + b
    j = torch.randint(0, 2, (n,), generator=g) * 2 + b
    u = torch.rand(n, generator=g)
    v = torch.rand(n, generator=g)
    pts = torch.stack([i + u, j + v], dim=1) - 2.0
    pts = pts / 2.0
    if noise > 0:
        pts = pts + noise * torch.randn(pts.shape, generator=g)
    return pts


def sample_swiss_roll(n: int, noise: float = 0.03, seed: int = None) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    u = torch.rand(n, generator=g)
    t = 0.5 * math.pi + 4.0 * math.pi * u
    pts = torch.stack([t * torch.cos(t), t * torch.sin(t)], dim=1)
    pts = pts / (pts.abs().max() + 1e-8)
    if noise > 0:
        pts = pts + noise * torch.randn(pts.shape, generator=g)
    return pts


def sample_moons(n: int, noise: float = 0.05, seed: int = None) -> torch.Tensor:
    """Two interleaving half circles (moons) dataset."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    n_samples_out = n // 2
    n_samples_in = n - n_samples_out

    outer_circ_x = torch.cos(torch.linspace(0, math.pi, n_samples_out))
    outer_circ_y = torch.sin(torch.linspace(0, math.pi, n_samples_out))
    inner_circ_x = 1 - torch.cos(torch.linspace(0, math.pi, n_samples_in))
    inner_circ_y = 1 - torch.sin(torch.linspace(0, math.pi, n_samples_in)) - 0.5

    X = torch.cat([
        torch.stack([outer_circ_x, outer_circ_y], dim=1),
        torch.stack([inner_circ_x, inner_circ_y], dim=1),
    ], dim=0)

    # Normalize to [-1, 1]
    X = X - X.mean(dim=0, keepdim=True)
    X = X / (X.abs().max() + 1e-8)

    if noise > 0:
        X = X + noise * torch.randn(X.shape, generator=g)

    # Shuffle
    idx = torch.randperm(n, generator=g)
    return X[idx]


# ============================================================
# MLP Generator (from official Colab)
# ============================================================

class MLP(nn.Module):
    """MLP: noise -> output. 3 hidden layers with SiLU."""
    def __init__(self, in_dim=32, hidden=256, out_dim=2):
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
# Training Loop (from official Colab)
# ============================================================

def train_toy(
    sampler,
    steps: int = 2000,
    data_batch_size: int = 2048,
    gen_batch_size: int = 2048,
    lr: float = 1e-3,
    temp: float = 0.05,
    in_dim: int = 32,
    hidden: int = 256,
    device: str = "cuda",
    save_dir: str = "./toy_outputs_official",
    log_every: int = 100,
    visualize: bool = True,
    seed: int = 42,
    dataset_name: str = "toy",
):
    """Train drifting model. Returns model and loss history."""
    device = torch.device(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    model = MLP(in_dim=in_dim, hidden=hidden, out_dim=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info(f"Training on {dataset_name}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Temperature: {temp}, LR: {lr}, Steps: {steps}")
    logger.info(f"Batch sizes: data={data_batch_size}, gen={gen_batch_size}")
    logger.info(f"Input noise dim: {in_dim}, Hidden dim: {hidden}")

    loss_history = []
    drift_norm_history = []
    generated_history = []
    history_steps = []
    ema = None

    pbar = tqdm(range(1, steps + 1))
    for step in pbar:
        pos = sampler(data_batch_size).to(device)
        gen = model(torch.randn(gen_batch_size, in_dim, device=device))

        # Compute drift for logging
        with torch.no_grad():
            V = compute_drift(gen, pos, temp=temp)
            drift_norm = V.norm(dim=-1).mean().item()

        loss = drifting_loss(gen, pos, compute_drift_fn=partial(compute_drift, temp=temp))

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        drift_norm_history.append(drift_norm)
        ema = loss.item() if ema is None else 0.96 * ema + 0.04 * loss.item()
        pbar.set_postfix(loss=f"{ema:.2e}")

        if step % log_every == 0:
            logger.info(f"Step {step} | Loss: {loss.item():.4e} | EMA: {ema:.4e} | Drift Norm: {drift_norm:.4f}")

        # Save snapshots
        if step in [1, steps // 4, steps // 2, 3 * steps // 4, steps]:
            with torch.no_grad():
                samples = model(torch.randn(5000, in_dim, device=device)).cpu()
                generated_history.append(samples)
                history_steps.append(step)

    # Final samples
    logger.info("Generating final samples...")
    model.eval()
    with torch.no_grad():
        final_samples = model(torch.randn(5000, in_dim, device=device)).cpu()

    # Save results
    logger.info(f"Saving results to {save_dir}...")
    torch.save(model.state_dict(), save_dir / "generator.pt")
    np.save(save_dir / "final_samples.npy", final_samples.numpy())
    np.save(save_dir / "losses.npy", {"total": np.array(loss_history), "drift_norm": np.array(drift_norm_history)})

    # Visualization
    if visualize:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            logger.info("Creating visualizations...")

            # Get target data for comparison
            target_data = sampler(5000).numpy()

            # Final samples comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.scatter(target_data[:, 0], target_data[:, 1], s=2, alpha=0.3, c='blue')
            ax1.set_title('Target Distribution')
            ax1.set_aspect('equal')
            ax1.axis('off')

            ax2.scatter(final_samples[:, 0], final_samples[:, 1], s=2, alpha=0.3, c='orange')
            ax2.set_title('Generated Samples')
            ax2.set_aspect('equal')
            ax2.axis('off')

            plt.tight_layout()
            plt.savefig(save_dir / "samples.png", dpi=150, bbox_inches='tight')
            plt.close()

            # Training progress
            n_snapshots = len(generated_history)
            fig, axes = plt.subplots(1, n_snapshots + 1, figsize=(3 * (n_snapshots + 1), 3))

            axes[0].scatter(target_data[:, 0], target_data[:, 1], s=1, alpha=0.3, c='blue')
            axes[0].set_title('Target')
            axes[0].set_aspect('equal')
            axes[0].axis('off')

            for i, (samples, step) in enumerate(zip(generated_history, history_steps)):
                axes[i + 1].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.3, c='orange')
                axes[i + 1].set_title(f'Step {step}')
                axes[i + 1].set_aspect('equal')
                axes[i + 1].axis('off')

            plt.tight_layout()
            plt.savefig(save_dir / "progress.png", dpi=150, bbox_inches='tight')
            plt.close()

            # Loss curve
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(loss_history, alpha=0.7)
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title(f'{dataset_name} Loss Curve')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / "losses.png", dpi=150, bbox_inches='tight')
            plt.close()

            # Drift field visualization
            with torch.no_grad():
                test_gen = model(torch.randn(100, in_dim, device=device))
                test_pos = sampler(500).to(device)
                test_V = compute_drift(test_gen, test_pos, temp=temp)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(test_pos.cpu()[:, 0], test_pos.cpu()[:, 1], s=5, alpha=0.3, c='blue', label='Data')
            ax.scatter(test_gen.cpu()[:, 0], test_gen.cpu()[:, 1], s=30, c='orange', label='Generated')
            ax.quiver(test_gen.cpu()[:, 0], test_gen.cpu()[:, 1],
                     test_V.cpu()[:, 0], test_V.cpu()[:, 1],
                     scale=3, color='black', alpha=0.7, width=0.004)
            ax.legend()
            ax.set_title('Drift Vectors')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / "drift_field.png", dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"Visualizations saved to {save_dir}")

        except Exception as e:
            logger.warning(f"Could not create visualizations: {e}")

    # Print summary
    logger.info("=" * 50)
    logger.info("Training Complete (Official Implementation)!")
    logger.info(f"Final loss: {loss_history[-1]:.4e}")
    logger.info(f"Final drift norm: {drift_norm_history[-1]:.4f}")
    logger.info(f"Results saved to: {save_dir}")
    logger.info("=" * 50)

    return model, loss_history


def main():
    parser = argparse.ArgumentParser(description="Train Drifting Model (Official) on 2D toy data")
    parser.add_argument(
        "--dataset",
        type=str,
        default="swiss_roll",
        choices=["swiss_roll", "checkerboard", "moons"],
        help="Toy dataset name",
    )
    parser.add_argument("--steps", type=int, default=2000, help="Number of training steps")
    parser.add_argument("--data_batch_size", type=int, default=2048, help="Data batch size")
    parser.add_argument("--gen_batch_size", type=int, default=2048, help="Generated batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--temp", type=float, default=0.05, help="Temperature for kernel")
    parser.add_argument("--in_dim", type=int, default=32, help="Input noise dimension")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--save_dir", type=str, default="./toy_outputs_official", help="Output directory")
    parser.add_argument("--no_viz", action="store_true", help="Disable visualization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Select sampler
    samplers = {
        "swiss_roll": sample_swiss_roll,
        "checkerboard": sample_checkerboard,
        "moons": sample_moons,
    }
    sampler = samplers[args.dataset]

    train_toy(
        sampler=sampler,
        steps=args.steps,
        data_batch_size=args.data_batch_size,
        gen_batch_size=args.gen_batch_size,
        lr=args.lr,
        temp=args.temp,
        in_dim=args.in_dim,
        hidden=args.hidden,
        device=args.device,
        save_dir=args.save_dir,
        visualize=not args.no_viz,
        seed=args.seed,
        dataset_name=args.dataset,
    )


if __name__ == "__main__":
    main()
