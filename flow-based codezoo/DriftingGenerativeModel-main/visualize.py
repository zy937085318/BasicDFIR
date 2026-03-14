"""Visualization utilities for Drifting Models.

Provides functions for:
- 2D toy data visualization
- Sample grid visualization
- Training progress plots
- Drifting field visualization
"""

from typing import List, Optional, Tuple

import numpy as np
import torch


def plot_2d_samples(
    real_data: torch.Tensor,
    generated_data: torch.Tensor,
    title: str = "Drifting Model Samples",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot 2D real and generated samples side by side.

    Args:
        real_data: Real data samples (N, 2)
        generated_data: Generated samples (M, 2)
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Real data
        real_np = real_data.cpu().numpy() if isinstance(real_data, torch.Tensor) else real_data
        axes[0].scatter(real_np[:, 0], real_np[:, 1], alpha=0.5, s=5)
        axes[0].set_title("Real Data")
        axes[0].set_aspect("equal")
        axes[0].grid(True, alpha=0.3)

        # Generated data
        gen_np = generated_data.cpu().numpy() if isinstance(generated_data, torch.Tensor) else generated_data
        axes[1].scatter(gen_np[:, 0], gen_np[:, 1], alpha=0.5, s=5)
        axes[1].set_title("Generated Data")
        axes[1].set_aspect("equal")
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    except ImportError:
        print("matplotlib not available for visualization")


def plot_training_progress(
    real_data: torch.Tensor,
    generated_history: List[torch.Tensor],
    steps: List[int],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
):
    """Plot training progress showing generated samples at different steps.

    Args:
        real_data: Real data samples
        generated_history: List of generated samples at different steps
        steps: Step numbers corresponding to generated_history
        save_path: Path to save figure
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt

        n_plots = len(generated_history) + 1
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)

        # Real data
        real_np = real_data.cpu().numpy() if isinstance(real_data, torch.Tensor) else real_data
        axes[0].scatter(real_np[:, 0], real_np[:, 1], alpha=0.5, s=5)
        axes[0].set_title("Real Data")
        axes[0].set_aspect("equal")

        # Generated at each step
        for i, (gen, step) in enumerate(zip(generated_history, steps)):
            gen_np = gen.cpu().numpy() if isinstance(gen, torch.Tensor) else gen
            axes[i + 1].scatter(gen_np[:, 0], gen_np[:, 1], alpha=0.5, s=5)
            axes[i + 1].set_title(f"Step {step}")
            axes[i + 1].set_aspect("equal")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    except ImportError:
        print("matplotlib not available for visualization")


def plot_drifting_field(
    samples: torch.Tensor,
    drift_vectors: torch.Tensor,
    real_data: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
    scale: float = 1.0,
):
    """Visualize the drifting field as a quiver plot.

    Args:
        samples: Sample positions (N, 2)
        drift_vectors: Drift vectors at each sample (N, 2)
        real_data: Real data samples for reference (optional)
        save_path: Path to save figure
        figsize: Figure size
        scale: Arrow scale factor
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        samples_np = samples.cpu().numpy() if isinstance(samples, torch.Tensor) else samples
        drift_np = drift_vectors.cpu().numpy() if isinstance(drift_vectors, torch.Tensor) else drift_vectors

        # Plot real data if provided
        if real_data is not None:
            real_np = real_data.cpu().numpy() if isinstance(real_data, torch.Tensor) else real_data
            ax.scatter(real_np[:, 0], real_np[:, 1], alpha=0.3, s=5, c="blue", label="Real")

        # Plot samples
        ax.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.5, s=10, c="red", label="Generated")

        # Plot drift vectors
        ax.quiver(
            samples_np[:, 0], samples_np[:, 1],
            drift_np[:, 0] * scale, drift_np[:, 1] * scale,
            alpha=0.5, color="green", angles="xy", scale_units="xy", scale=1,
        )

        ax.set_title("Drifting Field Visualization")
        ax.set_aspect("equal")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    except ImportError:
        print("matplotlib not available for visualization")


def make_image_grid(
    images: torch.Tensor,
    nrow: int = 8,
    normalize: bool = True,
    value_range: Tuple[float, float] = (-1, 1),
) -> torch.Tensor:
    """Create a grid of images.

    Args:
        images: Images tensor (N, C, H, W)
        nrow: Number of images per row
        normalize: Whether to normalize to [0, 1]
        value_range: Input value range for normalization

    Returns:
        Grid image (C, H, W)
    """
    N, C, H, W = images.shape
    ncol = (N + nrow - 1) // nrow

    # Pad if necessary
    if N < nrow * ncol:
        padding = torch.zeros(nrow * ncol - N, C, H, W, device=images.device)
        images = torch.cat([images, padding], dim=0)

    # Reshape to grid
    images = images.reshape(ncol, nrow, C, H, W)
    images = images.permute(2, 0, 3, 1, 4)  # (C, ncol, H, nrow, W)
    images = images.reshape(C, ncol * H, nrow * W)

    if normalize:
        min_val, max_val = value_range
        images = (images - min_val) / (max_val - min_val)
        images = images.clamp(0, 1)

    return images


def save_image_grid(
    images: torch.Tensor,
    path: str,
    nrow: int = 8,
    normalize: bool = True,
):
    """Save images as a grid.

    Args:
        images: Images tensor (N, C, H, W)
        path: Output path
        nrow: Number of images per row
        normalize: Whether to normalize to [0, 1]
    """
    try:
        from torchvision.utils import save_image
        save_image(images, path, nrow=nrow, normalize=normalize)
    except ImportError:
        # Fallback using PIL
        try:
            from PIL import Image

            grid = make_image_grid(images, nrow, normalize)
            grid_np = (grid.cpu().numpy() * 255).astype(np.uint8)

            if grid_np.shape[0] == 1:
                grid_np = grid_np[0]
            else:
                grid_np = grid_np.transpose(1, 2, 0)

            Image.fromarray(grid_np).save(path)
        except ImportError:
            print("Neither torchvision nor PIL available for saving images")


def plot_loss_curves(
    losses: dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """Plot training loss curves.

    Args:
        losses: Dictionary of loss name -> list of values
        save_path: Path to save figure
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        for name, values in losses.items():
            ax.plot(values, label=name, alpha=0.8)

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    except ImportError:
        print("matplotlib not available for visualization")


def visualize_cfg_interpolation(
    model: torch.nn.Module,
    noise: torch.Tensor,
    class_label: int,
    cfg_scales: List[float],
    save_path: Optional[str] = None,
    device: str = "cuda",
):
    """Visualize samples at different CFG scales.

    Args:
        model: Generator model
        noise: Input noise (1, C, H, W)
        class_label: Class label
        cfg_scales: List of CFG scales to visualize
        save_path: Path to save figure
        device: Device
    """
    try:
        import matplotlib.pyplot as plt

        samples = []
        model.eval()

        with torch.no_grad():
            for cfg_scale in cfg_scales:
                cfg_alpha = torch.full((1,), cfg_scale, device=device)
                labels = torch.tensor([class_label], device=device)
                sample = model(noise.to(device), labels, cfg_alpha)
                samples.append(sample.cpu())

        samples = torch.cat(samples, dim=0)

        # Plot
        n = len(cfg_scales)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))

        for i, (ax, scale) in enumerate(zip(axes, cfg_scales)):
            img = samples[i].permute(1, 2, 0).numpy()
            img = (img + 1) / 2  # Normalize to [0, 1]
            img = img.clip(0, 1)

            if img.shape[-1] == 1:
                img = img[:, :, 0]
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)

            ax.set_title(f"CFG = {scale}")
            ax.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    except ImportError:
        print("matplotlib not available for visualization")
