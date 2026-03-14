"""Sampling script for Drifting Models.

Drifting models use one-step inference - a single forward pass
generates samples, unlike diffusion/flow models that require
iterative refinement.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import Config
from models import DiT
from cfg import CFGSampler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DriftingSampler:
    """Sampler for Drifting Models.

    Key advantage: One-step inference (1 NFE) compared to
    multi-step diffusion/flow models.
    """

    def __init__(
        self,
        generator: nn.Module,
        device: torch.device,
        use_ema: bool = True,
        ema_state: Optional[dict] = None,
    ):
        self.generator = generator.to(device)
        self.device = device
        self.use_ema = use_ema

        # Load EMA weights if provided
        if use_ema and ema_state is not None:
            self._load_ema(ema_state)

        self.generator.eval()

    def _load_ema(self, ema_state: dict):
        """Load EMA parameters into the generator."""
        shadow = ema_state["shadow"]
        for name, param in self.generator.named_parameters():
            if name in shadow:
                param.data.copy_(shadow[name])

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        class_labels: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        batch_size: int = 256,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate samples.

        Args:
            num_samples: Total number of samples to generate
            class_labels: Optional class labels. If None, randomly sampled.
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance)
            batch_size: Batch size for generation
            seed: Random seed for reproducibility

        Returns:
            Tuple of (samples, labels)
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Get model config
        img_size = self.generator.img_size
        in_channels = self.generator.in_channels
        num_classes = self.generator.num_classes

        all_samples = []
        all_labels = []

        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Generating samples"):
            current_batch_size = min(batch_size, num_samples - i * batch_size)

            # Sample noise
            noise = torch.randn(
                current_batch_size, in_channels, img_size, img_size,
                device=self.device
            )

            # Sample or use provided class labels
            if class_labels is not None:
                batch_labels = class_labels[i * batch_size:(i + 1) * batch_size]
            else:
                batch_labels = torch.randint(
                    0, num_classes, (current_batch_size,), device=self.device
                )

            # CFG scale
            cfg_alpha = torch.full(
                (current_batch_size,), cfg_scale, device=self.device
            )

            # One-step generation
            samples = self.generator(noise, batch_labels, cfg_alpha)

            all_samples.append(samples.cpu())
            all_labels.append(batch_labels.cpu())

        samples = torch.cat(all_samples, dim=0)[:num_samples]
        labels = torch.cat(all_labels, dim=0)[:num_samples]

        return samples, labels

    @torch.no_grad()
    def sample_class(
        self,
        class_idx: int,
        num_samples: int,
        cfg_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate samples for a specific class.

        Args:
            class_idx: Class index
            num_samples: Number of samples to generate
            cfg_scale: Classifier-free guidance scale
            seed: Random seed

        Returns:
            Generated samples
        """
        class_labels = torch.full((num_samples,), class_idx, device=self.device)
        samples, _ = self.sample(
            num_samples=num_samples,
            class_labels=class_labels,
            cfg_scale=cfg_scale,
            seed=seed,
        )
        return samples

    @torch.no_grad()
    def sample_interpolation(
        self,
        class1: int,
        class2: int,
        num_steps: int = 10,
        cfg_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate interpolation between two classes using shared noise.

        Args:
            class1: First class index
            class2: Second class index
            num_steps: Number of interpolation steps
            cfg_scale: CFG scale
            seed: Random seed

        Returns:
            Interpolated samples (num_steps, C, H, W)
        """
        if seed is not None:
            torch.manual_seed(seed)

        img_size = self.generator.img_size
        in_channels = self.generator.in_channels

        # Use same noise for all samples
        noise = torch.randn(
            1, in_channels, img_size, img_size, device=self.device
        )
        noise = noise.repeat(num_steps, 1, 1, 1)

        # Interpolate class embeddings
        # Note: This requires access to the class embedding layer
        embed1 = self.generator.class_embed.weight[class1]  # (hidden_size,)
        embed2 = self.generator.class_embed.weight[class2]

        samples = []
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            # We'll generate with both classes and manually interpolate
            # Since we can't directly interpolate embeddings in the forward pass

            cfg_alpha = torch.full((1,), cfg_scale, device=self.device)

            # Generate for class1
            labels1 = torch.tensor([class1], device=self.device)
            sample1 = self.generator(noise[:1], labels1, cfg_alpha)

            # Generate for class2
            labels2 = torch.tensor([class2], device=self.device)
            sample2 = self.generator(noise[:1], labels2, cfg_alpha)

            # Interpolate in output space
            sample = (1 - alpha) * sample1 + alpha * sample2
            samples.append(sample)

        return torch.cat(samples, dim=0)


def load_model(
    checkpoint_path: str,
    device: torch.device,
    use_ema: bool = True,
) -> DriftingSampler:
    """Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        use_ema: Whether to use EMA weights

    Returns:
        DriftingSampler instance
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)

    # Get config
    config = state.get("config", Config())

    # Create generator
    generator = DiT(
        img_size=config.model.image_size,
        patch_size=config.model.patch_size,
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        hidden_size=config.model.hidden_size,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        num_classes=config.model.num_classes,
    )

    # Load weights
    if use_ema and "ema" in state:
        logger.info("Using EMA weights")
        ema_state = state["ema"]
    else:
        logger.info("Using regular weights")
        generator.load_state_dict(state["generator"])
        ema_state = None

    sampler = DriftingSampler(
        generator=generator,
        device=device,
        use_ema=use_ema,
        ema_state=ema_state,
    )

    return sampler


def save_samples(
    samples: torch.Tensor,
    labels: torch.Tensor,
    output_dir: str,
    prefix: str = "sample",
):
    """Save generated samples as images and numpy array.

    Args:
        samples: Generated samples (N, C, H, W)
        labels: Class labels (N,)
        output_dir: Output directory
        prefix: Filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as numpy
    np.save(output_dir / f"{prefix}_samples.npy", samples.numpy())
    np.save(output_dir / f"{prefix}_labels.npy", labels.numpy())

    # Save individual images
    try:
        from torchvision.utils import save_image

        # Normalize to [0, 1]
        samples_norm = (samples + 1) / 2
        samples_norm = samples_norm.clamp(0, 1)

        # Save grid
        from torchvision.utils import make_grid
        grid = make_grid(samples_norm[:64], nrow=8, normalize=False)
        save_image(grid, output_dir / f"{prefix}_grid.png")

        logger.info(f"Saved samples to {output_dir}")

    except ImportError:
        logger.warning("torchvision not available, skipping image save")


def compute_fid(
    samples: torch.Tensor,
    real_stats_path: str,
) -> float:
    """Compute FID score.

    Args:
        samples: Generated samples (N, C, H, W)
        real_stats_path: Path to pre-computed real dataset statistics

    Returns:
        FID score
    """
    try:
        from pytorch_fid import fid_score
        # This would require saving samples to disk and using pytorch-fid
        logger.warning("FID computation not yet implemented")
        return -1.0
    except ImportError:
        logger.warning("pytorch-fid not available")
        return -1.0


def main():
    parser = argparse.ArgumentParser(description="Sample from Drifting Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--output_dir", type=str, default="./samples", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=50000, help="Number of samples")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--no_ema", action="store_true", help="Don't use EMA weights")
    parser.add_argument("--class_idx", type=int, default=None, help="Specific class to sample")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    sampler = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        use_ema=not args.no_ema,
    )

    # Generate samples
    if args.class_idx is not None:
        logger.info(f"Sampling class {args.class_idx}")
        samples = sampler.sample_class(
            class_idx=args.class_idx,
            num_samples=args.num_samples,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
        )
        labels = torch.full((args.num_samples,), args.class_idx)
    else:
        logger.info(f"Sampling {args.num_samples} samples with CFG scale {args.cfg_scale}")
        samples, labels = sampler.sample(
            num_samples=args.num_samples,
            cfg_scale=args.cfg_scale,
            batch_size=args.batch_size,
            seed=args.seed,
        )

    # Save samples
    save_samples(samples, labels, args.output_dir)

    logger.info("Sampling complete!")
    logger.info(f"Generated {samples.shape[0]} samples")
    logger.info(f"Sample shape: {samples.shape}")


if __name__ == "__main__":
    main()
