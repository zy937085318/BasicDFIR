"""VAE utilities for latent space encoding/decoding.

Drifting models can work in latent space using a pretrained VAE
(e.g., Stable Diffusion VAE). This module provides utilities for:
- Encoding images to latent space
- Decoding latents back to images
- Pre-computing and caching latents

The VAE reduces 256x256x3 images to 32x32x4 latents (8x compression).
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class SimpleVAE(nn.Module):
    """Simple VAE for testing (not production quality).

    For production, use a pretrained VAE like:
    - Stable Diffusion VAE
    - VQGAN
    - Custom trained VAE
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        hidden_channels: int = 128,
        compression_factor: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.compression_factor = compression_factor

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels * 4, latent_channels * 2, 3, stride=1, padding=1),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels * 4, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels, in_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode images to latent distribution parameters."""
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=1)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to images."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    def get_latent(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Get latent representation of an image."""
        mu, log_var = self.encode(x)
        if deterministic:
            return mu
        return self.reparameterize(mu, log_var)


class VAEWrapper:
    """Wrapper for pretrained VAE (e.g., from diffusers).

    Example usage with Stable Diffusion VAE:
    ```python
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    wrapper = VAEWrapper(vae)
    ```
    """

    def __init__(
        self,
        vae: nn.Module,
        scaling_factor: float = 0.18215,  # SD VAE default
        device: torch.device = None,
    ):
        self.vae = vae
        self.scaling_factor = scaling_factor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae.to(self.device)
        self.vae.eval()

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latents.

        Args:
            images: (B, 3, H, W) in range [-1, 1]

        Returns:
            latents: (B, 4, H//8, W//8)
        """
        images = images.to(self.device)

        # Handle different VAE interfaces
        if hasattr(self.vae, "encode"):
            # Diffusers-style VAE
            latent_dist = self.vae.encode(images)
            if hasattr(latent_dist, "latent_dist"):
                latents = latent_dist.latent_dist.sample()
            else:
                latents = latent_dist.sample()
        else:
            # Custom VAE
            latents = self.vae.get_latent(images)

        # Scale latents
        latents = latents * self.scaling_factor

        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images.

        Args:
            latents: (B, 4, H//8, W//8)

        Returns:
            images: (B, 3, H, W) in range [-1, 1]
        """
        latents = latents.to(self.device)

        # Unscale latents
        latents = latents / self.scaling_factor

        # Decode
        if hasattr(self.vae, "decode"):
            images = self.vae.decode(latents).sample
        else:
            images = self.vae.decode(latents)

        return images


class LatentCache:
    """Cache for pre-computed latent representations.

    Pre-computing and caching latents significantly speeds up
    training for large datasets like ImageNet.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, image_path: str) -> Path:
        """Get cache path for an image."""
        # Use hash of image path as cache filename
        import hashlib
        hash_str = hashlib.md5(image_path.encode()).hexdigest()
        return self.cache_dir / f"{hash_str}.npy"

    def exists(self, image_path: str) -> bool:
        """Check if latent is cached."""
        return self.get_cache_path(image_path).exists()

    def load(self, image_path: str) -> Optional[np.ndarray]:
        """Load cached latent."""
        cache_path = self.get_cache_path(image_path)
        if cache_path.exists():
            return np.load(cache_path)
        return None

    def save(self, image_path: str, latent: np.ndarray):
        """Save latent to cache."""
        cache_path = self.get_cache_path(image_path)
        np.save(cache_path, latent)

    @staticmethod
    def precompute_latents(
        vae_wrapper: VAEWrapper,
        dataset,
        output_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """Pre-compute and save latents for entire dataset.

        Args:
            vae_wrapper: VAE wrapper for encoding
            dataset: PyTorch dataset returning (image, label, path)
            output_dir: Directory to save latents
            batch_size: Batch size for encoding
            num_workers: Number of data loading workers
        """
        from torch.utils.data import DataLoader

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        all_latents = []
        all_labels = []

        for batch in tqdm(loader, desc="Pre-computing latents"):
            images, labels = batch[:2]
            latents = vae_wrapper.encode(images)

            all_latents.append(latents.cpu().numpy())
            all_labels.append(labels.numpy())

        # Concatenate and save
        all_latents = np.concatenate(all_latents, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        np.save(output_dir / "latents.npy", all_latents)
        np.save(output_dir / "labels.npy", all_labels)

        print(f"Saved {len(all_latents)} latents to {output_dir}")


def load_sd_vae(model_id: str = "stabilityai/sd-vae-ft-mse") -> VAEWrapper:
    """Load Stable Diffusion VAE from HuggingFace.

    Args:
        model_id: HuggingFace model ID

    Returns:
        VAEWrapper instance
    """
    try:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(model_id)
        return VAEWrapper(vae)
    except ImportError:
        raise ImportError("diffusers is required for loading SD VAE. Install with: pip install diffusers")


class LatentNormalization:
    """Normalization utilities for latent space.

    Different VAEs have different latent statistics.
    This class helps normalize latents to a standard distribution.
    """

    def __init__(self, mean: torch.Tensor = None, std: torch.Tensor = None):
        self.mean = mean
        self.std = std

    @torch.no_grad()
    def fit(self, latents: torch.Tensor):
        """Compute mean and std from latents."""
        self.mean = latents.mean(dim=(0, 2, 3), keepdim=True)
        self.std = latents.std(dim=(0, 2, 3), keepdim=True)

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalize latents to zero mean and unit variance."""
        if self.mean is None or self.std is None:
            return latents
        return (latents - self.mean.to(latents.device)) / self.std.to(latents.device).clamp(min=1e-6)

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        """Denormalize latents back to original distribution."""
        if self.mean is None or self.std is None:
            return latents
        return latents * self.std.to(latents.device) + self.mean.to(latents.device)

    def save(self, path: str):
        """Save normalization statistics."""
        torch.save({"mean": self.mean, "std": self.std}, path)

    def load(self, path: str):
        """Load normalization statistics."""
        state = torch.load(path)
        self.mean = state["mean"]
        self.std = state["std"]
