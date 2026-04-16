"""Dataset utilities for Drifting Models.

Provides data loading for:
- ImageNet (raw images or pre-computed latents)
- Toy 2D datasets for testing
"""

from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np


class ToyDataset2D(Dataset):
    """Toy 2D datasets for testing and visualization."""

    def __init__(
        self,
        name: str = "swiss_roll",
        n_samples: int = 10000,
        noise: float = 0.0,
        seed: int = 42,
    ):
        super().__init__()
        self.name = name
        self.n_samples = n_samples
        self.noise = noise

        np.random.seed(seed)

        if name == "swiss_roll":
            self.data = self._make_swiss_roll()
        elif name == "checkerboard":
            self.data = self._make_checkerboard()
        elif name == "circles":
            self.data = self._make_circles()
        elif name == "moons":
            self.data = self._make_moons()
        elif name == "gaussian_mixture":
            self.data = self._make_gaussian_mixture()
        else:
            raise ValueError(f"Unknown dataset: {name}")

        # Normalize to [-1, 1]
        self.data = (self.data - self.data.mean(axis=0)) / self.data.std(axis=0)
        self.data = torch.from_numpy(self.data).float()

    def _make_swiss_roll(self) -> np.ndarray:
        """Generate Swiss roll dataset."""
        t = 1.5 * np.pi * (1 + 2 * np.random.rand(self.n_samples))
        x = t * np.cos(t)
        y = t * np.sin(t)
        data = np.stack([x, y], axis=1)
        data += self.noise * np.random.randn(*data.shape)
        return data

    def _make_checkerboard(self) -> np.ndarray:
        """Generate checkerboard dataset."""
        x1 = np.random.rand(self.n_samples) * 4 - 2
        x2 = np.random.rand(self.n_samples) - np.random.randint(0, 2, self.n_samples) * 2
        x2 += (np.floor(x1) % 2).astype(np.float32)
        data = np.stack([x1, x2], axis=1) * 2
        data += self.noise * np.random.randn(*data.shape)
        return data

    def _make_circles(self) -> np.ndarray:
        """Generate concentric circles dataset."""
        n_inner = self.n_samples // 2
        n_outer = self.n_samples - n_inner

        # Inner circle
        t_inner = 2 * np.pi * np.random.rand(n_inner)
        r_inner = 0.5 + self.noise * np.random.randn(n_inner)
        x_inner = r_inner * np.cos(t_inner)
        y_inner = r_inner * np.sin(t_inner)

        # Outer circle
        t_outer = 2 * np.pi * np.random.rand(n_outer)
        r_outer = 1.5 + self.noise * np.random.randn(n_outer)
        x_outer = r_outer * np.cos(t_outer)
        y_outer = r_outer * np.sin(t_outer)

        x = np.concatenate([x_inner, x_outer])
        y = np.concatenate([y_inner, y_outer])
        return np.stack([x, y], axis=1)

    def _make_moons(self) -> np.ndarray:
        """Generate two moons dataset."""
        n_upper = self.n_samples // 2
        n_lower = self.n_samples - n_upper

        # Upper moon
        t_upper = np.pi * np.random.rand(n_upper)
        x_upper = np.cos(t_upper)
        y_upper = np.sin(t_upper)

        # Lower moon
        t_lower = np.pi * np.random.rand(n_lower)
        x_lower = 1 - np.cos(t_lower)
        y_lower = -np.sin(t_lower) - 0.5

        x = np.concatenate([x_upper, x_lower])
        y = np.concatenate([y_upper, y_lower])
        data = np.stack([x, y], axis=1)
        data += self.noise * np.random.randn(*data.shape)
        return data

    def _make_gaussian_mixture(self, n_components: int = 8) -> np.ndarray:
        """Generate Gaussian mixture dataset."""
        samples_per_component = self.n_samples // n_components
        data = []

        for i in range(n_components):
            angle = 2 * np.pi * i / n_components
            center = np.array([2 * np.cos(angle), 2 * np.sin(angle)])
            samples = center + 0.2 * np.random.randn(samples_per_component, 2)
            data.append(samples)

        data = np.concatenate(data, axis=0)
        # Add remaining samples
        if data.shape[0] < self.n_samples:
            extra = self._make_gaussian_mixture(n_components)[:self.n_samples - data.shape[0]]
            data = np.concatenate([data, extra], axis=0)

        return data

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.data[idx], 0  # No class labels for toy data


class ClassBalancedSampler(Sampler):
    """Sampler that ensures balanced class sampling per batch.

    For drifting models, we need to sample N_c classes per batch,
    with N_neg generated samples per class.
    """

    def __init__(
        self,
        labels: torch.Tensor,
        n_classes_per_batch: int = 64,
        n_samples_per_class: int = 64,
    ):
        self.labels = labels
        self.n_classes = len(torch.unique(labels))
        self.n_classes_per_batch = min(n_classes_per_batch, self.n_classes)
        self.n_samples_per_class = n_samples_per_class

        # Build index for each class
        self.class_indices = {}
        for c in range(self.n_classes):
            self.class_indices[c] = torch.where(labels == c)[0].tolist()

        self.batch_size = self.n_classes_per_batch * self.n_samples_per_class

    def __iter__(self):
        # Generate batches
        all_classes = list(range(self.n_classes))

        while True:
            # Sample classes for this batch
            batch_classes = np.random.choice(
                all_classes, self.n_classes_per_batch, replace=False
            )

            batch_indices = []
            for c in batch_classes:
                # Sample indices for this class (with replacement if needed)
                class_idx = self.class_indices[c]
                if len(class_idx) >= self.n_samples_per_class:
                    sampled = np.random.choice(
                        class_idx, self.n_samples_per_class, replace=False
                    )
                else:
                    sampled = np.random.choice(
                        class_idx, self.n_samples_per_class, replace=True
                    )
                batch_indices.extend(sampled.tolist())

            yield from batch_indices

    def __len__(self) -> int:
        return len(self.labels)


class LatentDataset(Dataset):
    """Dataset for pre-computed latent representations.

    Latents are typically computed using a VAE (e.g., SD-VAE) and stored
    as numpy arrays or tensors.
    """

    def __init__(
        self,
        latent_path: str,
        label_path: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__()

        # Load latents
        if latent_path.endswith(".npy"):
            self.latents = torch.from_numpy(np.load(latent_path)).float()
        elif latent_path.endswith(".pt"):
            self.latents = torch.load(latent_path).float()
        else:
            raise ValueError(f"Unknown latent format: {latent_path}")

        # Load labels
        if label_path is not None:
            if label_path.endswith(".npy"):
                self.labels = torch.from_numpy(np.load(label_path)).long()
            elif label_path.endswith(".pt"):
                self.labels = torch.load(label_path).long()
            else:
                raise ValueError(f"Unknown label format: {label_path}")
        else:
            self.labels = torch.zeros(len(self.latents), dtype=torch.long)

        self.transform = transform

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        latent = self.latents[idx]
        label = self.labels[idx].item()

        if self.transform is not None:
            latent = self.transform(latent)

        return latent, label


class ImageNetLatentDataset(Dataset):
    """ImageNet dataset with pre-computed VAE latents.

    Expects latents to be stored per-class in directories.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        import os
        from pathlib import Path

        self.root_dir = Path(root_dir) / split
        self.transform = transform

        # Find all class directories
        self.class_dirs = sorted([
            d for d in self.root_dir.iterdir() if d.is_dir()
        ])
        self.num_classes = len(self.class_dirs)

        # Build index
        self.samples = []
        self.labels = []

        for class_idx, class_dir in enumerate(self.class_dirs):
            latent_files = list(class_dir.glob("*.npy")) + list(class_dir.glob("*.pt"))
            for f in latent_files:
                self.samples.append(str(f))
                self.labels.append(class_idx)

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.samples[idx]
        label = self.labels[idx].item()

        if path.endswith(".npy"):
            latent = torch.from_numpy(np.load(path)).float()
        else:
            latent = torch.load(path).float()

        if self.transform is not None:
            latent = self.transform(latent)

        return latent, label


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
    class_balanced: bool = False,
    n_classes_per_batch: int = 64,
    n_samples_per_class: int = 64,
) -> DataLoader:
    """Create a dataloader with optional class-balanced sampling.

    Args:
        dataset: The dataset
        batch_size: Batch size (ignored if class_balanced=True)
        num_workers: Number of worker processes
        shuffle: Whether to shuffle (ignored if class_balanced=True)
        pin_memory: Whether to pin memory
        drop_last: Whether to drop the last incomplete batch
        class_balanced: Whether to use class-balanced sampling
        n_classes_per_batch: Number of classes per batch (if class_balanced)
        n_samples_per_class: Samples per class (if class_balanced)

    Returns:
        DataLoader instance
    """
    if class_balanced and hasattr(dataset, "labels"):
        sampler = ClassBalancedSampler(
            dataset.labels,
            n_classes_per_batch=n_classes_per_batch,
            n_samples_per_class=n_samples_per_class,
        )
        return DataLoader(
            dataset,
            batch_size=n_classes_per_batch * n_samples_per_class,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
