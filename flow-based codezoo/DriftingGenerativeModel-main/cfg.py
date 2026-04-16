"""Classifier-Free Guidance for Drifting Models.

Implements CFG as described in the paper (Eq. 15-16):

Training:
    q̃(·|c) = (1-γ)q_θ(·|c) + γp_data(·|∅)

    The negative distribution is a mixture of conditional generated samples
    and unconditional real samples.

Inference:
    q_θ(·|c) = α·p_data(·|c) - (α-1)·p_data(·|∅)

    Where α = 1/(1-γ) ≥ 1 is the guidance scale.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_cfg_alpha(
    batch_size: int,
    alpha_min: float = 1.0,
    alpha_max: float = 4.0,
    power: float = 3.0,
    device: torch.device = None,
) -> torch.Tensor:
    """Sample CFG alpha values from power-law distribution.

    p(alpha) ∝ alpha^(-power)

    Args:
        batch_size: Number of alpha values to sample
        alpha_min: Minimum alpha value
        alpha_max: Maximum alpha value
        power: Power for the distribution (higher = more weight on small alpha)
        device: Device to place the tensor

    Returns:
        Alpha values of shape (batch_size,)
    """
    if device is None:
        device = torch.device("cpu")

    # Sample from uniform and transform using inverse CDF
    u = torch.rand(batch_size, device=device)

    if power == 1.0:
        # Special case: log-uniform
        log_alpha = torch.log(torch.tensor(alpha_min)) + u * (
            torch.log(torch.tensor(alpha_max)) - torch.log(torch.tensor(alpha_min))
        )
        alpha = torch.exp(log_alpha)
    else:
        # General power-law case
        # CDF: F(alpha) = (alpha^(1-power) - alpha_min^(1-power)) /
        #                 (alpha_max^(1-power) - alpha_min^(1-power))
        exp = 1 - power
        a_min_exp = alpha_min**exp
        a_max_exp = alpha_max**exp
        alpha = (a_min_exp + u * (a_max_exp - a_min_exp)) ** (1 / exp)

    return alpha


class CFGMixin:
    """Mixin class for classifier-free guidance in training."""

    def prepare_cfg_batch(
        self,
        generator: nn.Module,
        noise: torch.Tensor,
        class_labels: torch.Tensor,
        cfg_alpha: torch.Tensor,
        real_samples: torch.Tensor,
        n_uncond: int = 16,
        n_neg: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare batch with CFG-adjusted negative samples.

        The negative distribution is:
        q̃(·|c) = [(N_neg-1)q_θ(·|c) + N_uncond·w·p_data(·|∅)] / [(N_neg-1)+N_uncond·w]

        Args:
            generator: The generator model
            noise: Input noise (B, C, H, W)
            class_labels: Class labels (B,)
            cfg_alpha: CFG alpha values (B,) or scalar
            real_samples: Real samples for unconditional (N_uncond, C, H, W)
            n_uncond: Number of unconditional samples
            n_neg: Number of negative samples per class

        Returns:
            Tuple of (generated_samples, positive_samples, negative_samples)
        """
        B = noise.shape[0]
        device = noise.device

        # Generate conditional samples
        with torch.no_grad():
            generated = generator(noise, class_labels, cfg_alpha)

        # For CFG training, negative samples include:
        # 1. Conditional generated samples (weighted by N_neg - 1)
        # 2. Unconditional real samples (weighted by cfg_alpha - 1)

        # Sample unconditional real samples
        if n_uncond > 0 and real_samples is not None:
            uncond_idx = torch.randperm(real_samples.shape[0], device=device)[:n_uncond]
            uncond_samples = real_samples[uncond_idx]

            # Weight factor for unconditional samples
            # w = (alpha - 1) * N_uncond / (N_neg - 1)
            w = (cfg_alpha.mean() - 1) * n_uncond / max(n_neg - 1, 1)

            # Create weighted negative distribution
            # Replicate unconditional samples according to weight
            n_uncond_effective = int(w * n_uncond)
            if n_uncond_effective > 0:
                negative = torch.cat([
                    generated,
                    uncond_samples[:n_uncond_effective].repeat(
                        (B // n_uncond_effective + 1), 1, 1, 1
                    )[:B],
                ], dim=0)
            else:
                negative = generated
        else:
            negative = generated

        return generated, negative

    def compute_cfg_weight(
        self,
        cfg_alpha: torch.Tensor,
        n_uncond: int,
        n_neg: int,
    ) -> float:
        """Compute effective weight for unconditional samples in CFG."""
        # Eq. from paper: w such that effective mixture matches CFG formulation
        alpha_mean = cfg_alpha.mean().item()
        w = (alpha_mean - 1) / (1 + (n_neg - 1) / n_uncond)
        return w


class CFGSampler:
    """Sampling with classifier-free guidance.

    At inference, we use:
    x = model(z, c, alpha) where alpha is the guidance scale.

    The model is trained to produce samples that interpolate between
    conditional and unconditional distributions.
    """

    def __init__(
        self,
        generator: nn.Module,
        default_cfg_scale: float = 1.0,
    ):
        self.generator = generator
        self.default_cfg_scale = default_cfg_scale

    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        class_labels: torch.Tensor,
        cfg_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate samples with CFG.

        Args:
            noise: Input noise (B, C, H, W)
            class_labels: Class labels (B,)
            cfg_scale: Guidance scale (alpha). 1.0 = no guidance.

        Returns:
            Generated samples (B, C, H, W)
        """
        if cfg_scale is None:
            cfg_scale = self.default_cfg_scale

        B = noise.shape[0]
        device = noise.device

        # Create CFG scale tensor
        cfg_alpha = torch.full((B,), cfg_scale, device=device)

        # Single forward pass (Drifting models use one-step inference)
        samples = self.generator(noise, class_labels, cfg_alpha)

        return samples

    @torch.no_grad()
    def sample_batch(
        self,
        batch_size: int,
        img_size: int,
        in_channels: int,
        num_classes: int,
        cfg_scale: float = 1.0,
        device: torch.device = None,
        class_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of samples.

        Args:
            batch_size: Number of samples to generate
            img_size: Image size (height = width)
            in_channels: Number of input channels
            num_classes: Number of classes
            cfg_scale: Guidance scale
            device: Device to use
            class_labels: Optional class labels, random if None

        Returns:
            Tuple of (samples, class_labels)
        """
        if device is None:
            device = next(self.generator.parameters()).device

        # Sample noise
        noise = torch.randn(batch_size, in_channels, img_size, img_size, device=device)

        # Sample class labels if not provided
        if class_labels is None:
            class_labels = torch.randint(0, num_classes, (batch_size,), device=device)

        # Generate samples
        samples = self.sample(noise, class_labels, cfg_scale)

        return samples, class_labels


class GuidedDriftingField:
    """Compute drifting field with CFG-weighted negative samples.

    During training, the negative distribution is modified to include
    unconditional real samples, enabling CFG at inference time.
    """

    def __init__(
        self,
        drift_loss: nn.Module,
        n_uncond: int = 16,
    ):
        self.drift_loss = drift_loss
        self.n_uncond = n_uncond

    def compute_loss(
        self,
        generated: torch.Tensor,
        positive: torch.Tensor,
        uncond_positive: torch.Tensor,
        cfg_alpha: torch.Tensor,
        update_stats: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute drifting loss with CFG-weighted negatives.

        Args:
            generated: Generated conditional samples (B, C, H, W)
            positive: Positive samples with same class (B_pos, C, H, W)
            uncond_positive: Unconditional positive samples (N_uncond, C, H, W)
            cfg_alpha: CFG alpha values (B,)
            update_stats: Whether to update normalization statistics

        Returns:
            Tuple of (loss, metrics_dict)
        """
        B = generated.shape[0]
        device = generated.device

        # Compute weight for unconditional samples
        alpha_mean = cfg_alpha.mean().item()
        w = max(0, alpha_mean - 1)

        if w > 0 and self.n_uncond > 0:
            # Create weighted mixture of generated and unconditional samples
            # for the negative distribution
            n_uncond_weighted = int(w * self.n_uncond)
            if n_uncond_weighted > 0:
                # Sample unconditional positives
                idx = torch.randperm(uncond_positive.shape[0], device=device)
                uncond_subset = uncond_positive[idx[:n_uncond_weighted]]

                # Concatenate for negative samples
                negative = torch.cat([generated, uncond_subset], dim=0)
            else:
                negative = generated
        else:
            negative = generated

        # Compute loss (new API only takes generated and positive)
        # Note: For CFG, we could extend to use negative samples differently
        # but for now, the official implementation just uses gen and pos
        loss, metrics = self.drift_loss(
            generated=generated,
            positive=positive,
        )

        metrics["cfg_alpha_mean"] = alpha_mean
        metrics["cfg_weight"] = w

        return loss, metrics


def interpolate_cfg(
    model: nn.Module,
    noise: torch.Tensor,
    class_labels: torch.Tensor,
    cfg_scales: List[float],
) -> List[torch.Tensor]:
    """Generate samples at multiple CFG scales for visualization.

    Args:
        model: The generator model
        noise: Input noise (1, C, H, W) - same noise for all scales
        class_labels: Class labels (1,)
        cfg_scales: List of CFG scales to try

    Returns:
        List of generated samples at each scale
    """
    samples = []
    device = noise.device

    for cfg_scale in cfg_scales:
        cfg_alpha = torch.full((1,), cfg_scale, device=device)
        with torch.no_grad():
            sample = model(noise, class_labels, cfg_alpha)
        samples.append(sample)

    return samples
