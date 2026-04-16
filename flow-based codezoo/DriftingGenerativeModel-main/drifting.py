"""Drifting Field Computation for Generative Modeling.

Official implementation matching the Colab notebook from:
"Generative Modeling via Drifting" (arXiv:2602.04770)

The drifting field V is computed using batch-normalized kernel with:
1. Concatenate gen and pos samples
2. exp(-dist/temp) kernel
3. Bidirectional normalization: sqrt(row_sum * col_sum)
4. Cross-weighting for attraction and repulsion
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_drift(
    gen: torch.Tensor,
    pos: torch.Tensor,
    temp: float = 0.05,
    normalize_dist: bool = True,
) -> torch.Tensor:
    """
    Compute drift field V with attention-based kernel.

    This is the EXACT implementation from the official Colab notebook,
    with optional distance normalization for high-dimensional stability.

    Args:
        gen: Generated samples [G, D]
        pos: Data samples [P, D]
        temp: Temperature for kernel
        normalize_dist: If True, normalize distances by sqrt(D) for high-dim data

    Returns:
        V: Drift vectors [G, D]
    """
    targets = torch.cat([gen, pos], dim=0)
    G = gen.shape[0]
    D = gen.shape[1]

    dist = torch.cdist(gen, targets)
    # Normalize distance by sqrt(D) for high-dimensional stability
    if normalize_dist and D > 10:
        dist = dist / (D ** 0.5)
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


def drifting_loss(
    gen: torch.Tensor,
    pos: torch.Tensor,
    temp: float = 0.05,
) -> torch.Tensor:
    """Drifting loss: MSE(gen, stopgrad(gen + V))."""
    with torch.no_grad():
        V = compute_drift(gen, pos, temp)
        target = (gen + V).detach()
    return F.mse_loss(gen, target)


def compute_drift_multi_temp(
    gen: torch.Tensor,
    pos: torch.Tensor,
    temperatures: List[float] = [0.02, 0.05, 0.2],
    normalize_each: bool = True,
) -> torch.Tensor:
    """
    Compute drifting field with multiple temperatures.

    Each V is normalized before summing for balanced contribution.

    Args:
        gen: Generated samples [G, D]
        pos: Data samples [P, D]
        temperatures: List of temperature values
        normalize_each: Whether to normalize each V before summing

    Returns:
        V: Combined drift vectors [G, D]
    """
    V_total = torch.zeros_like(gen)

    for tau in temperatures:
        V_tau = compute_drift(gen, pos, tau)

        if normalize_each:
            # Normalize so E[||V||^2] ~ 1
            V_norm = torch.sqrt(torch.mean(V_tau ** 2) + 1e-8)
            V_tau = V_tau / (V_norm + 1e-8)

        V_total = V_total + V_tau

    return V_total


class DriftingField(nn.Module):
    """Compute the drifting field for generated samples.

    Official implementation matching the Colab notebook.
    """

    def __init__(
        self,
        temperature: float = 0.05,
        temperatures: Optional[List[float]] = None,
        use_multi_temp: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.temperatures = temperatures or [0.02, 0.05, 0.2]
        self.use_multi_temp = use_multi_temp

    def forward(
        self,
        gen_features: torch.Tensor,
        pos_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the drifting field.

        Args:
            gen_features: Generated sample features (N_gen, D)
            pos_features: Positive sample features (N_pos, D)

        Returns:
            Drifting field vectors (N_gen, D)
        """
        if self.use_multi_temp:
            return compute_drift_multi_temp(
                gen_features, pos_features, self.temperatures
            )
        else:
            return compute_drift(gen_features, pos_features, self.temperature)


class DriftingLoss(nn.Module):
    """Compute the drifting loss for training.

    L = E[||f_theta(epsilon) - stopgrad(f_theta(epsilon) + V)||^2]
    """

    def __init__(
        self,
        feature_encoder: Optional[nn.Module] = None,
        temperature: float = 0.05,
        temperatures: Optional[List[float]] = None,
        use_multi_temp: bool = False,
        flatten_features: bool = True,
    ):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.temperature = temperature
        self.temperatures = temperatures or [0.02, 0.05, 0.2]
        self.use_multi_temp = use_multi_temp
        self.flatten_features = flatten_features

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from samples."""
        if self.feature_encoder is not None:
            feat = self.feature_encoder(x)
            if isinstance(feat, dict):
                # Use the last feature layer if dict is returned
                feat = list(feat.values())[-1]
        else:
            feat = x

        if self.flatten_features and feat.dim() > 2:
            feat = feat.flatten(start_dim=1)

        return feat

    def forward(
        self,
        generated: torch.Tensor,
        positive: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the drifting loss.

        Args:
            generated: Generated samples (B, ...)
            positive: Positive (real) samples (B_pos, ...)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Extract features
        gen_feat = self.get_features(generated)
        pos_feat = self.get_features(positive)

        # Compute drifting field
        if self.use_multi_temp:
            V = compute_drift_multi_temp(gen_feat, pos_feat, self.temperatures)
        else:
            V = compute_drift(gen_feat, pos_feat, self.temperature)

        # Target is generated feature + drift (with stop gradient)
        target = (gen_feat + V).detach()

        # MSE loss
        loss = F.mse_loss(gen_feat, target)

        # Metrics
        metrics = {
            "loss": loss.item(),
            "drift_norm": V.norm(dim=-1).mean().item(),
        }

        return loss, metrics


class ClassConditionalDriftingLoss(nn.Module):
    """Class-conditional drifting loss.

    For class-conditional generation, computes drifting loss per class.
    """

    def __init__(
        self,
        feature_encoder: Optional[nn.Module] = None,
        temperature: float = 0.05,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.temperature = temperature
        self.num_classes = num_classes

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from samples."""
        if self.feature_encoder is not None:
            feat = self.feature_encoder(x)
            if isinstance(feat, dict):
                feat = list(feat.values())[-1]
        else:
            feat = x

        if feat.dim() > 2:
            feat = feat.flatten(start_dim=1)

        return feat

    def forward(
        self,
        generated: torch.Tensor,
        labels_gen: torch.Tensor,
        positive: torch.Tensor,
        labels_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute class-conditional drifting loss.

        Args:
            generated: Generated samples (N, ...)
            labels_gen: Labels for generated samples (N,)
            positive: Positive samples (N_pos, ...)
            labels_pos: Labels for positive samples (N_pos,)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        device = generated.device

        # Extract features
        gen_feat = self.get_features(generated)
        pos_feat = self.get_features(positive)

        total_loss = 0.0
        total_drift_norm = 0.0
        count = 0

        # Compute loss per class
        unique_labels = labels_gen.unique()
        for c in unique_labels:
            mask_gen = labels_gen == c
            mask_pos = labels_pos == c

            if not mask_gen.any() or not mask_pos.any():
                continue

            gen_c = gen_feat[mask_gen]
            pos_c = pos_feat[mask_pos]

            # Compute drift
            V = compute_drift(gen_c, pos_c, self.temperature)

            # Loss
            target = (gen_c + V).detach()
            loss_c = F.mse_loss(gen_c, target)

            n_c = len(gen_c)
            total_loss = total_loss + loss_c * n_c
            total_drift_norm = total_drift_norm + V.norm(dim=-1).mean().item() * n_c
            count += n_c

        if count == 0:
            return torch.tensor(0.0, device=device), {"loss": 0.0, "drift_norm": 0.0}

        loss = total_loss / count

        metrics = {
            "loss": loss.item(),
            "drift_norm": total_drift_norm / count,
        }

        return loss, metrics


# Convenience function for toy 2D experiments
def drift_step_2d(
    points: torch.Tensor,
    target_dist: torch.Tensor,
    temperature: float = 0.05,
    step_size: float = 0.1,
) -> torch.Tensor:
    """Single drift step for 2D toy experiments.

    Args:
        points: Current points (N, 2)
        target_dist: Target distribution samples (M, 2)
        temperature: Temperature for V computation
        step_size: Step size for drift

    Returns:
        Updated points after one drift step
    """
    V = compute_drift(points, target_dist, temperature)
    return points + step_size * V


# Legacy aliases for backward compatibility
SimpleDriftingLoss = DriftingLoss
