import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
from typing import Dict, List, Optional, Tuple
from basicsr.utils.registry import LOSS_REGISTRY
import torchvision
from .drifting import compute_drift, compute_drift_multi_temp

@LOSS_REGISTRY.register()
class DriftingLoss(nn.Module):
    def __init__(self, feature_encoder='resnet18', temperature: float = 1,
        temperatures: Optional[List[float]] = None, use_multi_temp: bool = False,
        flatten_features: bool = True, loss_weight=1.0, reduction='mean'):
        super(DriftingLoss, self).__init__()
        self.loss_weight = loss_weight
        if feature_encoder == 'resnet18':
            resnet = torchvision.models.resnet18(pretrained=True)
            self.feature_encoder = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
                resnet.avgpool,
            )
        else:
            self.feature_encoder = None
        self.temperature = temperature
        self.temperatures = temperatures or [0.02, 0.05, 0.2]
        self.use_multi_temp = use_multi_temp
        self.flatten_features = flatten_features
        self.reduction = reduction

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

    def forward(self, generated, positive, is_disc=False):
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
        loss = F.mse_loss(gen_feat, target, reduction=self.reduction)
        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight

