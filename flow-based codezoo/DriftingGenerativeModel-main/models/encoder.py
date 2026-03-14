"""Feature Encoder for Drifting Models.

Implements a ResNet-style encoder with:
- GroupNorm instead of BatchNorm
- Multi-scale feature extraction
- MAE-style architecture for representation learning
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNormAct(nn.Module):
    """GroupNorm followed by activation."""

    def __init__(
        self, num_channels: int, num_groups: int = 32, act: bool = True
    ):
        super().__init__()
        self.norm = nn.GroupNorm(
            min(num_groups, num_channels), num_channels
        )
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(x))


class ResBlock(nn.Module):
    """Residual block with GroupNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        num_groups: int = 32,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm1 = GroupNormAct(out_channels, num_groups)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        self.norm2 = GroupNormAct(out_channels, num_groups, act=False)

        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(min(num_groups, out_channels), out_channels),
            )
        else:
            self.skip = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.norm1(self.conv1(x))
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


class ResNetEncoder(nn.Module):
    """ResNet-style encoder with multi-scale feature extraction.

    This encoder extracts features at multiple spatial scales for
    computing the drifting field in feature space.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        channel_multipliers: List[int] = [1, 2, 4, 8],
        num_blocks_per_stage: int = 2,
        num_groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = len(channel_multipliers)

        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=1, padding=1, bias=False),
            GroupNormAct(base_channels, num_groups),
        )

        # Build stages
        self.stages = nn.ModuleList()
        in_ch = base_channels
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            stride = 2 if i > 0 else 1  # Downsample starting from stage 1
            stage = self._make_stage(
                in_ch, out_ch, num_blocks_per_stage, stride, num_groups
            )
            self.stages.append(stage)
            in_ch = out_ch

        # Channel dimensions at each stage
        self.stage_channels = [base_channels * m for m in channel_multipliers]

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        num_groups: int,
    ) -> nn.Sequential:
        """Create a stage with multiple residual blocks."""
        blocks = [ResBlock(in_channels, out_channels, stride, num_groups)]
        for _ in range(num_blocks - 1):
            blocks.append(ResBlock(out_channels, out_channels, 1, num_groups))
        return nn.Sequential(*blocks)

    def forward(
        self, x: torch.Tensor, return_all_features: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input tensor (B, C, H, W)
            return_all_features: If True, return features from all stages

        Returns:
            Dictionary of features at different scales
        """
        features = {}

        x = self.stem(x)
        features["stem"] = x

        for i, stage in enumerate(self.stages):
            x = stage(x)
            features[f"stage_{i}"] = x

        # Global average pooling
        features["global"] = F.adaptive_avg_pool2d(x, 1).flatten(1)

        return features


class MultiScaleFeatureExtractor(nn.Module):
    """Extract features at multiple scales with patch pooling.

    Implements the multi-scale feature extraction described in the paper
    with features at 4 stages and various spatial resolutions.
    """

    def __init__(
        self,
        encoder: ResNetEncoder,
        pool_sizes: List[int] = [2, 4],
    ):
        super().__init__()
        self.encoder = encoder
        self.pool_sizes = pool_sizes

        # Compute feature dimensions at each scale
        self.feature_dims = self._compute_feature_dims()

    def _compute_feature_dims(self) -> Dict[str, int]:
        """Compute feature dimensions for each scale."""
        dims = {}
        for i, ch in enumerate(self.encoder.stage_channels):
            dims[f"stage_{i}"] = ch
            for pool_size in self.pool_sizes:
                dims[f"stage_{i}_pool_{pool_size}"] = ch
        dims["global"] = self.encoder.stage_channels[-1]
        return dims

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features with pooling.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Dictionary of flattened features at different scales
        """
        # Get base features
        base_features = self.encoder(x)

        features = {}

        # Process each stage
        for i in range(self.encoder.num_stages):
            feat = base_features[f"stage_{i}"]
            B, C, H, W = feat.shape

            # Full resolution (flattened spatial)
            features[f"stage_{i}"] = feat.flatten(2)  # (B, C, H*W)

            # Pooled versions
            for pool_size in self.pool_sizes:
                if H >= pool_size and W >= pool_size:
                    pooled = F.adaptive_avg_pool2d(feat, pool_size)
                    features[f"stage_{i}_pool_{pool_size}"] = pooled.flatten(2)

        # Global features
        features["global"] = base_features["global"].unsqueeze(-1)

        return features


class FeatureEncoder(nn.Module):
    """Complete feature encoder for drifting field computation.

    Combines the ResNet encoder with multi-scale extraction and
    optional normalization for stable training.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        channel_multipliers: List[int] = [1, 2, 4, 8],
        num_blocks_per_stage: int = 2,
        num_groups: int = 32,
        pool_sizes: List[int] = [2, 4],
        normalize_features: bool = True,
    ):
        super().__init__()

        self.encoder = ResNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_blocks_per_stage=num_blocks_per_stage,
            num_groups=num_groups,
        )

        self.multi_scale = MultiScaleFeatureExtractor(
            self.encoder, pool_sizes
        )

        self.normalize_features = normalize_features

        # Running statistics for normalization (Eq. 21 in paper)
        # S_j = (1/sqrt(C_j)) * E[||phi_j(x) - phi_j(y)||]
        self.register_buffer(
            "feature_scales",
            torch.ones(len(self.multi_scale.feature_dims)),
        )
        self.register_buffer("num_updates", torch.tensor(0))

    def get_feature_names(self) -> List[str]:
        """Get names of all feature scales."""
        return list(self.multi_scale.feature_dims.keys())

    def forward(
        self, x: torch.Tensor, update_stats: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Extract normalized multi-scale features.

        Args:
            x: Input tensor (B, C, H, W)
            update_stats: Whether to update running normalization stats

        Returns:
            Dictionary of normalized features at different scales
        """
        features = self.multi_scale(x)

        if self.normalize_features:
            normalized_features = {}
            for i, (name, feat) in enumerate(features.items()):
                # feat shape: (B, C, N) where N is spatial size

                if update_stats and self.training:
                    # Update running statistics
                    with torch.no_grad():
                        # Compute pairwise distances
                        B, C, N = feat.shape
                        if B > 1:
                            # Sample pairs
                            idx1 = torch.randperm(B, device=feat.device)[: B // 2]
                            idx2 = torch.randperm(B, device=feat.device)[: B // 2]
                            dist = (feat[idx1] - feat[idx2]).norm(dim=1).mean()
                            scale = dist / (C**0.5)

                            # Exponential moving average
                            momentum = 0.01
                            self.feature_scales[i] = (
                                1 - momentum
                            ) * self.feature_scales[i] + momentum * scale
                            self.num_updates += 1

                # Normalize
                scale = self.feature_scales[i].clamp(min=1e-6)
                normalized_features[name] = feat / scale

            return normalized_features

        return features


class PretrainedFeatureEncoder(nn.Module):
    """Wrapper for using pretrained encoders (SimCLR, MoCo, MAE).

    This allows using pretrained representations for computing
    the drifting field in a more semantically meaningful space.
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        pretrained: bool = True,
        freeze: bool = True,
        pool_sizes: List[int] = [2, 4],
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.pool_sizes = pool_sizes

        # Import torchvision for pretrained models
        try:
            import torchvision.models as models

            if encoder_name == "resnet50":
                resnet = models.resnet50(pretrained=pretrained)
                self.stem = nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.maxpool,
                )
                self.layer1 = resnet.layer1
                self.layer2 = resnet.layer2
                self.layer3 = resnet.layer3
                self.layer4 = resnet.layer4
                self.stage_channels = [256, 512, 1024, 2048]
            else:
                raise ValueError(f"Unknown encoder: {encoder_name}")

            if freeze:
                for param in self.parameters():
                    param.requires_grad = False

        except ImportError:
            raise ImportError("torchvision is required for pretrained encoders")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features from pretrained encoder."""
        features = {}

        x = self.stem(x)
        x = self.layer1(x)
        features["stage_0"] = x.flatten(2)

        x = self.layer2(x)
        features["stage_1"] = x.flatten(2)

        x = self.layer3(x)
        features["stage_2"] = x.flatten(2)

        x = self.layer4(x)
        features["stage_3"] = x.flatten(2)

        features["global"] = F.adaptive_avg_pool2d(x, 1).flatten(1).unsqueeze(-1)

        # Add pooled versions
        for name in list(features.keys()):
            if name.startswith("stage_"):
                feat = features[name]
                B, C, N = feat.shape
                H = W = int(N**0.5)
                if H * W == N:
                    feat_2d = feat.reshape(B, C, H, W)
                    for pool_size in self.pool_sizes:
                        if H >= pool_size:
                            pooled = F.adaptive_avg_pool2d(feat_2d, pool_size)
                            features[f"{name}_pool_{pool_size}"] = pooled.flatten(2)

        return features
