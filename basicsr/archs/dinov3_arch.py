# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch
from torch import nn as nn
from basicsr.archs.dinov3.convnext import ConvNeXt

from basicsr.utils.registry import ARCH_REGISTRY


class UpsampleConvNeXt(nn.Module):
    """Upsampling head for ConvNeXt super-resolution."""

    def __init__(self, upscale, embed_dim, num_out_ch):
        super(UpsampleConvNeXt, self).__init__()
        self.upscale = upscale

        if upscale == 1:
            self.conv = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        else:
            self.conv = nn.Conv2d(embed_dim, num_out_ch * upscale ** 2, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(upscale)
        # elif upscale == 2:
        #     self.conv = nn.Conv2d(embed_dim, num_out_ch * 4, 3, 1, 1)
        #     self.pixel_shuffle = nn.PixelShuffle(2)
        # elif upscale == 4:
        #     self.conv = nn.Conv2d(embed_dim, num_out_ch * 16, 3, 1, 1)
        #     self.pixel_shuffle = nn.PixelShuffle(4)
        # elif upscale == 8:
        #     self.conv = nn.Conv2d(embed_dim, num_out_ch * 64, 3, 1, 1)
        #     self.pixel_shuffle = nn.PixelShuffle(8)
        # else:
        #     raise ValueError(f'Unsupported upscale factor: {upscale}')

    def forward(self, x):
        if self.upscale == 1:
            return self.conv(x)
        else:
            return self.pixel_shuffle(self.conv(x))


@ARCH_REGISTRY.register()
class ConvNeXtSR(nn.Module):
    """ConvNeXt for Super-Resolution.

    This is a wrapper around ConvNeXt from the DINOv3 project,
    adapted for image super-resolution tasks with an upsampling head.

    Args:
        in_chans (int): Channel number of inputs. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 27, 3]
        dims (list): Feature dimension at each stage. Default: [128, 256, 512, 1024]
        drop_path_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6
        patch_size (int | None): Pseudo patch size for feature resizing. Default: None
        upscale (int): Upsampling factor. Default: 4
        num_out_ch (int): Output channel number. Default: 3
    """

    def __init__(
        self,
        in_chans=3,
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        patch_size=None,
        upscale=4,
        out_chans=3,
        pretrained=None,
        img_size=None
    ):
        super(ConvNeXtSR, self).__init__()

        self.in_chans = in_chans
        self.depths = depths
        self.dims = dims
        self.upscale = upscale
        self.out_chans = out_chans

        # Build ConvNeXt backbone from DINOv3
        self.backbone = ConvNeXt(
            in_chans=in_chans,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            patch_size=patch_size,
            scale_factor=upscale
        )
        if pretrained is not None:
            skip_layers = [] #['downsample_layers.0.0.weight', 'downsample_layers.0.0.bias']  # 例如跳过最后的全连接层

            # 过滤：只加载不在 skip_layers 中的权重
            state_dict = torch.load(pretrained, map_location='cpu')
            model_dict = self.backbone.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items()
                             if k in model_dict and k not in skip_layers}
            self.backbone.load_state_dict(filtered_dict, strict=False)

        # Get final embedding dimension
        embed_dim = dims[-1]

        # Feature projection
        self.proj = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # Upsampling head
        self.upsample = UpsampleConvNeXt(upscale, embed_dim, self.out_chans)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Output tensor of shape [B, C, H*upscale, W*upscale]
        """
        # Get backbone features
        # with torch.no_grad:
        features = self.backbone.forward_features(x)

        # Handle both single dict and list return
        if isinstance(features, list):
            features = features[0]

        # Extract patch tokens: x_norm_patchtokens
        x_norm_patch = features['x_norm_patchtokens']  # [B, num_patches, embed_dim]

        # Reshape to 2D feature map
        B = x_norm_patch.shape[0]
        num_patches = x_norm_patch.shape[1]

        # Calculate spatial dimensions (assuming square patches)
        H = W = int(num_patches ** 0.5)
        x_patch = x_norm_patch.reshape(B, H, W, self.dims[-1]).permute(0, 3, 1, 2)  # [B, C, H, W]

        # Project features
        x_patch = self.proj(x_patch)

        # Upsample
        output = self.upsample(x_patch)

        return output


# Convenience constructors for different ConvNeXt sizes

def convnext_tiny(upscale=4, **kwargs):
    """ConvNeXt-Tiny for super-resolution."""
    model = ConvNeXtSR(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        upscale=upscale,
        pretrained='/8T2/Project/DINOV3_pretrained_weight/ConvNeXt/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth',
        **kwargs,
    )
    return model


def convnext_small(upscale=4, **kwargs):
    """ConvNeXt-Small for super-resolution."""
    model = ConvNeXtSR(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        upscale=upscale,
        pretrained='/8T2/Project/DINOV3_pretrained_weight/ConvNeXt/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth',
        **kwargs,
    )
    return model


def convnext_base(upscale=4, **kwargs):
    """ConvNeXt-Base for super-resolution."""
    model = ConvNeXtSR(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        upscale=upscale,
        pretrained = '/8T2/Project/DINOV3_pretrained_weight/ConvNeXt/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth',
        **kwargs,
    )
    return model


def convnext_large(upscale=4, **kwargs):
    """ConvNeXt-Large for super-resolution."""
    model = ConvNeXtSR(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        upscale=upscale,
        pretrained = '/8T2/Project/DINOV3_pretrained_weight/ConvNeXt/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth',
        **kwargs,
    )
    return model

if __name__ == "__main__":
    model = convnext_tiny()
    x = torch.randn(2, 3, 256, 256)
    out = model.backbone._get_intermediate_layers(x, 4)
    for i in out:
        print(len(i))