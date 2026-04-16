# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
# DINOv3-based ConvNeXt UNet for image super-resolution.
#
# Architecture:
#   1. ConvNeXtEncoder: pretrained ConvNeXt backbone extracting multi-scale features
#   2. UNetDecoder: upsampling path with skip connections from encoder
#   3. ConvNeXtUNet: full model combining encoder + decoder for flow matching / SR

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.resnext_dino import ConvNeXt
from basicsr.archs.unet_arch import (
    Swish, group_norm, upsample, ResidualBlock,
    SelfAttention, TimestepEmbedding, Dual_TimestepEmbedding,
    conv2d,
)
from basicsr.utils.registry import ARCH_REGISTRY


# =============================================================================
# ConvNeXt Encoder - extracts multi-scale features from pretrained ConvNeXt
# =============================================================================

class ConvNeXtEncoder(nn.Module):
    """
    ConvNeXt encoder that extracts multi-scale features from a pretrained ConvNeXt backbone.
    Returns a list of feature maps at different resolutions for UNet skip connections.

    The encoder processes through 4 stages:
      - Stage 0: H/4  x W/4  (via stem + stage0)
      - Stage 1: H/8  x W/8  (via downsample + stage1)
      - Stage 2: H/16 x W/16 (via downsample + stage2)
      - Stage 3: H/32 x W/32 (via downsample + stage3, bottleneck)

    Args:
        in_chans (int): Number of input channels. Default: 3
        pretrained (str | None): Path to pretrained ConvNeXt weights. Default: None
        depths (list): Number of blocks per stage. Default: [3, 3, 9, 3]
        dims (list): Feature dimension per stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.0
        freeze_backbone (bool): If True, freeze backbone parameters. Default: False
    """

    def __init__(
        self,
        in_chans=3,
        pretrained=None,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        freeze_backbone=False,
        scale_factor=[4, 2, 2, 2]
    ):
        super().__init__()
        self.dims = dims
        self.depths = depths
        self.num_resolutions = 4

        # Build ConvNeXt backbone
        self.backbone = ConvNeXt(
            in_chans=in_chans,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            scale_factor=scale_factor,
        )

        # Load pretrained weights
        if pretrained is not None:
            self._load_pretrained(pretrained)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(self.backbone.downsample_layers[0])
            stem = self.backbone.downsample_layers[0]
            for i in stem:
                i.weight.requires_grad = True
                i.bias.requires_grad = True

    def _load_pretrained(self, pretrained_path):
        """Load pretrained ConvNeXt weights with filtering."""
        skip_layers = ['downsample_layers.0.0.weight', 'downsample_layers.0.0.bias']  # 例如跳过最后的全连接层
        # 过滤：只加载不在 skip_layers 中的权重
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model_dict = self.backbone.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items()
                         if k in model_dict and k not in skip_layers}
        self.backbone.load_state_dict(filtered_dict, strict=False)

    def forward(self, x):
        """
        Forward pass extracting multi-scale features.

        Args:
            x: Input tensor of shape [B, C, H, W]
        Returns:
            List of feature maps [stage0, stage1, stage2, stage3]
            stage0 has highest spatial resolution, stage3 has lowest (bottleneck).
            Each stage_i has shape [B, dims[i], H/4, W/4] when use_multiscale=False,
            or [B, dims[i], H/4*2^{-i}, W/4*2^{-i}] when use_multiscale=True.
        """
        features = []

        # Process through ConvNeXt stages (all at H/4 resolution)
        for i in range(4):
            x = self.backbone.downsample_layers[i](x)
            x = self.backbone.stages[i](x)
            features.append(x)
        return features


# =============================================================================
# UNet Decoder - upsampling path with skip connections from encoder
# =============================================================================

class UNetDecoder(nn.Module):
    """
    UNet decoder with upsampling and skip connections.

    Architecture (mirroring PnPFlowUNet style):
        - 4 decoder levels, each upsampling by 2x and concatenating with encoder features
        - ResidualBlock(s) with optional SelfAttention at each level
        - Timestep embedding conditioning via ResidualBlock's temb input

    Channel matching:
        - Decoder starts from bottleneck (encoder_dims[-1])
        - Each level upsamples by 2x, concatenates with encoder skip features,
          and processes through ResidualBlock(s)
        - Output channels of each level match the corresponding encoder stage dims

    Args:
        encoder_dims (list): Channel dimensions of encoder stages [C0, C1, C2, C3]
        num_res_blocks (int): Residual blocks per decoder level. Default: 2
        attn_resolutions (tuple): Spatial resolutions at which to apply attention.
                                  Default: (16,)
        dropout (float): Dropout rate. Default: 0.0
        resamp_with_conv (bool): Whether to use conv in upsample. Default: True
        act: Activation function. Default: Swish()
        normalize: Normalization function. Default: group_norm
        meanflow (bool): Use dual timestep embedding. Default: False
    """

    def __init__(
        self,
        encoder_dims=[96, 192, 384, 768],
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        act=Swish(),
        normalize=group_norm,
        meanflow=False,
    ):
        super().__init__()
        self.encoder_dims = encoder_dims
        self.num_resolutions = len(encoder_dims)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.resamp_with_conv = resamp_with_conv
        self.act = act
        self.normalize = normalize
        self.meanflow = meanflow

        # Timestep embedding (matches PnPFlowUNet interface)
        ch = encoder_dims[-1]  # Start from bottleneck channel dim for temb
        temb_ch = ch * 4
        if meanflow:
            self.temb_net = Dual_TimestepEmbedding(
                embedding_dim=ch, hidden_dim=temb_ch, output_dim=temb_ch, act=act)
        else:
            self.temb_net = TimestepEmbedding(
                embedding_dim=ch, hidden_dim=temb_ch, output_dim=temb_ch, act=act)

        # Build decoder levels (from bottleneck to full resolution)
        up_modules = []

        # encoder_dims = [C0, C1, C2, C3]
        # reversed = [C3, C2, C1, C0]
        # level 0: C3 -> project(C3->C2) -> upsample -> concat(C2) -> blocks -> C2
        # level 1: C2 -> project(C2->C1) -> upsample -> concat(C1) -> blocks -> C1
        # level 2: C1 -> project(C1->C0) -> upsample -> concat(C0) -> blocks -> C0
        # level 3: C0 -> blocks -> C0 (output)
        in_ch = encoder_dims[-1]  # Start: bottleneck channels

        for i_level in range(self.num_resolutions): #4
            block_modules = {}
            enc_stage_idx = self.num_resolutions - 1 - i_level  # 3, 2, 1, 0
            skip_ch = encoder_dims[enc_stage_idx]  # encoder channels at this stage

            # Number of residual blocks
            n_blocks = num_res_blocks + 1 if i_level > 0 else num_res_blocks
            out_ch = skip_ch  # Each level outputs channels matching the skip connection

            # Project channels before upsampling (bottleneck C3 -> target C2)
            if i_level == 0:
                # First level: project bottleneck to match next level's channels
                block_modules['bottleneck_proj'] = nn.Sequential(
                    normalize(in_ch),
                    act,
                    conv2d(in_ch, skip_ch),
                )
                in_ch = skip_ch  # After projection, in_ch becomes skip_ch

            for i_block in range(n_blocks):
                if i_block == 0 and i_level > 0:
                    # First block: concatenate upsampled + skip features
                    block_in_ch = in_ch + skip_ch
                elif i_block == 0 and i_level == 0:
                    # First level after projection: no concat yet
                    block_in_ch = in_ch
                else:
                    block_in_ch = out_ch

                block_modules[f'{i_level}a_{i_block}a_block'] = ResidualBlock(
                    in_ch=block_in_ch,
                    temb_ch=temb_ch,
                    out_ch=out_ch,
                    dropout=dropout,
                    act=act,
                    normalize=normalize,
                )

            # Upsample (except for the last/finest level which outputs at full resolution)
            if i_level < self.num_resolutions - 1:
                block_modules[f'{i_level}b_upsample'] = upsample(
                    out_ch, with_conv=resamp_with_conv)

            up_modules.append(nn.ModuleDict(block_modules))
            # Update in_ch to out_ch for the next level's concatenation
            in_ch = out_ch

        self.up_modules = nn.ModuleList(up_modules)

        # End convolution (applied at full resolution)
        self.end_conv = nn.Sequential(
            normalize(encoder_dims[0]),
            self.act,
            conv2d(encoder_dims[0], encoder_dims[0], init_scale=0.),
        )

    def forward(self, encoder_features, temb):
        """
        Forward pass through the decoder.

        Args:
            encoder_features: List of encoder feature maps [stage0, stage1, stage2, stage3]
                              stage0 has highest spatial resolution (full H,W),
                              stage3 has lowest (bottleneck, H/32, W/32).
            temb: Pre-computed timestep embedding of shape [B, temb_ch]

        Returns:
            Decoded features at full (stage0) resolution, shape [B, encoder_dims[0], H, W]
        """
        # Reverse: stage3 (bottleneck) first, stage0 last
        enc_feats = list(reversed(encoder_features))
        # Start from bottleneck
        h = enc_feats[0]

        for i_level in range(self.num_resolutions):
            block_modules = self.up_modules[i_level]
            n_blocks = self.num_res_blocks + 1 if i_level > 0 else self.num_res_blocks

            # Apply bottleneck projection at first level
            if i_level == 0:
                h = block_modules['bottleneck_proj'](h)

            # Residual blocks
            for i_block in range(n_blocks):
                resnet_block = block_modules[f'{i_level}a_{i_block}a_block']
                h = resnet_block(h, temb)

            # Upsample + concatenate with skip connection (except for last level)
            if i_level < self.num_resolutions - 1:
                h = self.up_modules[i_level][f'{i_level}b_upsample'](h)
                # Concatenate with next encoder feature (which has the matching resolution)
                h = torch.cat([h, enc_feats[i_level + 1]], dim=1)

        # End convolution
        h = self.end_conv(h)
        return h


# =============================================================================
# ConvNeXt UNet - Full model combining encoder + decoder
# =============================================================================

@ARCH_REGISTRY.register()
class ConvNeXtUNet(nn.Module):
    """
    ConvNeXt-based UNet for image super-resolution / flow matching.

    Architecture:
        1. ConvNeXtEncoder: pretrained ConvNeXt backbone (freeze or fine-tune)
        2. UNetDecoder: upsampling path with skip connections and timestep conditioning
        3. Output projection: predicts residual or velocity field

    Design:
        - Takes concatenated [x_t, lq_bicubic] as input (like PnPFlowUNet)
        - Supports both flow matching (velocity prediction) and standard SR
        - Uses timestep embedding for diffusion-style conditioning

    Args:
        input_channels (int): Number of input channels. Default: 6 ([x_t, lq_bicubic])
        img_size (int): Input image size. Default: 256
        output_channels (int): Number of output channels. Default: 3
        pretrained (str | None): Path to pretrained ConvNeXt weights. Default: None
        depths (list): ConvNeXt stage depths. Default: [3, 3, 9, 3]
        dims (list): ConvNeXt stage dimensions. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.0
        num_res_blocks (int): Residual blocks per decoder level. Default: 2
        attn_resolutions (tuple): Resolutions for attention. Default: (16,)
        dropout (float): Dropout rate. Default: 0.0
        resamp_with_conv (bool): Use conv in upsample. Default: True
        act: Activation function. Default: Swish()
        normalize: Normalization function. Default: group_norm
        meanflow (bool): Use dual timestep embedding. Default: False
        freeze_backbone (bool): Freeze encoder weights. Default: False
        check_point (str | None): Path to checkpoint. Default: None
    """

    def __init__(
        self,
        input_channels=6,
        img_size=256,
        output_channels=3,
        encoder_type='tiny',
        drop_path_rate=0.0,
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        scale_factor_first_layer=2,
        resamp_with_conv=True,
        act=Swish(),
        normalize=group_norm,
        meanflow=False,
        freeze_backbone=False,
        check_point=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        # self.img_size = img_size
        self.output_channels = output_channels
        self.depths = dinov3[encoder_type]['depths']
        self.dims = dinov3[encoder_type]['dims']
        self.pretrained = dinov3[encoder_type]['pretrained']
        dims = self.dims
        depths = self.depths
        pretrained = self.pretrained
        self.num_res_blocks = num_res_blocks
        self.meanflow = meanflow

        # Resolve string act/normalize
        if isinstance(act, str):
            if act.lower() == 'swish':
                act = Swish()
            elif act.lower() == 'relu':
                act = nn.ReLU(inplace=True)
            elif act.lower() == 'gelu':
                act = nn.GELU()
            else:
                raise ValueError(f"Unknown act: {act}")
        if isinstance(normalize, str):
            if normalize.lower() == 'group_norm':
                normalize = group_norm
        self.act = act
        self.normalize = normalize

        # Build encoder
        self.encoder = ConvNeXtEncoder(
            in_chans=input_channels,
            pretrained=pretrained,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            freeze_backbone=freeze_backbone,
        )

        # Build decoder
        self.decoder = UNetDecoder(
            encoder_dims=dims,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
            act=act,
            normalize=normalize,
            meanflow=meanflow,
        )

        # Output projection: project from decoder's output channels to output channels
        # Decoder outputs at H/4 resolution. We need to upsample by factor 4 to reach H.
        if hasattr(self.encoder, 'use_multiscale') and self.encoder.use_multiscale:
            # With multi-scale encoder (scale_factor=4), decoder outputs at H/4
            # Upsample by 4x to match original resolution
            self.output_proj = nn.Sequential(
                normalize(dims[0]),
                act,
                nn.Upsample(scale_factor=scale_factor_first_layer, mode='bilinear', align_corners=False),
                conv2d(dims[0], output_channels, init_scale=0.),
            )
        else:
            self.output_proj = nn.Sequential(
                normalize(dims[0]),
                act,
                conv2d(dims[0], output_channels, init_scale=0.),
            )

        # Load checkpoint if provided
        if check_point is not None:
            checkpoint = torch.load(check_point, map_location='cpu')
            self.load_state_dict(checkpoint, strict=False)

    def forward(self, x, c=None, temp=None):
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W] (typically [x_t, lq_bicubic] concatenated)
            c: Optional condition (not used, kept for interface compatibility)
            temp: Timestep(s) [B] or [B, ...]. If None, uses default timestep.

        Returns:
            Output tensor [B, output_channels, H, W]
        """
        B, C, H, W = x.size()

        # Timestep embedding
        if temp is None:
            temp = torch.ones(B, device=x.device)
        elif temp.dim() > 1 and temp.size(1) > 1:
            temp = temp.squeeze()
        if self.meanflow:
            if temp.dim() == 1:
                temp_t = temp
                temp_r = temp
            else:
                temp_t = temp[:, 0]
                temp_r = temp[:, 1]
            temb = self.decoder.temb_net(temp_t, temp_r)
        else:
            if temp.dim() > 1:
                temp = temp.squeeze()
            temb = self.decoder.temb_net(temp)

        # Encode: get multi-scale features
        encoder_features = self.encoder(x)  # [stage0, stage1, stage2, stage3]
        # Decode: upsample with skip connections
        decoded = self.decoder(encoder_features, temb)

        # Output projection
        out = self.output_proj(decoded)
        return out


# =============================================================================
# Convenience constructors for different ConvNeXt sizes
# =============================================================================

dinov3 = {'tiny': {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768], 'pretrained': '/8T2/pretrained_models/DINOV3_pretrained_weight/ConvNeXt/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth'}, #/8T2/pretrained_models/EUPE/resnext/EUPE-ConvNeXt-T.pt
        'small': {'depths': [3, 3, 27, 3], 'dims': [96, 192, 384, 768], 'pretrained': '/8T2/pretrained_models/DINOV3_pretrained_weight/ConvNeXt/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth'},
        'base': {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024], 'pretrained': '/8T2/pretrained_models/DINOV3_pretrained_weight/ConvNeXt/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth'},
        'large': {'depths': [3, 3, 27, 3], 'dims': [192, 384, 768, 1536], 'pretrained': '/8T2/pretrained_models/DINOV3_pretrained_weight/ConvNeXt/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth'}}

# def convnext_unet_tiny(input_channels=6, img_size=256, output_channels=3,
#                         pretrained=None, **kwargs):
#     """ConvNeXt-Tiny UNet for super-resolution."""
#     if pretrained is None:
#         pretrained = '/8T2/pretrained_models/DINOV3_pretrained_weight/ConvNeXt/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth'
#     model = ConvNeXtUNet(
#         input_channels=input_channels,
#         img_size=img_size,
#         output_channels=output_channels,
#         pretrained=pretrained,
#         depths=[3, 3, 9, 3],
#         dims=[96, 192, 384, 768],
#         **kwargs,
#     )
#     return model
#
#
# def convnext_unet_small(input_channels=6, img_size=256, output_channels=3,
#                         pretrained=None, **kwargs):
#     """ConvNeXt-Small UNet for super-resolution."""
#     if pretrained is None:
#         pretrained = '/8T2/pretrained_models/DINOV3_pretrained_weight/ConvNeXt/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth'
#     model = ConvNeXtUNet(
#         input_channels=input_channels,
#         img_size=img_size,
#         output_channels=output_channels,
#         pretrained=pretrained,
#         depths=[3, 3, 27, 3],
#         dims=[96, 192, 384, 768],
#         **kwargs,
#     )
#     return model
#
#
# def convnext_unet_base(input_channels=6, img_size=256, output_channels=3,
#                        pretrained=None, **kwargs):
#     """ConvNeXt-Base UNet for super-resolution."""
#     if pretrained is None:
#         pretrained = '/8T2/pretrained_models/DINOV3_pretrained_weight/ConvNeXt/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth'
#     model = ConvNeXtUNet(
#         input_channels=input_channels,
#         img_size=img_size,
#         output_channels=output_channels,
#         pretrained=pretrained,
#         depths=[3, 3, 27, 3],
#         dims=[128, 256, 512, 1024],
#         **kwargs,
#     )
#     return model
#
#
# def convnext_unet_large(input_channels=6, img_size=256, output_channels=3,
#                          pretrained=None, **kwargs):
#     """ConvNeXt-Large UNet for super-resolution."""
#     if pretrained is None:
#         pretrained = '/8T2/pretrained_models/DINOV3_pretrained_weight/ConvNeXt/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth'
#     model = ConvNeXtUNet(
#         input_channels=input_channels,
#         img_size=img_size,
#         output_channels=output_channels,
#         pretrained=pretrained,
#         depths=[3, 3, 27, 3],
#         dims=[192, 384, 768, 1536],
#         **kwargs,
#     )
#     return model
#
#
# if __name__ == "__main__":
#     # Quick test
#     model = convnext_unet_tiny(input_channels=6, img_size=256, pretrained=None)
#     x = torch.randn(2, 6, 256, 256)
#     t = torch.tensor([0.5, 0.5])
#     out = model(x, None, t)
#     print(f"Input:  {x.shape}")
#     print(f"Output: {out.shape}")
#     print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
