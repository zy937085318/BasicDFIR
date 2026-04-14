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
#
# ATD Integration:
#   ATD (Adaptive Token Dictionary) components are ported from atd_arch.py
#   and integrated into UNetDecoder for enhanced feature processing.

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.atd_arch import ATDB
from basicsr.archs.dinov3.convnext import ConvNeXt
from basicsr.archs.unet_arch import (
    Swish, group_norm, upsample, ResidualBlock,
    SelfAttention, TimestepEmbedding, Dual_TimestepEmbedding,
    conv2d,
)
from basicsr.utils.registry import ARCH_REGISTRY
from timm.layers import to_2tuple, trunc_normal_
from basicsr.archs.swinir_arch import Upsample

# =============================================================================
# ATD Components (ported from atd_arch.py)
# =============================================================================

def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r

def feature_shuffle(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)
    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)
    shuffled_x = torch.gather(x, dim=dim-1, index=index)
    return shuffled_x

class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1, groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

class ConvFFN_td(nn.Module):
    def __init__(self, in_features, hidden_features=None, td_features=0, out_features=None,
                 kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        hidden_features += td_features
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_td, x_size):
        x = self.fc1(x)
        x = torch.cat([self.act(x), x_td], dim=-1)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x

def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows

def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

class ATD_CA(nn.Module):
    """ATD Cross-Attention: Query from features, Key/Value from Token Dictionary."""
    def __init__(self, dim, num_tokens=64, reducted_dim=10, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.dr = reducted_dim
        self.qkv_bias = qkv_bias

        self.wq = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.scale = nn.Parameter(torch.ones([1]), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, td, x_size):
        b, n, c = x.shape
        b, m, c = td.shape

        q = self.wq(x)  # b, n, dr
        k = self.wk(td)  # b, m, dr
        v = self.wv(td)  # b, m, c

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        scale = 1 + torch.clamp(self.scale, 0, 3) * np.log(self.num_tokens)
        attn = attn * scale
        attn = self.softmax(attn)

        x = (attn @ v).reshape(b, n, c)
        return x, attn

class AC_MSA(nn.Module):
    """Adaptive Category Multi-Head Self-Attention."""
    def __init__(self, dim, num_heads=4, category_size=128, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.category_size = category_size
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.scale = (dim // num_heads) ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, tk_id, x_size):
        b, n, c3 = qkv.shape
        c = c3 // 3
        gs = min(n, self.category_size)
        ng = (n + gs - 1) // gs

        x_sort_values, x_sort_indices = torch.sort(tk_id, dim=-1, stable=False)
        tk_id_inv = index_reverse(x_sort_indices)

        shuffled_qkv = feature_shuffle(qkv, x_sort_indices)
        pad_n = ng * gs - n
        paded_qkv = torch.cat((shuffled_qkv, torch.flip(shuffled_qkv[:, n-pad_n:n, :], dims=[1])), dim=1)
        y = paded_qkv.reshape(b, -1, gs, c3)

        qkv = y.reshape(b, ng, gs, 3, self.num_heads, c // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))
        attn = attn * self.scale
        attn = self.softmax(attn)

        y = (attn @ v).permute(0, 1, 3, 2, 4).reshape(b, n+pad_n, c)[:, :n, :]
        x = feature_shuffle(y, tk_id_inv)
        x = self.proj(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.img_size if input_resolution is None else input_resolution
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops

class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x

    def flops(self, input_resolution=None):
        return 0

class ATDDecoderBlock(nn.Module):
    """
    ATD-enhanced decoder block combining:
      - ATD_CA: Cross-attention with learnable Token Dictionary
      - AC_MSA: Adaptive Category Multi-Head Self-Attention
      - ConvFFN_td: Token-Dictionary-conditioned Feed-Forward Network

    Integrated into UNetDecoder to replace/enhance ResidualBlocks at each upsampling level.
    """
    def __init__(self, dim, num_tokens=64, reducted_dim=10, num_heads=4,
                 category_size=128, dim_ffn_td=16, convffn_kernel_size=5,
                 mlp_ratio=4., qkv_bias=True, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.dim_ffn_td = dim_ffn_td
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.category_size = category_size

        # Token Dictionary (learnable)
        self.td = nn.Parameter(torch.randn([num_tokens, dim]), requires_grad=True)

        # ATD Cross-Attention
        self.atd_ca = ATD_CA(dim, num_tokens=num_tokens, reducted_dim=reducted_dim, qkv_bias=qkv_bias)
        # AC-MSA
        self.ac_msa = AC_MSA(dim, num_heads=num_heads, category_size=category_size, qkv_bias=qkv_bias)
        # QKV projection for AC-MSA
        self.wqkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
        # FFN with TD conditioning
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc_td = nn.Linear(dim, dim_ffn_td)
        self.convffn = ConvFFN_td(in_features=dim, hidden_features=mlp_hidden_dim,
                                   td_features=dim_ffn_td, kernel_size=convffn_kernel_size,
                                   act_layer=act_layer)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

        # Patch embedding/unembedding for sequence conversion
        self.patch_embed = PatchEmbed(img_size=64, patch_size=1, in_chans=0, embed_dim=dim)
        self.patch_unembed = PatchUnEmbed(img_size=64, patch_size=1, in_chans=0, embed_dim=dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_size, params=None):
        """
        Args:
            x: Input tensor [B, C, H, W]
            x_size: (H, W) tuple
            params: Optional dict containing 'attn_mask', 'rpi_sa' (for SW-MSA compatibility)
        Returns:
            Enhanced tensor [B, C, H, W]
        """
        b, c, h, w = x.shape
        x_size = (h, w)

        # Convert to sequence: BCHW -> BNC
        x_seq = self.patch_embed(x)  # [B, N, C] where N = h*w
        shortcut = x_seq

        x_seq = self.norm1(x_seq)

        # ATD_CA: Cross-attention with Token Dictionary
        td_expanded = self.td.expand([b, -1, -1])  # [B, num_tokens, C]
        x_atd, sim_atd = self.atd_ca(x_seq, td_expanded, x_size)  # x_atd: [B, N, C]

        # AC_MSA: Adaptive Category attention
        tk_id = torch.argmax(sim_atd, dim=-1, keepdim=False)  # [B, N]
        qkv = self.wqkv(x_seq)  # [B, N, 3C]
        x_aca = self.ac_msa(qkv, tk_id, x_size)  # [B, N, C]

        # Token-gathered FFN conditioning
        x_td = torch.gather(
            self.fc_td(td_expanded),
            dim=1,
            index=tk_id.reshape(b, h*w, 1).expand(-1, -1, self.dim_ffn_td)
        )  # [B, N, dim_ffn_td]

        # FFN with TD
        x_ffn = self.convffn(self.norm2(x_seq), x_td, x_size)
        x_ffn = self.norm3(x_ffn)

        # Combine: attention outputs + FFN
        x_out = shortcut + x_atd + x_aca + x_ffn

        # Convert back to spatial: BNC -> BCHW
        x_out = self.patch_unembed(x_out, x_size)

        return x_out


# =============================================================================
# Feature Refine - refines multi-scale encoder features using ATDB blocks
# =============================================================================

class FeatureRefine(nn.Module):
    """
    Feature refinement module using ATDB blocks from atd_arch.py.
    Processes ONLY the bottleneck (last stage) encoder feature before it enters the decoder.

    Design:
      - Takes encoder_features [stage0, stage1, stage2, stage3] as input
      - Refines only stage3 (bottleneck) with ATDB
      - Returns [stage0, stage1, stage2, refined_stage3]

    This reduces computation while still refining the most critical bottleneck features.

    Definition follows ATD class pattern:
      - PatchEmbed/PatchUnEmbed for bottleneck BCHW <-> BNC conversion
      - calculate_rpi_sa / calculate_mask for SW-MSA
      - Residual connection at FeatureRefine level
    """

    def __init__(
        self,
        dims,
        depths,
        num_heads,
        window_size=8,
        dim_ffn_td=16,
        category_size=128,
        num_tokens=64,
        reducted_dim=4,
        convffn_kernel_size=5,
        mlp_ratio=2.,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        resi_connection='1conv',
        use_checkpoint=False,
        img_size=64,
        patch_size=1,
    ):
        super().__init__()
        self.num_layers = len(dims)
        self.window_size = window_size
        self.bottleneck_idx = self.num_layers - 1  # Last stage index

        # Bottleneck resolution (H/32, W/32 for scale_factor=4)
        bottleneck_resolution = (img_size // (4 * (2 ** self.bottleneck_idx)),
                                  img_size // (4 * (2 ** self.bottleneck_idx)))
        bottleneck_dim = dims[self.bottleneck_idx]

        # PatchEmbed / PatchUnEmbed / norm for bottleneck only
        self.patch_embed = PatchEmbed(
            img_size=bottleneck_resolution[0], patch_size=patch_size,
            in_chans=0, embed_dim=bottleneck_dim)
        self.patch_unembed = PatchUnEmbed(
            img_size=bottleneck_resolution[0], patch_size=patch_size,
            in_chans=0, embed_dim=bottleneck_dim)
        self.norm = norm_layer(bottleneck_dim)

        # Single ATDB block for bottleneck refinement
        self.atdb = ATDB(
            dim=bottleneck_dim,
            idx=self.bottleneck_idx,
            input_resolution=bottleneck_resolution,
            depth=depths[self.bottleneck_idx],  # Use bottleneck depth
            num_heads=num_heads[self.bottleneck_idx],  # Use bottleneck heads
            window_size=window_size,
            dim_ffn_td=dim_ffn_td,
            category_size=category_size,
            num_tokens=num_tokens,
            reducted_dim=reducted_dim,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=bottleneck_resolution[0],
            patch_size=patch_size,
            resi_connection=resi_connection,
        )

        # Relative position index for SW-MSA
        relative_position_index_SA = self._calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

    def _calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def _calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -(self.window_size // 2)),
                    slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -(self.window_size // 2)),
                    slice(-(self.window_size // 2), None))
        cnt = 0
        for h_s in h_slices:
            for w_s in w_slices:
                img_mask[:, h_s, w_s, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, encoder_features):
        """
        Args:
            encoder_features: [stage0, stage1, stage2, stage3], each BCHW
        Returns:
            refined_features: [stage0, stage1, stage2, refined_stage3]
                             Only the bottleneck (last stage) is refined by ATDB
        """
        # Pass through stages 0-2 unchanged (for skip connections)
        refined = list(encoder_features[:-1])

        # Refine only the bottleneck (stage3)
        feat = encoder_features[self.bottleneck_idx]
        h, w = feat.shape[2], feat.shape[3]

        # Padding to window_size (like ATD.forward)
        mod = self.window_size
        h_pad = ((h + mod - 1) // mod) * mod - h
        w_pad = ((w + mod - 1) // mod) * mod - w
        feat_padded = feat
        if h_pad > 0 or w_pad > 0:
            feat_padded = torch.cat([feat, torch.flip(feat, [2])], 2)[:, :, :h + h_pad, :]
            feat_padded = torch.cat([feat_padded, torch.flip(feat_padded, [3])], 3)[:, :, :, :w + w_pad]

        x_size = (h + h_pad, w + w_pad)
        attn_mask = self._calculate_mask(x_size).to(feat.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        # BCHW -> BNC -> ATDB -> BNC -> BCHW (like ATD.forward_features)
        x = self.patch_embed(feat_padded)
        x = self.atdb(x, x_size, params)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        # Residual + unpad (like ATD: conv_after_body(forward_features(x)) + x)
        x = x + feat_padded
        x = x[:, :, :h, :w]

        refined.append(x)

        return refined


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
        use_multiscale=True,
        scale_factor_first_layer=2
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
            scale_factor=scale_factor_first_layer,  # Default stride-4 in stem (H/4)
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

        # Multi-scale projection layers (only meaningful with use_multiscale=True)
        # The ConvNeXt backbone keeps spatial resolution constant across stages (all at H/4).
        # For a UNet, we need proper spatial pyramids.
        # We add stride-2 downsampling convs to create multi-scale features.
        self.use_multiscale = use_multiscale
        if use_multiscale:
            # Project each stage's output to the next resolution (H/8, H/16, H/32)
            # Stages 0-2 each get a stride-2 conv, stage 3 is already at bottleneck
            self.down_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                )
                for i in range(3)
            ])

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

        # Create multi-scale pyramid via additional downsampling projections
        if self.use_multiscale:
            # features[0]: H/4 x W/4, dims[0]=96
            # features[1]: H/4 x W/4, dims[1]=192 -> project to H/8
            # features[2]: H/4 x W/4, dims[2]=384 -> project to H/16
            # features[3]: H/4 x W/4, dims[3]=768 -> project to H/32
            scaled_feats = [features[0]]  # stage0 at H/4
            for i in range(3):
                # Downsample previous feature to match next stage's spatial resolution
                # Using non-overlapping 2x2 -> 1x1 conv (like original ConvNeXt downsample)
                scaled_feats.append(self.down_proj[i](scaled_feats[-1]))
            return scaled_feats

        return features  # [stage0, stage1, stage2, stage3] all at H/4


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
        # ATD parameters
        use_atd=False,
        atd_num_tokens=64,
        atd_reducted_dim=10,
        atd_num_heads=4,
        atd_category_size=128,
        atd_dim_ffn_td=16,
        atd_convffn_kernel_size=5,
        atd_mlp_ratio=4.,
        norm_layer=nn.LayerNorm,
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
        self.use_atd = use_atd

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

        for i_level in range(self.num_resolutions):  # 4 levels
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

            # ATD block: inserted after ResidualBlocks, before upsample
            if use_atd and i_level < self.num_resolutions - 1:
                block_modules[f'{i_level}c_atd'] = ATDDecoderBlock(
                    dim=out_ch,
                    num_tokens=atd_num_tokens,
                    reducted_dim=atd_reducted_dim,
                    num_heads=atd_num_heads,
                    category_size=atd_category_size,
                    dim_ffn_td=atd_dim_ffn_td,
                    convffn_kernel_size=atd_convffn_kernel_size,
                    mlp_ratio=atd_mlp_ratio,
                    norm_layer=norm_layer,
                )

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

    def forward(self, encoder_features, temb, params=None):
        """
        Forward pass through the decoder.

        Args:
            encoder_features: List of encoder feature maps [stage0, stage1, stage2, stage3]
                              stage0 has highest spatial resolution (full H,W),
                              stage3 has lowest (bottleneck, H/32, W/32).
            temb: Pre-computed timestep embedding of shape [B, temb_ch]
            params: Optional dict containing ATD-related parameters (attn_mask, rpi_sa)

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

            # ATD block: applied before upsample (except for last level)
            if self.use_atd and i_level < self.num_resolutions - 1:
                h = block_modules[f'{i_level}c_atd'](h, (h.shape[2], h.shape[3]), params)

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
class Dino_ATD_UNet(nn.Module):
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
        # ATD parameters
        use_atd=False,
        atd_num_tokens=64,
        atd_reducted_dim=10,
        atd_num_heads=4,
        atd_category_size=128,
        atd_dim_ffn_td=16,
        atd_convffn_kernel_size=5,
        atd_mlp_ratio=4.,
        # Feature refine parameters
        use_feature_refine=False,
        fr_depths=[6, 6, 6, 6],
        fr_num_heads=[6, 6, 6, 6],
        fr_window_size=8,
        fr_dim_ffn_td=16,
        fr_category_size=128,
        fr_num_tokens=64,
        fr_reducted_dim=4,
        fr_convffn_kernel_size=5,
        fr_mlp_ratio=2.,
        fr_resi_connection='1conv',
        fr_use_checkpoint=False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.img_size = img_size
        self.output_channels = output_channels
        self.depths = dinov3[encoder_type]['depths']
        self.dims = dinov3[encoder_type]['dims']
        self.pretrained = dinov3[encoder_type]['pretrained']
        dims = self.dims
        depths = self.depths
        pretrained = self.pretrained
        self.num_res_blocks = num_res_blocks
        self.meanflow = meanflow
        self.use_atd = use_atd
        self.use_feature_refine = use_feature_refine

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

        # self.encoder = nn.Conv2d(input_channels, dims[0], kernel_size=4, stride=4)

        # Build feature refine
        if use_feature_refine:
            self.feature_refine = FeatureRefine(
                dims=dims,
                depths=fr_depths,
                num_heads=fr_num_heads,
                window_size=fr_window_size,
                dim_ffn_td=fr_dim_ffn_td,
                category_size=fr_category_size,
                num_tokens=fr_num_tokens,
                reducted_dim=fr_reducted_dim,
                convffn_kernel_size=fr_convffn_kernel_size,
                mlp_ratio=fr_mlp_ratio,
                resi_connection=fr_resi_connection,
                use_checkpoint=fr_use_checkpoint,
                img_size=img_size,
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
            use_atd=use_atd,
            atd_num_tokens=atd_num_tokens,
            atd_reducted_dim=atd_reducted_dim,
            atd_num_heads=atd_num_heads,
            atd_category_size=atd_category_size,
            atd_dim_ffn_td=atd_dim_ffn_td,
            atd_convffn_kernel_size=atd_convffn_kernel_size,
            atd_mlp_ratio=atd_mlp_ratio,
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

        # Feature refine: ATDB-based refinement of encoder features
        if self.use_feature_refine:
            encoder_features = self.feature_refine(encoder_features)

        # Decode: upsample with skip connections
        decoded = self.decoder(encoder_features, temb)

        # Output projection
        out = self.output_proj(decoded)
        return out


# =============================================================================
# Convenience constructors for different ConvNeXt sizes
# =============================================================================

dinov3 = {'tiny': {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768], 'pretrained': '/8T2/pretrained_models/DINOV3_pretrained_weight/ConvNeXt/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth'},
        'small': {'depths': [3, 3, 27, 3], 'dims': [96, 192, 384, 768], 'pretrained': '/8T2/pretrained_models/DINOV3_pretrained_weight/ConvNeXt/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth'},
        'base': {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024], 'pretrained': '/8T2/pretrained_models/DINOV3_pretrained_weight/ConvNeXt/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth'},
        'large': {'depths': [3, 3, 27, 3], 'dims': [192, 384, 768, 1536], 'pretrained': '/8T2/pretrained_models/DINOV3_pretrained_weight/ConvNeXt/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth'}}

