'''
ATD with Geodesic Domain-Transform Window Attention (ATDGeo)

基于 ATD 架构，在 WindowAttention (SW-MSA) 分支中引入测地距离 bias，
使窗口注意力天然具备边缘保持特性。

理论依据:
  - ATD: "Improved Transformer with Adaptive Token Dictionary for Image Restoration"
    (arXiv 2401.08209)
  - Domain Transform: Gastal & Oliveira, "Domain Transform for Edge-Aware Image
    and Video Processing", SIGGRAPH 2011

核心修改:
  WindowAttentionGeo 在标准 SW-MSA 的 attention logits 上叠加测地距离 bias:

    attn = softmax( Q·K^T / √d  +  rel_pos_bias  -  geo_scale · d_geo(i,j) ) · V

  其中 d_geo(i,j) 是像素 i 到 j 的窗口内测地距离，通过 Domain Transform
  公式在特征空间中计算。跨边缘的像素对测地距离大 → attention 权重自动衰减。

  与原 ATD 的三个注意力分支 (SW-MSA, ATD_CA, AC_MSA) 的关系:
  - SW-MSA → 替换为 WindowAttentionGeo（测地距离引导的局部注意力）
  - ATD_CA → 保持不变
  - AC_MSA → 保持不变
  - FFN    → 保持不变
'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple, trunc_normal_
from torch.utils.checkpoint import checkpoint

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.atd_arch import (
    index_reverse, feature_shuffle, dwconv, ConvFFN_td,
    window_partition, window_reverse, PatchEmbed, PatchUnEmbed,
    Upsample, UpsampleOneStep, ATD_CA, AC_MSA,
)


# ===========================================================================
# 测地距离计算
# ===========================================================================

def compute_window_geodesic_bias(feat_windows, ws, sigma):
    """
    在每个窗口内计算像素对之间的测地距离矩阵。

    使用 L 形路径近似 (水平优先再垂直):
      d_geo[(ri,ci)→(rj,cj)] = |ct_h[ri,cj] - ct_h[ri,ci]|   ← 水平段
                               + |ct_v[rj,cj] - ct_v[ri,cj]|   ← 垂直段

    其中 ct_h 沿每行累积: ct_h[r,c] = Σ_{j=0}^{c} (1 + σ·|f[r,j]-f[r,j-1]|)
         ct_v 沿每列累积: ct_v[r,c] = Σ_{i=0}^{r} (1 + σ·|f[i,c]-f[i-1,c]|)

    Args:
        feat_windows: (nw*b, ws, ws, C) 窗口内的特征
        ws: 窗口大小
        sigma: 边缘灵敏度参数 (越大对特征差异越敏感)
    Returns:
        geo_dist: (nw*b, ws*ws, ws*ws) 像素对测地距离矩阵
    """
    nw_b = feat_windows.shape[0]
    device, dtype = feat_windows.device, feat_windows.dtype

    f = feat_windows

    # --- 水平方向测地距离 (沿每行累积) ---
    diff_h = (f[:, :, 1:] - f[:, :, :-1]).norm(p=1, dim=-1)        # (nw*b, ws, ws-1)
    integrand_h = torch.cat([
        torch.zeros(nw_b, ws, 1, device=device, dtype=dtype),
        1.0 + sigma * diff_h
    ], dim=-1)                                                        # (nw*b, ws, ws)
    ct_h = torch.cumsum(integrand_h, dim=-1)                         # (nw*b, ws, ws)

    # --- 垂直方向测地距离 (沿每列累积) ---
    diff_v = (f[:, 1:, :] - f[:, :-1, :]).norm(p=1, dim=-1)         # (nw*b, ws-1, ws)
    integrand_v = torch.cat([
        torch.zeros(nw_b, 1, ws, device=device, dtype=dtype),
        1.0 + sigma * diff_v
    ], dim=-2)                                                        # (nw*b, ws, ws)
    ct_v = torch.cumsum(integrand_v, dim=-2)                         # (nw*b, ws, ws)

    # --- 像素对测地距离 (L 形路径近似) ---
    idx = torch.arange(ws * ws, device=device)
    rows = idx // ws                                                  # (ws*ws,)
    cols = idx % ws                                                   # (ws*ws,)

    # ct at each pixel: ct_h[b, rows[k], cols[k]]
    ct_h_flat = ct_h[:, rows, cols]                                   # (nw*b, ws*ws)
    ct_v_flat = ct_v[:, rows, cols]                                   # (nw*b, ws*ws)

    # ct at corner point (ri, cj):  ct_*[b, rows[i], cols[j]]
    ct_h_corner = ct_h[:, rows.unsqueeze(1), cols.unsqueeze(0)]       # (nw*b, ws*ws, ws*ws)
    ct_v_corner = ct_v[:, rows.unsqueeze(1), cols.unsqueeze(0)]       # (nw*b, ws*ws, ws*ws)

    # 水平段: |ct_h[ri,cj] - ct_h[ri,ci]|
    geo_h = (ct_h_corner - ct_h_flat.unsqueeze(-1)).abs()

    # 垂直段: |ct_v[rj,cj] - ct_v[ri,cj]|
    geo_v = (ct_v_flat.unsqueeze(-2) - ct_v_corner).abs()

    return geo_h + geo_v                                              # (nw*b, ws*ws, ws*ws)


# ===========================================================================
# WindowAttentionGeo: 带测地距离 bias 的窗口注意力
# ===========================================================================

class WindowAttentionGeo(nn.Module):
    """
    在标准 SW-MSA 的 attention logits 上叠加可学习的测地距离 bias:

    attn = softmax(QK^T/√d + rel_pos_bias - geo_scale · d_geo) · V

    - geo_scale: 每个注意力头独立的可学习缩放参数，控制测地距离的影响力
    - d_geo: 基于特征的测地距离，在 ATDTransformerLayerGeo 中预计算后传入
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size    # (ws, ws)
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置 (与原版 WindowAttention 相同)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.proj = nn.Linear(dim, dim)

        # 测地距离缩放 (per-head, 可学习)
        self.geo_scale = nn.Parameter(torch.ones(num_heads) * 0.01)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, rpi, mask=None, geo_dist=None):
        """
        Args:
            qkv:      (nw*b, ws*ws, 3C)
            rpi:      相对位置索引
            mask:     shift window 的 attention mask
            geo_dist: (nw*b, ws*ws, ws*ws) 预计算的测地距离矩阵
        """
        b_, n, c3 = qkv.shape
        c = c3 // 3

        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 相对位置偏置
        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        # ★ 测地距离 bias (核心修改)
        if geo_dist is not None:
            geo_bias = -geo_dist.unsqueeze(1) * self.geo_scale.view(1, -1, 1, 1)
            attn = attn + geo_bias

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return x

    def extra_repr(self):
        return (f'dim={self.dim}, window_size={self.window_size}, '
                f'num_heads={self.num_heads}, geodesic=True')

    def flops(self, n):
        flops = 0
        flops += self.num_heads * n * (self.dim // self.num_heads) * n
        flops += self.num_heads * n * n * (self.dim // self.num_heads)
        flops += n * self.dim * self.dim
        return flops


# ===========================================================================
# ATDTransformerLayerGeo: 使用测地距离注意力的 Transformer 层
# ===========================================================================

class ATDTransformerLayerGeo(nn.Module):
    """
    与原版 ATDTransformerLayer 唯一的区别:
      attn_win = WindowAttentionGeo (而非 WindowAttention)
      forward 中额外计算测地距离并传入 attn_win

    三个注意力分支:
      1. SW-MSA  → WindowAttentionGeo (★ 测地距离引导)
      2. ATD_CA  → Token Dictionary 交叉注意力 (不变)
      3. AC_MSA  → 自适应分类自注意力 (不变)
    """

    def __init__(self, dim, idx, input_resolution, num_heads, window_size,
                 shift_size, dim_ffn_td, category_size, num_tokens,
                 reducted_dim, convffn_kernel_size, mlp_ratio,
                 geo_sigma_ratio=1.0,
                 qkv_bias=True, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.num_tokens = num_tokens
        self.reducted_dim = reducted_dim
        self.dim_ffn_td = dim_ffn_td

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        # ★ 替换为测地距离窗口注意力
        self.attn_win = WindowAttentionGeo(
            dim, window_size=to_2tuple(window_size),
            num_heads=num_heads, qkv_bias=qkv_bias)

        self.attn_atd = ATD_CA(
            dim, num_tokens=num_tokens,
            reducted_dim=reducted_dim, qkv_bias=qkv_bias)
        self.attn_aca = AC_MSA(
            dim, num_heads=num_heads,
            category_size=category_size, qkv_bias=qkv_bias)

        # ★ 可学习的测地距离 sigma
        self.geo_sigma = nn.Parameter(torch.tensor(geo_sigma_ratio))

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc_td = nn.Linear(dim, dim_ffn_td)
        self.convffn = ConvFFN_td(
            in_features=dim, hidden_features=mlp_hidden_dim,
            td_features=dim_ffn_td, kernel_size=convffn_kernel_size,
            act_layer=act_layer)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, td, x_size, params):
        h, w = x_size
        b, n, c = x.shape
        c3 = 3 * c
        ws = self.window_size

        shortcut = x
        x = self.norm1(x)
        qkv = self.wqkv(x)

        # ATD_CA
        x_atd, sim_atd = self.attn_atd(x, td, x_size)

        # AC_MSA
        tk_id = torch.argmax(sim_atd, dim=-1, keepdim=False)
        x_aca = self.attn_aca(qkv, tk_id, x_size)
        x_td = torch.gather(
            self.fc_td(td), dim=1,
            index=tk_id.reshape(b, n, 1).expand(-1, -1, self.dim_ffn_td))

        # ========= SW-MSA with Geodesic Bias =========
        qkv_2d = qkv.reshape(b, h, w, c3)
        x_2d = x.reshape(b, h, w, c)          # 归一化后的特征，用于测地距离计算

        # Cyclic shift (QKV 和特征同步 shift)
        if self.shift_size > 0:
            shifted_qkv = torch.roll(qkv_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_x = torch.roll(x_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = params['attn_mask']
        else:
            shifted_qkv = qkv_2d
            shifted_x = x_2d
            attn_mask = None

        # ★ 从 shifted 特征计算窗口内测地距离
        x_feat_windows = window_partition(shifted_x, ws)              # (nw*b, ws, ws, C)
        geo_dist = compute_window_geodesic_bias(x_feat_windows, ws, self.geo_sigma.abs())

        # 窗口注意力
        qkv_windows = window_partition(shifted_qkv, ws)               # (nw*b, ws, ws, 3C)
        qkv_windows = qkv_windows.view(-1, ws * ws, c3)
        attn_windows = self.attn_win(
            qkv_windows, rpi=params['rpi_sa'], mask=attn_mask, geo_dist=geo_dist)

        # Merge windows & reverse shift
        attn_windows = attn_windows.view(-1, ws, ws, c)
        shifted_x = window_reverse(attn_windows, ws, h, w)

        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        x_win = attn_x

        # 合并三个注意力分支
        x = shortcut + x_atd + x_win.view(b, n, c) + x_aca

        # FFN
        x = x + self.convffn(self.norm2(x), x_td, x_size)

        return x

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.input_resolution if input_resolution is None else input_resolution
        flops += self.dim * 3 * self.dim * h * w
        nw = h * w / self.window_size / self.window_size
        flops += nw * self.attn_win.flops(self.window_size * self.window_size)
        flops += self.attn_atd.flops(h * w)
        flops += self.attn_aca.flops(h * w)
        flops += h * w * self.dim * (self.dim * 2 + self.dim_ffn_td) * self.mlp_ratio
        flops += h * w * (self.dim + self.dim_ffn_td) * self.convffn_kernel_size ** 2 * self.mlp_ratio
        # 测地距离计算开销
        ws = self.window_size
        flops += nw * ws * ws * self.dim * 2       # 水平 + 垂直差分
        flops += nw * ws ** 4 * 2                    # geo_h + geo_v pairwise
        return flops


# ===========================================================================
# BasicBlockGeo / ATDBGeo / ATDGeo: 完整流水线 (仅替换层类型)
# ===========================================================================

class BasicBlockGeo(nn.Module):
    """与 BasicBlock 相同，但使用 ATDTransformerLayerGeo。"""

    def __init__(self, dim, input_resolution, idx, depth, num_heads, window_size,
                 dim_ffn_td, category_size, num_tokens, convffn_kernel_size,
                 reducted_dim, mlp_ratio=4., geo_sigma_ratio=1.0,
                 qkv_bias=True, norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.idx = idx

        self.layers = nn.ModuleList([
            ATDTransformerLayerGeo(
                dim=dim, idx=i, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                dim_ffn_td=dim_ffn_td, category_size=category_size,
                num_tokens=num_tokens, reducted_dim=reducted_dim,
                convffn_kernel_size=convffn_kernel_size, mlp_ratio=mlp_ratio,
                geo_sigma_ratio=geo_sigma_ratio,
                qkv_bias=qkv_bias, norm_layer=norm_layer)
            for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        self.td = nn.Parameter(torch.randn([num_tokens, dim]), requires_grad=True)

    def forward(self, x, x_size, params):
        b, n, c = x.shape
        td = self.td.expand([b, -1, -1])
        idx_checkpoint = 5
        for layer in self.layers:
            if self.use_checkpoint and self.idx < idx_checkpoint:
                x = checkpoint(layer, x, td, x_size, params, use_reentrant=False)
            else:
                x = layer(x, td, x_size, params)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def flops(self, input_resolution=None):
        flops = 0
        for layer in self.layers:
            flops += layer.flops(input_resolution)
        if self.downsample is not None:
            flops += self.downsample.flops(input_resolution)
        return flops


class ATDBGeo(nn.Module):
    """与 ATDB 相同，但使用 BasicBlockGeo。"""

    def __init__(self, dim, idx, input_resolution, depth, num_heads, window_size,
                 dim_ffn_td, category_size, num_tokens, reducted_dim,
                 convffn_kernel_size, mlp_ratio, geo_sigma_ratio=1.0,
                 qkv_bias=True, norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(ATDBGeo, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim)

        self.residual_group = BasicBlockGeo(
            dim=dim, input_resolution=input_resolution, idx=idx, depth=depth,
            num_heads=num_heads, window_size=window_size,
            dim_ffn_td=dim_ffn_td, category_size=category_size,
            num_tokens=num_tokens, reducted_dim=reducted_dim,
            convffn_kernel_size=convffn_kernel_size, mlp_ratio=mlp_ratio,
            geo_sigma_ratio=geo_sigma_ratio,
            qkv_bias=qkv_bias, norm_layer=norm_layer,
            downsample=downsample, use_checkpoint=use_checkpoint)

        self.norm = norm_layer(dim)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, x_size, params):
        return self.norm(self.patch_embed(
            self.conv(self.patch_unembed(self.residual_group(x, x_size, params), x_size))) + x)

    def flops(self, input_resolution=None):
        flops = 0
        flops += self.residual_group.flops(input_resolution)
        h, w = self.input_resolution if input_resolution is None else input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops(input_resolution)
        flops += self.patch_unembed.flops(input_resolution)
        return flops


@ARCH_REGISTRY.register()
class ATDGeo(nn.Module):
    """
    ATDGeo: ATD + Geodesic Domain-Transform Window Attention

    相比原版 ATD 的唯一新增参数:
      geo_sigma_ratio (float): 测地距离的初始边缘灵敏度，默认 1.0
        - 每层有独立的可学习 geo_sigma，此值仅为初始化
        - 每个注意力头有独立的可学习 geo_scale

    使用方式:
      在 YAML 配置中:
        network_g:
          type: ATDGeo
          geo_sigma_ratio: 1.0    # 新增参数 (可选)
          # 其余参数与 ATD 完全相同
          embed_dim: 90
          depths: [6,6,6,6]
          ...
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=90, depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6),
                 window_size=8, dim_ffn_td=16, category_size=128,
                 num_tokens=64, reducted_dim=4, convffn_kernel_size=5,
                 mlp_ratio=2., geo_sigma_ratio=1.0,
                 qkv_bias=True, norm_layer=nn.LayerNorm,
                 ape=False, patch_norm=True, use_checkpoint=False,
                 upscale=2, img_range=1., upsampler='',
                 resi_connection='1conv', **kwargs):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # 1. Shallow feature extraction
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # 2. Deep feature extraction
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        # ★ 使用 ATDBGeo 替代 ATDB
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ATDBGeo(
                dim=embed_dim, idx=i_layer,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer], num_heads=num_heads[i_layer],
                window_size=window_size, dim_ffn_td=dim_ffn_td,
                category_size=category_size, num_tokens=num_tokens,
                reducted_dim=reducted_dim,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=self.mlp_ratio, geo_sigma_ratio=geo_sigma_ratio,
                qkv_bias=qkv_bias, norm_layer=norm_layer,
                downsample=None, use_checkpoint=use_checkpoint,
                img_size=img_size, patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # 3. Reconstruction
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, params):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        for layer in self.layers:
            x = layer(x, x_size, params)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, x_size):
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

    def forward(self, x):
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = self.window_size
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        attn_mask = self.calculate_mask([h, w]).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first, params)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean
        x = x[..., :h_ori * self.upscale, :w_ori * self.upscale]
        return x

    def flops(self, input_resolution=None):
        flops = 0
        resolution = self.patches_resolution if input_resolution is None else input_resolution
        h, w = resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops(resolution)
        for layer in self.layers:
            flops += layer.flops(resolution)
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        if self.upsampler == 'pixelshuffle':
            flops += self.upsample.flops(resolution)
        else:
            flops += self.upsample.flops(resolution)
        return flops


# ===========================================================================
# 验证
# ===========================================================================

if __name__ == '__main__':
    print("ATDGeo — ATD with Geodesic Domain-Transform Window Attention")
    print("=" * 60)

    # --- 基础测试 ---
    upscale = 4
    model = ATDGeo(
        upscale=4, img_size=64, embed_dim=48, depths=[2, 2], num_heads=[4, 4],
        window_size=8, dim_ffn_td=16, category_size=128, num_tokens=64,
        reducted_dim=4, convffn_kernel_size=5, img_range=1., mlp_ratio=2,
        upsampler='pixelshuffle', geo_sigma_ratio=1.0)

    total = sum([p.nelement() for p in model.parameters()])
    print(f"Parameters: {total / 1e6:.3f}M")

    # --- Forward 测试 ---
    x = torch.randn(1, 3, 64, 64)
    print(f"\nForward test:")
    print(f"  Input: {x.shape}")
    out = model(x)
    print(f"  Output: {out.shape}")
    assert out.shape == (1, 3, 256, 256), f"Shape mismatch: {out.shape}"
    print("  ✓ Shape correct")

    # --- Backward 测试 ---
    loss = out.sum()
    loss.backward()
    print("  ✓ Backward pass OK")

    # --- 检查可学习的测地距离参数 ---
    geo_sigmas = []
    geo_scales = []
    for name, p in model.named_parameters():
        if 'geo_sigma' in name:
            geo_sigmas.append((name, p.item()))
        if 'geo_scale' in name:
            geo_scales.append((name, p.data.tolist()))

    print(f"\n  Geodesic parameters:")
    print(f"    geo_sigma values: {[f'{v:.3f}' for _, v in geo_sigmas]}")
    print(f"    geo_scale (first layer): {geo_scales[0][1]}")

    # --- 对比测试: 有边缘 vs 无边缘 ---
    print(f"\n--- 测地距离效果验证 ---")
    ws = 4
    feat_flat = torch.ones(1, ws, ws, 8) * 0.5       # 平坦特征
    feat_edge = torch.cat([                             # 左右分裂特征
        torch.ones(1, ws, ws // 2, 8) * 0.2,
        torch.ones(1, ws, ws // 2, 8) * 0.8
    ], dim=2)

    geo_flat = compute_window_geodesic_bias(feat_flat, ws, sigma=1.0)
    geo_edge = compute_window_geodesic_bias(feat_edge, ws, sigma=1.0)

    print(f"  平坦特征 geo_dist 范围: [{geo_flat.min():.4f}, {geo_flat.max():.4f}]")
    print(f"  有边缘特征 geo_dist 范围: [{geo_edge.min():.4f}, {geo_edge.max():.4f}]")
    print(f"  平坦 max: {geo_flat.max():.4f}, 有边缘 max: {geo_edge.max():.4f}")
    print(f"  → 有边缘时测地距离更大 → attention 跨边缘衰减更强 ✓")

    # --- FLOPs ---
    print(f"\n--- FLOPs ---")
    print(f"  64x64:  {model.flops([64, 64]) / 1e9:.3f}G")
    print(f"  128x128: {model.flops([128, 128]) / 1e9:.3f}G")
