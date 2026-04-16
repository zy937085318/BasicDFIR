'''
MPv2_3d: Shared TD Across Blocks (P6 ablation)

Based on MPv2_3_arch, sharing a single TD across all 4 BasicBlocks.

=== Improvement over MPv2_3 ===

Problem (P6): In MPv2_3, each of the 4 BasicBlocks has its own independent TD
(4 x 64 x 96 = 24,576 params). This may introduce redundancy — all blocks learn
separate memory dictionaries for the same SR task.

Solution: Share a single TD at the MPv2_3d_arch level, passed to each ATDB/BasicBlock.
Each block still retains its own LoRA matrices (lora_A, lora_B) and alpha_proj,
so blocks can adapt the shared TD differently.

Key difference from MPv2_3:
- self.td moved from BasicBlock to MPv2_3d_arch (self.shared_td)
- ATDB/BasicBlock accept td as forward parameter instead of owning it
- Parameter reduction: TD from 4x64x96=24576 to 1x64x96=6144
- Each block still has unique lora_A, lora_B, alpha_proj for block-specific adaptation

ATDTransformerLayer timestep injection unchanged (simple additive).
x + x_lq additive fusion is enabled.
'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple, trunc_normal_
from torch.utils.checkpoint import checkpoint

from basicsr.utils.registry import ARCH_REGISTRY


# ==================== Timestep Embedding ====================

def get_sinusoidal_embedding(timesteps, embedding_dim):
    if len(timesteps.size()) != 1:
        timesteps = timesteps.squeeze()
    timesteps = timesteps.to(torch.get_default_dtype())
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), "constant", 0)
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, t):
        return self.net(get_sinusoidal_embedding(t, self.embedding_dim))


# ==================== ATD Core Modules ====================

def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def feature_shuffle(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape
    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)
    return torch.gather(x, dim=dim - 1, index=index)


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super().__init__()
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


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        self.proj = nn.Linear(dim, dim)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, rpi, mask=None):
        b_, n, c3 = qkv.shape
        c = c3 // 3
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        attn = attn + relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.softmax(attn)
        return self.proj((attn @ v).transpose(1, 2).reshape(b_, n, c))


class ATD_CA(nn.Module):
    def __init__(self, dim, num_tokens=64, reducted_dim=10, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.dr = reducted_dim
        self.wq = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.scale = nn.Parameter(torch.ones([1]), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, td, x_size):
        b, n, c = x.shape
        b, m, c = td.shape
        q = self.wq(x)
        k = self.wk(td)
        v = self.wv(td)
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        attn = attn * (1 + torch.clamp(self.scale, 0, 3) * np.log(self.num_tokens))
        attn = self.softmax(attn)
        return (attn @ v).reshape(b, n, c), attn


class AC_MSA(nn.Module):
    def __init__(self, dim, num_heads=4, category_size=128, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.category_size = category_size
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.scale = (dim // self.num_heads) ** -0.5
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
        paded_qkv = torch.cat((shuffled_qkv, torch.flip(shuffled_qkv[:, n - pad_n:n, :], dims=[1])), dim=1)
        y = paded_qkv.reshape(b, -1, gs, c3)
        qkv_ = y.reshape(b, ng, gs, 3, self.num_heads, c // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv_[0], qkv_[1], qkv_[2]
        attn = self.softmax((q @ k.transpose(-2, -1)) * self.scale)
        y = (attn @ v).permute(0, 1, 3, 2, 4).reshape(b, n + pad_n, c)[:, :n, :]
        return self.proj(feature_shuffle(y, tk_id_inv))


class ATDTransformerLayer(nn.Module):
    """ATD layer with simple additive timestep injection (same as MPv2_3)."""
    def __init__(self, dim, idx, input_resolution, num_heads, window_size, shift_size,
                 dim_ffn_td, category_size, num_tokens, reducted_dim,
                 convffn_kernel_size, mlp_ratio, qkv_bias=True,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, temb_ch=0):
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
        self.temb_ch = temb_ch

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self.attn_win = WindowAttention(dim, window_size=to_2tuple(window_size),
                                        num_heads=num_heads, qkv_bias=qkv_bias)
        self.attn_atd = ATD_CA(dim, num_tokens=num_tokens, reducted_dim=reducted_dim, qkv_bias=qkv_bias)
        self.attn_aca = AC_MSA(dim, num_heads=num_heads, category_size=category_size, qkv_bias=qkv_bias)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc_td = nn.Linear(dim, dim_ffn_td)
        self.convffn = ConvFFN_td(in_features=dim, hidden_features=mlp_hidden_dim,
                                  td_features=dim_ffn_td, kernel_size=convffn_kernel_size, act_layer=act_layer)

        if temb_ch > 0:
            self.temb_proj = nn.Linear(temb_ch, dim)
            nn.init.zeros_(self.temb_proj.weight)
            nn.init.zeros_(self.temb_proj.bias)

    def forward(self, x, td, x_size, params, temb=None):
        h, w = x_size
        b, n, c = x.shape
        c3 = 3 * c

        shortcut = x
        x = self.norm1(x)
        qkv = self.wqkv(x)

        x_atd, sim_atd = self.attn_atd(x, td, x_size)
        tk_id = torch.argmax(sim_atd, dim=-1, keepdim=False)
        x_aca = self.attn_aca(qkv, tk_id, x_size)
        x_td = torch.gather(self.fc_td(td), dim=1,
                             index=tk_id.reshape(b, n, 1).expand(-1, -1, self.dim_ffn_td))

        # SW-MSA
        qkv = qkv.reshape(b, h, w, c3)
        if self.shift_size > 0:
            shifted_qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = params['attn_mask']
        else:
            shifted_qkv = qkv
            attn_mask = None
        x_windows = window_partition(shifted_qkv, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c3)
        attn_windows = self.attn_win(x_windows, rpi=params['rpi_sa'], mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x

        x = shortcut + x_atd + attn_x.view(b, n, c) + x_aca

        # Simple additive timestep injection
        if temb is not None and self.temb_ch > 0:
            x = x + self.temb_proj(temb)[:, None, :]

        x = x + self.convffn(self.norm2(x), x_td, x_size)
        return x


class BasicBlock(nn.Module):
    """BasicBlock with LoRA-TD, TD passed from outside (shared across blocks)."""
    def __init__(self, dim, input_resolution, idx, depth, num_heads, window_size,
                 dim_ffn_td, category_size, num_tokens, convffn_kernel_size,
                 reducted_dim, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False, temb_ch=0, lora_rank=8):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.idx = idx

        self.layers = nn.ModuleList([
            ATDTransformerLayer(
                dim=dim, idx=i, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                dim_ffn_td=dim_ffn_td, category_size=category_size, num_tokens=num_tokens,
                convffn_kernel_size=convffn_kernel_size, reducted_dim=reducted_dim,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer, temb_ch=temb_ch)
            for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        # No self.td — TD is passed from outside (shared)

        # LoRA decomposition (per-block, different adaptation of shared TD)
        self.lora_rank = lora_rank
        self.lora_A = nn.Parameter(torch.randn(lora_rank, dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(dim, lora_rank))

        # Condition -> per-rank activation (per-block)
        cond_dim = dim + max(temb_ch, 0)
        self.alpha_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, lora_rank, bias=False),
        )
        nn.init.zeros_(self.alpha_proj[-1].weight)

    def forward(self, x, x_size, params, temb=None, x_lq=None, td=None):
        b, n, c = x.shape

        # Use shared TD or fallback
        if td is None:
            td = torch.zeros(b, 0, c, device=x.device)
        else:
            td = td.unsqueeze(0).expand(b, -1, -1)  # [B, M, C]

        # LoRA memory extraction (per-block adaptation of shared TD)
        if x_lq is not None and td.shape[1] > 0:
            lq_global = x_lq.mean(dim=1)
            if temb is not None:
                cond = torch.cat([lq_global, temb], dim=-1)
            else:
                cond = lq_global
            alpha = torch.sigmoid(self.alpha_proj(cond))
            td_lowrank = td @ self.lora_A.T
            td_lowrank = td_lowrank * alpha.unsqueeze(1)
            delta_td = td_lowrank @ self.lora_B.T
            td = td + delta_td

        idx_checkpoint = 5
        for layer in self.layers:
            if self.use_checkpoint and self.idx < idx_checkpoint:
                x = checkpoint(layer, x, td, x_size, params, temb, use_reentrant=False)
            else:
                x = layer(x, td, x_size, params, temb)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class ATDB(nn.Module):
    def __init__(self, dim, idx, input_resolution, depth, num_heads, window_size,
                 dim_ffn_td, category_size, num_tokens, reducted_dim, convffn_kernel_size,
                 mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False, img_size=224, patch_size=4, resi_connection='1conv',
                 temb_ch=0, lora_rank=8):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim)
        self.residual_group = BasicBlock(
            dim=dim, input_resolution=input_resolution, idx=idx, depth=depth,
            num_heads=num_heads, window_size=window_size, num_tokens=num_tokens,
            dim_ffn_td=dim_ffn_td, category_size=category_size, reducted_dim=reducted_dim,
            convffn_kernel_size=convffn_kernel_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            norm_layer=norm_layer, downsample=downsample, use_checkpoint=use_checkpoint,
            temb_ch=temb_ch, lora_rank=lora_rank)
        self.norm = norm_layer(dim)
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, x_size, params, temb=None, x_lq=None, td=None):
        x = x + x_lq
        return self.norm(
            self.patch_embed(self.conv(
                self.patch_unembed(self.residual_group(x, x_size, params, temb, x_lq, td=td), x_size))) + x)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        return x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        self.scale = scale
        self.num_feat = num_feat
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported.')
        super().__init__(*m)


@ARCH_REGISTRY.register()
class MPv2_3d_arch(nn.Module):
    """MPv2_3d: Shared TD Across Blocks + LoRA-TD.

    Key change from MPv2_3: Single shared TD at top level, passed to all blocks.
    Each block still has its own LoRA matrices for block-specific adaptation.
    """
    def __init__(self, img_size=64, patch_size=1, in_chans=3, out_chans=3,
                 embed_dim=90, depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6),
                 window_size=8, dim_ffn_td=16, category_size=128, num_tokens=64,
                 reducted_dim=4, convffn_kernel_size=5, mlp_ratio=2.,
                 qkv_bias=True, norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='',
                 resi_connection='1conv', temb_ch=0, embedding_dim=256, downscale=2,
                 lora_rank=8, **kwargs):
        super().__init__()
        num_feat = 64
        self.img_range = img_range
        self.mean = torch.Tensor((0.4488, 0.4371, 0.4040) if in_chans == 3 else (0,)).view(1, 3 if in_chans == 3 else 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.embed_dim = embed_dim
        self.temb_ch = temb_ch
        self.embedding_dim = embedding_dim
        self.downscale = downscale if upscale == 1 else 1

        # Timestep Embedding
        if temb_ch > 0:
            self.temb_net = TimestepEmbedding(embedding_dim, temb_ch, temb_ch)

        # Shallow Feature Extraction
        if upscale == 1:
            ds = self.downscale
            in_ch_per_branch = in_chans * ds ** 2

            self.pixel_unshuffle_xt = nn.PixelUnshuffle(ds)
            self.conv_xt = nn.Conv2d(in_ch_per_branch, embed_dim, 3, 1, 1)
            self.conv_lr = nn.Conv2d(in_chans, embed_dim, 3, 1, upscale)

            img_size = img_size // ds
        else:
            img_size = img_size // upscale
            self.pixel_unshuffle = nn.PixelUnshuffle(upscale)
            self.conv_first = nn.Conv2d(in_chans * upscale ** 2, embed_dim, 3, 1, 1, groups=upscale)

        # Deep Feature Extraction
        self.num_layers = len(depths)
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim, embed_dim,
                                      norm_layer if self.patch_norm else None)
        self.lq_patch_embed = PatchEmbed(img_size, patch_size, embed_dim, embed_dim,
                                          norm_layer if self.patch_norm else None)
        self.patches_resolution = self.patch_embed.patches_resolution
        self.patch_unembed = PatchUnEmbed(img_size, patch_size, embed_dim, embed_dim,
                                          norm_layer if self.patch_norm else None)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.register_buffer('relative_position_index_SA', self.calculate_rpi_sa())

        self.layers = nn.ModuleList([
            ATDB(dim=embed_dim, idx=i, input_resolution=tuple(self.patches_resolution),
                 depth=depths[i], num_heads=num_heads[i], window_size=window_size,
                 dim_ffn_td=dim_ffn_td, category_size=category_size, num_tokens=num_tokens,
                 reducted_dim=reducted_dim, convffn_kernel_size=convffn_kernel_size,
                 mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer,
                 downsample=None, use_checkpoint=use_checkpoint, img_size=img_size,
                 patch_size=patch_size, resi_connection=resi_connection, temb_ch=temb_ch,
                 lora_rank=lora_rank)
            for i in range(self.num_layers)
        ])
        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # Reconstruction
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, out_chans, 3, 1, 1)
        elif self.upscale == 1 and self.downscale > 1:
            self.conv_last = nn.Conv2d(embed_dim, out_chans * self.downscale ** 2, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.downscale)
        else:
            self.conv_last = nn.Conv2d(embed_dim, out_chans, 3, 1, 1)

        # ===== Shared TD across all blocks =====
        self.shared_td = nn.Parameter(torch.randn([num_tokens, embed_dim]), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, params, temb=None, x_lq=None):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x_lq = self.lq_patch_embed(x_lq)
        if self.ape:
            x = x + self.absolute_pos_embed
            x_lq = x_lq + self.absolute_pos_embed
        for layer in self.layers:
            x = layer(x, x_size, params, temb, x_lq, td=self.shared_td)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def calculate_rpi_sa(self):
        coords = torch.stack(torch.meshgrid([torch.arange(self.window_size)] * 2))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = (coords_flatten[:, :, None] - coords_flatten[:, None, :]).permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        return relative_coords.sum(-1)

    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -(self.window_size // 2)),
                    slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -(self.window_size // 2)),
                    slice(-(self.window_size // 2), None))
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x, c=None, temp=None):
        self.mean = self.mean.type_as(x)

        B, C, H, W = x.shape
        if temp is None:
            temp = torch.zeros([B]).to(x.device)
        temb = self.temb_net(temp) if self.temb_ch > 0 else None

        # Split & Shallow Feature Extraction
        if self.upscale == 1:
            x_t = x
            x_lq = c if c is not None else nn.functional.interpolate(
                x, scale_factor=1 / self.downscale, mode='bilinear', align_corners=False)

            x_lq = (x_lq - self.mean) * self.img_range

            x = self.conv_xt(self.pixel_unshuffle_xt(x_t))
            x_lq = self.conv_lr(x_lq)
            assert x_lq.shape[2:] == x.shape[2:], f"x_lq.shape: {x_lq.shape}, x.shape: {x.shape}"

        else:
            x = (x - self.mean) * self.img_range
            x = self.conv_first(self.pixel_unshuffle(x))
            x_lq = x

        # Padding
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = self.window_size
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]
        x_lq = torch.cat([x_lq, torch.flip(x_lq, [2])], 2)[:, :, :h, :]
        x_lq = torch.cat([x_lq, torch.flip(x_lq, [3])], 3)[:, :, :, :w]

        attn_mask = self.calculate_mask([h, w]).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        # Deep Feature Extraction (Shared TD + LoRA-TD)
        x = self.conv_after_body(self.forward_features(x, params, temb, x_lq=x_lq)) + x

        # Reconstruction
        if self.upsampler == 'pixelshuffle':
            x = self.conv_last(self.upsample(self.conv_before_upsample(x)))
        elif self.upscale == 1 and self.downscale > 1:
            x = self.pixel_shuffle(self.conv_last(x))
        else:
            x = self.conv_last(x)

        x = x / self.img_range + self.mean
        x = x[..., :h_ori * self.upscale * self.downscale, :w_ori * self.upscale * self.downscale]
        return x


if __name__ == '__main__':
    model = MPv2_3d_arch(
        upscale=1, img_size=256, embed_dim=96,
        depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6], window_size=8,
        dim_ffn_td=16, category_size=128, num_tokens=64, reducted_dim=4,
        convffn_kernel_size=5, img_range=1., mlp_ratio=2, upsampler='',
        temb_ch=96, embedding_dim=96, downscale=4, lora_rank=8,
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total / 1e6:.3f}M")

    x = torch.randn(2, 6, 256, 256)
    temp = torch.rand(2)
    out = model(x, None, temp)
    print(f"Input: {x.shape} -> Output: {out.shape}")
