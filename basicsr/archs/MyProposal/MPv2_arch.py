'''
TC-ATD: Timestep-Conditioned Adaptive Token Dictionary for Flow Matching SR
基于 MPv1_1_arch 的两个创新点:
1. Timestep-Adaptive Shallow Feature Extraction (t-gated dual-branch fusion)
2. Timestep-Conditioned Token Dictionary (AdaLN-Zero modulation on TD)
'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple, trunc_normal_
from torch.utils.checkpoint import checkpoint

from basicsr.utils.registry import ARCH_REGISTRY


# ==================== Timestep Embedding Modules ====================

def get_sinusoidal_embedding(timesteps, embedding_dim):
    """Sinusoidal timestep embedding (DiT-style, cos-then-sin)."""
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
    """Sinusoidal embedding → 2-layer MLP."""
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, t):
        temb = get_sinusoidal_embedding(t, self.embedding_dim)
        return self.net(temb)


class AdaLNModulation(nn.Module):
    """Adaptive Layer Norm: x' = (1 + scale) * LN(x) + shift, conditioned on temb.
    Zero-initialized for stable training start."""
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * dim),
        )
        # Zero-init for residual-like behavior at start
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x, cond):
        """
        x: [B, ..., C] — features to modulate
        cond: [B, cond_dim] — timestep embedding
        """
        scale, shift = self.proj(cond).chunk(2, dim=-1)  # [B, C] each
        # Reshape for broadcasting over spatial/token dims
        for _ in range(x.dim() - 2):
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return self.norm(x) * (1 + scale) + shift

# ==================== End of Timestep Embedding Modules ====================


# Shuffle operation for Categorization and UnCategorization operations.
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
    return torch.gather(x, dim=dim - 1, index=index)


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1, groups=hidden_features),
            nn.GELU())
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
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5,
                 act_layer=nn.GELU):
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


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
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
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
            -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'

    def flops(self, n):
        flops = 0
        flops += self.num_heads * n * (self.dim // self.num_heads) * n
        flops += self.num_heads * n * n * (self.dim // self.num_heads)
        flops += n * self.dim * self.dim
        return flops


class ATD_CA(nn.Module):
    """Cross-attention between input features and Token Dictionary."""
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
        h, w = x_size
        b, n, c = x.shape
        b, m, c = td.shape

        q = self.wq(x)
        k = self.wk(td)
        v = self.wv(td)

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        scale = 1 + torch.clamp(self.scale, 0, 3) * np.log(self.num_tokens)
        attn = attn * scale
        attn = self.softmax(attn)

        x = (attn @ v).reshape(b, n, c)
        return x, attn

    def flops(self, n):
        flops = 0
        flops += n * self.dim * self.dr
        flops += self.num_tokens * self.dim * self.dr
        flops += self.num_tokens * self.dim * self.dim
        flops += n * self.dim * self.dr
        flops += n * self.num_tokens * self.dim
        return flops


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

        qkv = y.reshape(b, ng, gs, 3, self.num_heads, c // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))
        attn = attn * self.scale
        attn = self.softmax(attn)

        y = (attn @ v).permute(0, 1, 3, 2, 4).reshape(b, n + pad_n, c)[:, :n, :]
        x = feature_shuffle(y, tk_id_inv)
        x = self.proj(x)
        return x

    def flops(self, n):
        flops = 0
        flops += n * self.dim * self.category_size
        flops += n * self.dim * self.category_size
        flops += n * self.dim * self.dim
        return flops


class ATDTransformerLayer(nn.Module):
    """ATD Transformer Layer with Timestep Conditioning (TC-ATD).

    Two timestep-conditioned modifications:
    1. Token Dictionary is modulated by temb via AdaLN before cross-attention
    2. Additive temb injection after multi-branch attention aggregation
    """
    def __init__(self,
                 dim,
                 idx,
                 input_resolution,
                 num_heads,
                 window_size,
                 shift_size,
                 dim_ffn_td,
                 category_size,
                 num_tokens,
                 reducted_dim,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 temb_ch=0,
                 ):
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

        self.attn_win = WindowAttention(
            self.dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.attn_atd = ATD_CA(
            self.dim,
            num_tokens=num_tokens,
            reducted_dim=reducted_dim,
            qkv_bias=qkv_bias,
        )
        self.attn_aca = AC_MSA(
            self.dim,
            num_heads=num_heads,
            category_size=category_size,
            qkv_bias=qkv_bias,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc_td = nn.Linear(dim, dim_ffn_td)
        self.convffn = ConvFFN_td(in_features=dim, hidden_features=mlp_hidden_dim,
                                  td_features=dim_ffn_td, kernel_size=convffn_kernel_size,
                                  act_layer=act_layer)

        # ===== TC-ATD: Timestep conditioning modules =====
        if temb_ch > 0:
            # Innovation 2: AdaLN on Token Dictionary
            self.td_adaln = AdaLNModulation(dim, temb_ch)
            # Additive timestep injection
            self.temb_proj = nn.Linear(temb_ch, dim)
            nn.init.zeros_(self.temb_proj.weight)
            nn.init.zeros_(self.temb_proj.bias)

    def forward(self, x, td, x_size, params, temb=None):
        h, w = x_size
        b, n, c = x.shape
        c3 = 3 * c

        # ===== TC-ATD: Modulate Token Dictionary with timestep =====
        if temb is not None and self.temb_ch > 0:
            td = self.td_adaln(td, temb)  # [B, M, C] timestep-conditioned TD

        shortcut = x
        x = self.norm1(x)
        qkv = self.wqkv(x)

        # ATD_CA with (now time-conditioned) Token Dictionary
        x_atd, sim_atd = self.attn_atd(x, td, x_size)

        # AC_MSA (category adapts automatically because tk_id comes from modulated td)
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
        x_win = attn_x

        x = shortcut + x_atd + x_win.view(b, n, c) + x_aca

        # ===== TC-ATD: Additive timestep injection =====
        if temb is not None and self.temb_ch > 0:
            x = x + self.temb_proj(temb)[:, None, :]

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
        return flops


class BasicBlock(nn.Module):
    """BasicBlock containing Token Dictionary and multiple ATDTransformerLayers."""
    def __init__(self,
                 dim,
                 input_resolution,
                 idx,
                 depth,
                 num_heads,
                 window_size,
                 dim_ffn_td,
                 category_size,
                 num_tokens,
                 convffn_kernel_size,
                 reducted_dim,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 temb_ch=0,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.idx = idx

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                ATDTransformerLayer(
                    dim=dim,
                    idx=i,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    dim_ffn_td=dim_ffn_td,
                    category_size=category_size,
                    num_tokens=num_tokens,
                    convffn_kernel_size=convffn_kernel_size,
                    reducted_dim=reducted_dim,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    temb_ch=temb_ch,
                )
            )

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        # Token Dictionary (static learnable, modulated by temb in each layer)
        self.td = nn.Parameter(torch.randn([num_tokens, dim]), requires_grad=True)

    def forward(self, x, x_size, params, temb=None):
        b, n, c = x.shape
        td = self.td.expand([b, -1, -1])
        idx_checkpoint = 5
        for layer in self.layers:
            if self.use_checkpoint and self.idx < idx_checkpoint:
                x = checkpoint(layer, x, td, x_size, params, temb, use_reentrant=False)
            else:
                x = layer(x, td, x_size, params, temb)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self, input_resolution=None):
        flops = 0
        for layer in self.layers:
            flops += layer.flops(input_resolution)
        if self.downsample is not None:
            flops += self.downsample.flops(input_resolution)
        return flops


class ATDB(nn.Module):
    def __init__(self,
                 dim,
                 idx,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 dim_ffn_td,
                 category_size,
                 num_tokens,
                 reducted_dim,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv',
                 temb_ch=0,
                 ):
        super(ATDB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim)

        self.residual_group = BasicBlock(
            dim=dim,
            input_resolution=input_resolution,
            idx=idx,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            num_tokens=num_tokens,
            dim_ffn_td=dim_ffn_td,
            category_size=category_size,
            reducted_dim=reducted_dim,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            temb_ch=temb_ch,
        )
        self.norm = norm_layer(dim)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, x_size, params, temb=None):
        return self.norm(
            self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, params, temb), x_size))) + x)

    def flops(self, input_resolution=None):
        flops = 0
        flops += self.residual_group.flops(input_resolution)
        h, w = self.input_resolution if input_resolution is None else input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops(input_resolution)
        flops += self.patch_unembed.flops(input_resolution)
        return flops


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


class SE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(self.pool(x).view(b, c))
        return x * w.view(b, c, 1, 1)


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
        super(Upsample, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        x, y = input_resolution
        if (self.scale & (self.scale - 1)) == 0:
            flops += self.num_feat * 4 * self.num_feat * 9 * x * y * int(math.log(self.scale, 2))
        else:
            flops += self.num_feat * 9 * self.num_feat * 9 * x * y
        return flops


@ARCH_REGISTRY.register()
class MPv2_arch(nn.Module):
    """TC-ATD: Timestep-Conditioned Adaptive Token Dictionary for Flow Matching SR.

    Two innovations over MPv1_1 (ATD baseline):
    1. Timestep-Adaptive Shallow Feature Extraction:
       - Dual-branch conv (x_t / lq_bicubic) with t-gated fusion
       - t→0: focus on lq structure; t→1: focus on x_t details
    2. Timestep-Conditioned Token Dictionary (TC-ATD):
       - AdaLN-Zero modulation on Token Dictionary per timestep
       - Memory retrieval, category attention, feature fusion all adapt to t

    Forward signature: (x, c=None, temp=None) for JiTModel compatibility.
    """
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 out_chans=3,
                 embed_dim=90,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=8,
                 dim_ffn_td=16,
                 category_size=128,
                 num_tokens=64,
                 reducted_dim=4,
                 convffn_kernel_size=5,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 # ===== TC-ATD new parameters =====
                 temb_ch=0,
                 embedding_dim=256,
                 downscale=2,
                 **kwargs):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = out_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.embed_dim = embed_dim
        self.temb_ch = temb_ch
        self.embedding_dim = embedding_dim
        self.downscale = downscale if upscale == 1 else 1  # only for JiT branch

        # ===== Timestep Embedding =====
        if temb_ch > 0:
            self.temb_net = TimestepEmbedding(
                embedding_dim=embedding_dim,
                hidden_dim=temb_ch,
                output_dim=temb_ch,
            )

        # ===== 1. Shallow Feature Extraction =====
        if upscale == 1:
            # Flow matching mode: input is 6ch (x_t + lq_bicubic)
            ds = self.downscale
            in_ch_per_branch = num_in_ch * ds ** 2  # channels after PixelUnshuffle

            # PixelUnshuffle per branch for spatial reduction
            self.pixel_unshuffle_xt = nn.PixelUnshuffle(ds)
            self.pixel_unshuffle_lr = nn.PixelUnshuffle(ds)

            # Innovation 1: Dual-branch conv with t-gated fusion
            self.conv_xt = nn.Conv2d(in_ch_per_branch, embed_dim, 3, 1, 1)
            self.conv_lr = nn.Conv2d(in_ch_per_branch, embed_dim, 3, 1, 1)
            if temb_ch > 0:
                # t-gated fusion: gate=0 → use LR, gate=1 → use x_t
                self.t_gate = nn.Linear(temb_ch, embed_dim)
                nn.init.zeros_(self.t_gate.weight)
                nn.init.zeros_(self.t_gate.bias)
                # T-adaptive modulation on fused features
                self.shallow_adaln = AdaLNModulation(embed_dim, temb_ch)

            # Adjust img_size to reduced resolution for PatchEmbed
            img_size = img_size // ds
        else:
            # Standard SR mode: PixelUnshuffle + SE + grouped conv
            img_size = img_size // upscale
            self.pixel_unshuffle = nn.PixelUnshuffle(upscale)
            self.se = SE(num_in_ch * upscale ** 2)
            self.conv_first = nn.Sequential(
                nn.Conv2d(num_in_ch * upscale ** 2, embed_dim, 3, 1, 1, groups=upscale))
            if temb_ch > 0:
                self.shallow_adaln_conv = AdaLNModulation(embed_dim, temb_ch)

        # ===== 2. Deep Feature Extraction =====
        self.num_layers = len(depths)
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ATDB(
                dim=embed_dim,
                idx=i_layer,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                dim_ffn_td=dim_ffn_td,
                category_size=category_size,
                num_tokens=num_tokens,
                reducted_dim=reducted_dim,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                temb_ch=temb_ch,
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ===== 3. Reconstruction =====
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            from basicsr.archs.atd_arch import UpsampleOneStep
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upscale == 1 and self.downscale > 1:
            # Flow matching mode with PixelShuffle to restore resolution
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch * self.downscale ** 2, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.downscale)
        else:
            # Flow matching mode without downscale: direct 3ch output
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

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

    def forward_features(self, x, params, temb=None):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed

        for layer in self.layers:
            x = layer(x, x_size, params, temb)

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
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, c=None, temp=None):
        """
        Args:
            x: [B, C, H, W] — for flow matching (upscale=1): 6ch (x_t + lq_bicubic)
               for standard SR (upscale>1): 3ch LR image
            c: unused, reserved for condition
            temp: [B] — timestep tensor for flow matching
        """
        self.mean = self.mean.type_as(x)
        B = x.shape[0]

        # Timestep embedding
        temb = None
        if temp is not None and self.temb_ch > 0:
            temb = self.temb_net(temp)  # [B, temb_ch]

        # ===== 1. Shallow Feature Extraction =====
        if self.upscale == 1:
            # Flow matching mode
            # Handle 3→6 channel auto-expansion
            if x.shape[1] == 3:
                x = torch.cat([x, x], dim=1)

            x_xt, x_lr = x[:, :3, :, :], x[:, 3:, :, :]

            # Normalize
            x_xt = (x_xt - self.mean) * self.img_range
            x_lr = (x_lr - self.mean) * self.img_range

            # PixelUnshuffle each branch for spatial reduction
            x_xt = self.pixel_unshuffle_xt(x_xt)  # [B, 3*ds², H/ds, W/ds]
            x_lr = self.pixel_unshuffle_lr(x_lr)  # [B, 3*ds², H/ds, W/ds]

            # Dual-branch feature extraction
            feat_xt = self.conv_xt(x_xt)  # [B, embed_dim, H/ds, W/ds]
            feat_lr = self.conv_lr(x_lr)  # [B, embed_dim, H/ds, W/ds]

            if temb is not None:
                # Innovation 1: t-gated fusion
                gate = torch.sigmoid(self.t_gate(temb))  # [B, embed_dim]
                x = feat_lr * (1 - gate[:, :, None, None]) + feat_xt * gate[:, :, None, None]
                # T-adaptive modulation on fused features
                B_f, C_f, H_f, W_f = x.shape
                x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
                x_flat = self.shallow_adaln(x_flat, temb)  # [B, H*W, C]
                x = x_flat.transpose(1, 2).view(B_f, C_f, H_f, W_f)
            else:
                x = feat_xt + feat_lr
        else:
            # Standard SR mode
            x = (x - self.mean) * self.img_range
            x = self.pixel_unshuffle(x)
            x = self.se(x)

            # Padding
            h_ori, w_ori = x.size()[-2], x.size()[-1]
            mod = self.window_size
            h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
            w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
            h, w = h_ori + h_pad, w_ori + w_pad
            x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
            x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]

            x = self.conv_first(x)

            if temb is not None:
                B_f, C_f, H_f, W_f = x.shape
                x_flat = x.flatten(2).transpose(1, 2)
                x_flat = self.shallow_adaln_conv(x_flat, temb)
                x = x_flat.transpose(1, 2).view(B_f, C_f, H_f, W_f)

        # ===== Padding for window attention =====
        if self.upscale == 1:
            h_ori, w_ori = x.size()[-2], x.size()[-1]
            mod = self.window_size
            h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
            w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
            h, w = h_ori + h_pad, w_ori + w_pad
            x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
            x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]

        attn_mask = self.calculate_mask([h, w]).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        # ===== 2. Deep Feature Extraction (TC-ATD) =====
        x = self.conv_after_body(self.forward_features(x, params, temb)) + x

        # ===== 3. Reconstruction =====
        if self.upsampler == 'pixelshuffle':
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            x = self.upsample(x)
        elif self.upscale == 1 and self.downscale > 1:
            # Flow matching with PixelShuffle to restore resolution
            x = self.conv_last(x)            # [B, 3*ds², H/ds, W/ds]
            x = self.pixel_shuffle(x)        # [B, 3, H, W]
        else:
            x = self.conv_last(x)

        x = x / self.img_range + self.mean

        # Unpadding
        x = x[..., :h_ori * self.upscale * self.downscale, :w_ori * self.upscale * self.downscale]

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


if __name__ == '__main__':
    upscale = 1  # Flow matching mode

    model = MPv2_arch(
        upscale=upscale,
        img_size=64,
        embed_dim=180,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        window_size=16,
        dim_ffn_td=16,
        category_size=256,
        num_tokens=512,
        reducted_dim=16,
        convffn_kernel_size=5,
        img_range=1.,
        mlp_ratio=2,
        upsampler='',  # No pixelshuffle for flow matching
        temb_ch=256,
        embedding_dim=256,
        downscale=2,  # PixelUnshuffle factor for JiT branch
    )

    # Model Size
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))

    # Test forward
    x = torch.randn([2, 6, 64, 64])  # x_t + lq_bicubic
    temp = torch.rand([2])
    output = model(x, None, temp)
    print(f"Input: {x.shape}, Output: {output.shape}")

    # Test without timestep (fallback)
    output_no_t = model(x[:, :3, :, :])
    print(f"No-t input: {x[:, :3, :, :].shape}, Output: {output_no_t.shape}")
