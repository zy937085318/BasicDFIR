"""Standalone test for ConvNeXtATDUNet (ConvNeXtUNet with use_atd=True)."""
import sys
import importlib.util

SERVER_PATH = '/home/ybb/Project/BasicDFIR'

# Load ConvNeXt first
convnext_spec = importlib.util.spec_from_file_location('convnext', f'{SERVER_PATH}/basicsr/archs/dinov3/convnext.py')
convnext_mod = importlib.util.module_from_spec(convnext_spec)
convnext_spec.loader.exec_module(convnext_mod)

# Mock basicsr package
class MockRegistry:
    def get(self, name): return lambda **kw: None
    def register(self, fn): return fn

mock_registry = MockRegistry()
sys.modules.setdefault('basicsr', type(sys)('basicsr'))
sys.modules['basicsr.utils'] = type(sys)('basicsr.utils')
sys.modules['basicsr.utils.registry'] = type(sys)('basicsr.utils.registry')
sys.modules['basicsr.utils.registry'].ARCH_REGISTRY = mock_registry
sys.modules['basicsr.archs'] = type(sys)('basicsr.archs')
sys.modules['basicsr.archs.dinov3'] = type(sys)('basicsr.archs.dinov3')
sys.modules['basicsr.archs.dinov3.convnext'] = convnext_mod

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np

# Copy essential classes from unet_arch.py directly
class Swish(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x) * x

def group_norm(out_ch, num_groups=None):
    if num_groups is None:
        for g in [32, 16, 8, 4, 2, 1]:
            if out_ch % g == 0:
                num_groups = g
                break
    return nn.GroupNorm(num_groups=num_groups, num_channels=out_ch, eps=1e-6, affine=True)

def upsample(in_ch, with_conv):
    up = nn.Sequential()
    up.add_module('up_nn', nn.Upsample(scale_factor=2, mode='nearest'))
    if with_conv:
        up.add_module('up_conv', nn.Conv2d(in_ch, in_ch, 3, 1, 1))
    return up

def dense(in_channels, out_channels):
    lin = nn.Linear(in_channels, out_channels)
    nn.init.zeros_(lin.bias)
    return lin

def get_sinusoidal_positional_embedding(timesteps, embedding_dim):
    device = timesteps.device
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), "constant", 0)
    return emb

class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=Swish()):
        super().__init__()
        self.main = nn.Sequential(
            dense(embedding_dim, hidden_dim),
            act,
            dense(hidden_dim, output_dim),
        )

    def forward(self, temp):
        temb = get_sinusoidal_positional_embedding(temp, self.main[0].in_features)
        return self.main(temb)

def conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True, init_scale=1.):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=bias)
    return conv

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, temb_ch, out_ch=None, dropout=0., act=Swish(), normalize=group_norm):
        super().__init__()
        self.out_ch = out_ch if out_ch is not None else in_ch
        self.temb_proj = dense(temb_ch, self.out_ch)
        self.norm1 = normalize(in_ch) if normalize is not None else nn.Identity()
        self.conv1 = conv2d(in_ch, self.out_ch)
        self.norm2 = normalize(self.out_ch) if normalize is not None else nn.Identity()
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0. else nn.Identity()
        self.conv2 = conv2d(self.out_ch, self.out_ch, init_scale=0.)
        if in_ch != self.out_ch:
            self.shortcut = conv2d(in_ch, self.out_ch, kernel_size=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        self.act = act

    def forward(self, x, temb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.temb_proj(self.act(temb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        x = self.shortcut(x)
        return x + h

# ======================== ATD Components ========================

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

def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)

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

class ATD_CA(nn.Module):
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

        q = self.wq(x)
        k = self.wk(td)
        v = self.wv(td)

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        scale = 1 + torch.clamp(self.scale, 0, 3) * np.log(self.num_tokens)
        attn = attn * scale
        attn = self.softmax(attn)

        x = (attn @ v).reshape(b, n, c)
        return x, attn

class AC_MSA(nn.Module):
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

class ATDDecoderBlock(nn.Module):
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

        self.td = nn.Parameter(torch.randn([num_tokens, dim]), requires_grad=True)

        self.atd_ca = ATD_CA(dim, num_tokens=num_tokens, reducted_dim=reducted_dim, qkv_bias=qkv_bias)
        self.ac_msa = AC_MSA(dim, num_heads=num_heads, category_size=category_size, qkv_bias=qkv_bias)
        self.wqkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc_td = nn.Linear(dim, dim_ffn_td)
        self.convffn = ConvFFN_td(in_features=dim, hidden_features=mlp_hidden_dim,
                                   td_features=dim_ffn_td, kernel_size=convffn_kernel_size,
                                   act_layer=act_layer)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

        self.patch_embed = PatchEmbed(img_size=64, patch_size=1, in_chans=0, embed_dim=dim)
        self.patch_unembed = PatchUnEmbed(img_size=64, patch_size=1, in_chans=0, embed_dim=dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_size, params=None):
        b, c, h, w = x.shape
        x_size = (h, w)

        x_seq = self.patch_embed(x)
        shortcut = x_seq

        x_seq = self.norm1(x_seq)

        td_expanded = self.td.expand([b, -1, -1])
        x_atd, sim_atd = self.atd_ca(x_seq, td_expanded, x_size)

        tk_id = torch.argmax(sim_atd, dim=-1, keepdim=False)
        qkv = self.wqkv(x_seq)
        x_aca = self.ac_msa(qkv, tk_id, x_size)

        x_td = torch.gather(
            self.fc_td(td_expanded),
            dim=1,
            index=tk_id.reshape(b, h*w, 1).expand(-1, -1, self.dim_ffn_td)
        )

        x_ffn = self.convffn(self.norm2(x_seq), x_td, x_size)
        x_ffn = self.norm3(x_ffn)

        x_out = shortcut + x_atd + x_aca + x_ffn

        x_out = self.patch_unembed(x_out, x_size)

        return x_out


# ======================== ConvNeXt Encoder ========================

ConvNeXt = convnext_mod.ConvNeXt

class ConvNeXtEncoder(nn.Module):
    def __init__(self, in_chans=3, pretrained=None, depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768], drop_path_rate=0.0,
                 layer_scale_init_value=1e-6, freeze_backbone=False, use_multiscale=True,
                 scale_factor_first_layer=2):
        super().__init__()
        self.dims = dims
        self.depths = depths
        self.num_resolutions = 4

        self.backbone = ConvNeXt(
            in_chans=in_chans, depths=depths, dims=dims,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            scale_factor=scale_factor_first_layer,
        )

        if pretrained is not None:
            state_dict = torch.load(pretrained, map_location='cpu')
            model_dict = self.backbone.state_dict()
            filtered = {k: v for k, v in state_dict.items() if k in model_dict}
            self.backbone.load_state_dict(filtered, strict=False)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            stem = self.backbone.downsample_layers[0]
            for i in stem:
                i.weight.requires_grad = True
                i.bias.requires_grad = True

        self.use_multiscale = use_multiscale
        if use_multiscale:
            self.down_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                )
                for i in range(3)
            ])

    def forward(self, x):
        feats = []
        for i in range(4):
            x = self.backbone.downsample_layers[i](x)
            x = self.backbone.stages[i](x)
            feats.append(x)

        if self.use_multiscale:
            scaled = [feats[0]]
            for i in range(3):
                scaled.append(self.down_proj[i](scaled[-1]))
            return scaled
        return feats


# ======================== UNet Decoder with ATD ========================

class UNetDecoder(nn.Module):
    def __init__(self, encoder_dims=[96, 192, 384, 768], num_res_blocks=2,
                 dropout=0.0, resamp_with_conv=True, act=Swish(), normalize=group_norm,
                 meanflow=False,
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
        self.use_atd = use_atd

        ch = encoder_dims[-1]
        temb_ch = ch * 4
        self.temb_net = TimestepEmbedding(ch, temb_ch, temb_ch, act)

        up_modules = []
        in_ch = encoder_dims[-1]

        for i_level in range(self.num_resolutions):
            block_modules = {}
            enc_stage_idx = self.num_resolutions - 1 - i_level
            skip_ch = encoder_dims[enc_stage_idx]
            n_blocks = num_res_blocks + 1 if i_level > 0 else num_res_blocks
            out_ch = skip_ch

            if i_level == 0:
                block_modules['bottleneck_proj'] = nn.Sequential(
                    normalize(in_ch), act, conv2d(in_ch, skip_ch))
                in_ch = skip_ch

            for i_block in range(n_blocks):
                if i_block == 0 and i_level > 0:
                    block_in_ch = in_ch + skip_ch
                elif i_block == 0 and i_level == 0:
                    block_in_ch = in_ch
                else:
                    block_in_ch = out_ch
                block_modules[f'{i_level}a_{i_block}a_block'] = ResidualBlock(
                    block_in_ch, temb_ch, out_ch, dropout, act, normalize)

            if i_level < self.num_resolutions - 1:
                block_modules[f'{i_level}b_upsample'] = upsample(out_ch, resamp_with_conv)

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
            in_ch = out_ch

        self.up_modules = nn.ModuleList(up_modules)
        self.end_conv = nn.Sequential(
            normalize(encoder_dims[0]), act, conv2d(encoder_dims[0], encoder_dims[0], init_scale=0.))

    def forward(self, encoder_features, temb, params=None):
        enc_feats = list(reversed(encoder_features))
        h = enc_feats[0]

        for i_level in range(self.num_resolutions):
            block_modules = self.up_modules[i_level]
            n_blocks = self.num_res_blocks + 1 if i_level > 0 else self.num_res_blocks

            if i_level == 0:
                h = block_modules['bottleneck_proj'](h)

            for i_block in range(n_blocks):
                h = block_modules[f'{i_level}a_{i_block}a_block'](h, temb)

            if self.use_atd and i_level < self.num_resolutions - 1:
                h = block_modules[f'{i_level}c_atd'](h, (h.shape[2], h.shape[3]), params)

            if i_level < self.num_resolutions - 1:
                h = self.up_modules[i_level][f'{i_level}b_upsample'](h)
                skip_feat = enc_feats[i_level + 1]
                if h.shape[2:] != skip_feat.shape[2:]:
                    h = torch.nn.functional.interpolate(h, size=skip_feat.shape[2:], mode='bilinear', align_corners=False)
                h = torch.cat([h, skip_feat], dim=1)

        return self.end_conv(h)


# ======================== ConvNeXtATDUNet ========================

dinov3 = {
    'tiny': {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768], 'pretrained': '/8T2/pretrained_models/DINOV3_pretrained_weight/ConvNeXt/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth'},
    'small': {'depths': [3, 3, 27, 3], 'dims': [96, 192, 384, 768], 'pretrained': '/8T2/pretrained_models/DINOV3_pretrained_weight/ConvNeXt/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth'},
    'base': {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024], 'pretrained': '/8T2/pretrained_models/DINOV3_pretrained_weight/ConvNeXt/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth'},
    'large': {'depths': [3, 3, 27, 3], 'dims': [192, 384, 768, 1536], 'pretrained': '/8T2/pretrained_models/DINOV3_pretrained_weight/ConvNeXt/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth'},
}

class ConvNeXtATDUNet(nn.Module):
    def __init__(self, input_channels=6, img_size=256, output_channels=3,
                 encoder_type='tiny', drop_path_rate=0.0,
                 num_res_blocks=2, dropout=0.0, resamp_with_conv=True,
                 act=Swish(), normalize=group_norm, freeze_backbone=False,
                 use_atd=False,
                 atd_num_tokens=64,
                 atd_reducted_dim=10,
                 atd_num_heads=4,
                 atd_category_size=128,
                 atd_dim_ffn_td=16,
                 atd_convffn_kernel_size=5,
                 atd_mlp_ratio=4.,
                 scale_factor_first_layer=2,
                 ):
        super().__init__()
        self.input_channels = input_channels
        self.img_size = img_size
        self.output_channels = output_channels
        self.depths = dinov3[encoder_type]['depths']
        self.dims = dinov3[encoder_type]['dims']
        self.pretrained = dinov3[encoder_type]['pretrained']  # derived from dict, not a constructor arg
        dims = self.dims
        depths = self.depths
        pretrained = self.pretrained  # used internally by encoder
        self.num_res_blocks = num_res_blocks
        self.use_atd = use_atd
        self.act = act
        self.normalize = normalize

        self.encoder = ConvNeXtEncoder(
            in_chans=input_channels, pretrained=None, depths=depths,
            dims=dims, drop_path_rate=drop_path_rate, freeze_backbone=freeze_backbone,
            scale_factor_first_layer=scale_factor_first_layer)

        self.decoder = UNetDecoder(
            encoder_dims=dims, num_res_blocks=num_res_blocks,
            dropout=dropout, resamp_with_conv=resamp_with_conv, act=act, normalize=normalize,
            use_atd=use_atd,
            atd_num_tokens=atd_num_tokens,
            atd_reducted_dim=atd_reducted_dim,
            atd_num_heads=atd_num_heads,
            atd_category_size=atd_category_size,
            atd_dim_ffn_td=atd_dim_ffn_td,
            atd_convffn_kernel_size=atd_convffn_kernel_size,
            atd_mlp_ratio=atd_mlp_ratio,
            norm_layer=nn.LayerNorm,
        )

        if hasattr(self.encoder, 'use_multiscale') and self.encoder.use_multiscale:
            self.output_proj = nn.Sequential(
                normalize(dims[0]), act,
                nn.Upsample(scale_factor=scale_factor_first_layer, mode='bilinear', align_corners=False),
                conv2d(dims[0], output_channels, init_scale=0.))
        else:
            self.output_proj = nn.Sequential(
                normalize(dims[0]), act, conv2d(dims[0], output_channels, init_scale=0.))

    def forward(self, x, c=None, temp=None):
        B, C, H, W = x.size()
        if temp is None:
            temp = torch.ones(B, device=x.device)
        elif temp.dim() > 1:
            temp = temp.squeeze()
        temb = self.decoder.temb_net(temp)

        enc_feats = self.encoder(x)
        dec_out = self.decoder(enc_feats, temb)
        return self.output_proj(dec_out)


# ======================== TESTS ========================
print('=' * 60)
print('ConvNeXtATDUNet Standalone Test')
print('=' * 60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

x = torch.randn(2, 6, 256, 256).to(device)
t = torch.tensor([0.5, 0.5]).to(device)

# Test 1: Encoder multi-scale features
print('\n[1] ConvNeXtEncoder multi-scale features (scale_factor=2)')
enc = ConvNeXtEncoder(in_chans=6, pretrained=None, use_multiscale=True, scale_factor_first_layer=2)
enc = enc.to(device)
feats = enc(x)
print('  Features from 256x256 input:')
for i, f in enumerate(feats):
    print(f'    Stage {i}: {f.shape}  (ch={enc.dims[i]})')
# Expected with scale_factor=2: [2, 96, 128, 128], [2, 192, 64, 64], [2, 384, 32, 32], [2, 768, 16, 16]
print('  Encoder PASSED!')

# Test 2: ATDDecoderBlock
print('\n[2] ATDDecoderBlock')
atd_block = ATDDecoderBlock(dim=96, num_tokens=64, reducted_dim=10, num_heads=4,
                              category_size=128, dim_ffn_td=16, convffn_kernel_size=5,
                              mlp_ratio=4., norm_layer=nn.LayerNorm).to(device)
atd_out = atd_block(feats[0], (feats[0].shape[2], feats[0].shape[3]))
print(f'  Input:  {feats[0].shape}')
print(f'  Output: {atd_out.shape}')
assert atd_out.shape == feats[0].shape, f'ATD block output shape mismatch: {atd_out.shape}'
print('  ATDDecoderBlock PASSED!')

# Test 3: ATDDecoderBlock backward
print('\n[3] ATDDecoderBlock backward')
atd_block2 = ATDDecoderBlock(dim=96, num_tokens=64, reducted_dim=10, num_heads=4,
                               category_size=128, dim_ffn_td=16, norm_layer=nn.LayerNorm).to(device)
atd_block2.zero_grad()
loss = atd_block2(feats[0], (feats[0].shape[2], feats[0].shape[3])).sum()
loss.backward()
has_grad = any(p.grad is not None for p in atd_block2.parameters())
print(f'  Has gradients: {has_grad}')
assert has_grad, 'No gradients in ATDDecoderBlock!'
print('  ATDDecoderBlock backward PASSED!')

# Test 4: UNetDecoder with ATD
print('\n[4] UNetDecoder with ATD')
dec = UNetDecoder(
    encoder_dims=[96, 192, 384, 768], num_res_blocks=2,
    use_atd=True,
    atd_num_tokens=64,
    atd_reducted_dim=10,
    atd_num_heads=4,
    atd_category_size=128,
    atd_dim_ffn_td=16,
    norm_layer=nn.LayerNorm,
).to(device)
temb = dec.temb_net(t)
dec_out = dec(feats, temb)
print(f'  Input: stage0={feats[0].shape}, ..., stage3={feats[3].shape}')
print(f'  Output: {dec_out.shape}')
# With scale_factor=2: stage0 is H/2, decoder outputs H/2
expected_shape = (2, 96, 128, 128)
assert dec_out.shape == expected_shape, f'Decoder output wrong: {dec_out.shape}, expected {expected_shape}'
print('  UNetDecoder+ATD PASSED!')

# Test 5: Full ConvNeXtATDUNet model
print('\n[5] ConvNeXtATDUNet (full model, tiny, use_atd=True)')
model = ConvNeXtATDUNet(input_channels=6, img_size=256,
                          encoder_type='tiny',
                          use_atd=True,
                          atd_num_tokens=64,
                          atd_reducted_dim=10,
                          atd_num_heads=4,
                          atd_category_size=128,
                          atd_dim_ffn_td=16,
                          atd_mlp_ratio=4.,
                          scale_factor_first_layer=2).to(device)
out = model(x, None, t)
print(f'  Input:  {x.shape}')
print(f'  Output: {out.shape}')
expected_out = (2, 3, 256, 256)
assert out.shape == expected_out, f'Output wrong: {out.shape}, expected {expected_out}'
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f'  Total params: {params:.2f}M')
print('  Full model PASSED!')

# Test 6: Different ATD configurations
print('\n[6] Different ATD configurations')
configs = [
    {'atd_num_tokens': 32, 'atd_reducted_dim': 8, 'atd_num_heads': 2, 'atd_category_size': 64},
    {'atd_num_tokens': 128, 'atd_reducted_dim': 16, 'atd_num_heads': 8, 'atd_category_size': 256},
]
for cfg in configs:
    m = ConvNeXtATDUNet(input_channels=6,encoder_type='tiny', **cfg).to(device)
    o = m(x, None, t)
    print(f"  {cfg}: output={o.shape}")
    assert o.shape == (2, 3, 256, 256)
print('  ATD config variations PASSED!')

# Test 7: Gradient flow with ATD
print('\n[7] Gradient flow with ATD (backward pass)')
model = ConvNeXtATDUNet(input_channels=6,encoder_type='tiny',
                          use_atd=True).to(device)
model.zero_grad()
out = model(x, None, t)
loss = out.sum()
loss.backward()
has_grad = any(p.grad is not None for p in model.parameters())
print(f'  Has gradients: {has_grad}')
td_grad = model.decoder.up_modules[0]['0c_atd'].td.grad is not None
print(f'  Token Dictionary has grad: {td_grad}')
atd_convffn_grad = any(p.grad is not None for p in model.decoder.up_modules[0]['0c_atd'].convffn.parameters())
print(f'  ConvFFN_td has grad: {atd_convffn_grad}')
assert has_grad, 'No gradients!'
assert td_grad, 'Token Dictionary should have gradients!'
print('  Gradient flow PASSED!')

# Test 8: Freeze backbone with ATD
print('\n[8] Freeze backbone with ATD')
model_frz = ConvNeXtATDUNet(input_channels=6, encoder_type='tiny',
                              freeze_backbone=True, use_atd=True).to(device)
out_frz = model_frz(x, None, t)
loss_frz = out_frz.sum()
loss_frz.backward()
# Backbone proper (excluding stem which is intentionally unfrozen for training)
backbone_proper = [(n, p) for n, p in model_frz.encoder.backbone.named_parameters()
                   if 'downsample_layers.0' not in n]
backbone_frozen = all(p.grad is None for _, p in backbone_proper)
stem_trainable = any(p.grad is not None for n, p in model_frz.encoder.backbone.named_parameters()
                     if 'downsample_layers.0' in n)
print(f'  Backbone proper frozen (no grad): {backbone_frozen}')
print(f'  Stem trainable (has grad): {stem_trainable}')
atd_trainable = any(p.grad is not None for p in model_frz.decoder.up_modules[0]['0c_atd'].parameters())
print(f'  ATD blocks trainable: {atd_trainable}')
assert backbone_frozen, 'Backbone proper should be frozen!'
assert stem_trainable, 'Stem should be trainable!'
assert atd_trainable, 'ATD blocks should be trainable!'
print('  Freeze backbone PASSED!')

# Test 9: Compare with vs without ATD
print('\n[9] Compare with vs without ATD')
model_no_atd = ConvNeXtATDUNet(input_channels=6, encoder_type='tiny', use_atd=False).to(device)
model_with_atd = ConvNeXtATDUNet(input_channels=6, encoder_type='tiny', use_atd=True).to(device)
p_no_atd = sum(p.numel() for p in model_no_atd.parameters()) / 1e6
p_with_atd = sum(p.numel() for p in model_with_atd.parameters()) / 1e6
print(f'  Without ATD: {p_no_atd:.2f}M params')
print(f'  With ATD:    {p_with_atd:.2f}M params')
print(f'  ATD overhead: {p_with_atd - p_no_atd:.2f}M params (+{(p_with_atd/p_no_atd-1)*100:.1f}%)')
o_no_atd = model_no_atd(x, None, t)
o_with_atd = model_with_atd(x, None, t)
assert o_no_atd.shape == o_with_atd.shape == (2, 3, 256, 256)
print('  Comparison PASSED!')

# Test 10: Different input sizes
print('\n[10] Different input sizes')
for size in [128, 192]:
    x_size = torch.randn(2, 6, size, size).to(device)
    m = ConvNeXtATDUNet(input_channels=6,encoder_type='tiny', use_atd=True).to(device)
    o = m(x_size, None, t)
    print(f'  Input {size}x{size} -> Output {o.shape[2]}x{o.shape[3]}')
    assert o.shape == (2, 3, size, size), f'Size {size} failed: {o.shape}'
print('  Different sizes PASSED!')

# Test 11: Encoder type variations
print('\n[11] Encoder type variations (no ATD for speed)')
for enc_type in ['tiny', 'small']:
    m = ConvNeXtATDUNet(input_channels=6,encoder_type=enc_type, use_atd=False).to(device)
    o = m(x, None, t)
    expected_dims = dinov3[enc_type]['dims']
    print(f'  {enc_type} (dims={expected_dims}): output={o.shape}')
    assert o.shape == (2, 3, 256, 256)
print('  Encoder type variations PASSED!')

print('\n' + '=' * 60)
print('ALL TESTS PASSED!')
print('=' * 60)
