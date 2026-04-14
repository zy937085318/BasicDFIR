"""Standalone test for ConvNeXtUNet without triggering full package imports."""
import sys
import importlib.util

# Load ConvNeXt first
convnext_spec = importlib.util.spec_from_file_location('convnext', 'basicsr/archs/dinov3/convnext.py')
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

# Now define ConvNeXtEncoder, UNetDecoder, ConvNeXtUNet from the arch file
ConvNeXt = convnext_mod.ConvNeXt

class ConvNeXtEncoder(nn.Module):
    def __init__(self, in_chans=3, pretrained=None, depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768], drop_path_rate=0.0,
                 layer_scale_init_value=1e-6, freeze_backbone=False, use_multiscale=True):
        super().__init__()
        self.dims = dims
        self.depths = depths
        self.num_resolutions = 4

        self.backbone = ConvNeXt(
            in_chans=in_chans, depths=depths, dims=dims,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            scale_factor=4,
        )

        if pretrained is not None:
            state_dict = torch.load(pretrained, map_location='cpu')
            model_dict = self.backbone.state_dict()
            filtered = {k: v for k, v in state_dict.items() if k in model_dict}
            self.backbone.load_state_dict(filtered, strict=False)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

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


class UNetDecoder(nn.Module):
    def __init__(self, encoder_dims=[96, 192, 384, 768], num_res_blocks=2,
                 dropout=0.0, resamp_with_conv=True, act=Swish(), normalize=group_norm):
        super().__init__()
        self.encoder_dims = encoder_dims
        self.num_resolutions = len(encoder_dims)
        self.num_res_blocks = num_res_blocks

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
                block_in_ch = in_ch if (i_block == 0 and i_level == 0) else \
                              (in_ch + skip_ch if (i_block == 0 and i_level > 0) else out_ch)
                block_modules[f'{i_level}a_{i_block}a_block'] = ResidualBlock(
                    block_in_ch, temb_ch, out_ch, dropout, act, normalize)

            if i_level < self.num_resolutions - 1:
                block_modules[f'{i_level}b_upsample'] = upsample(out_ch, resamp_with_conv)

            up_modules.append(nn.ModuleDict(block_modules))
            in_ch = out_ch

        self.up_modules = nn.ModuleList(up_modules)
        self.end_conv = nn.Sequential(
            normalize(encoder_dims[0]), act, conv2d(encoder_dims[0], encoder_dims[0], init_scale=0.))

    def forward(self, encoder_features, temb):
        enc_feats = list(reversed(encoder_features))
        h = enc_feats[0]

        for i_level in range(self.num_resolutions):
            block_modules = self.up_modules[i_level]
            n_blocks = self.num_res_blocks + 1 if i_level > 0 else self.num_res_blocks

            if i_level == 0:
                h = block_modules['bottleneck_proj'](h)

            for i_block in range(n_blocks):
                h = block_modules[f'{i_level}a_{i_block}a_block'](h, temb)

            if i_level < self.num_resolutions - 1:
                h = self.up_modules[i_level][f'{i_level}b_upsample'](h)
                skip_feat = enc_feats[i_level + 1]
                if h.shape[2:] != skip_feat.shape[2:]:
                    h = torch.nn.functional.interpolate(h, size=skip_feat.shape[2:], mode='bilinear', align_corners=False)
                h = torch.cat([h, skip_feat], dim=1)

        return self.end_conv(h)


class ConvNeXtUNet(nn.Module):
    def __init__(self, input_channels=6, img_size=256, output_channels=3, pretrained=None,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.0,
                 num_res_blocks=2, dropout=0.0, resamp_with_conv=True,
                 act=Swish(), normalize=group_norm, freeze_backbone=False):
        super().__init__()
        self.input_channels = input_channels
        self.img_size = img_size
        self.output_channels = output_channels
        self.dims = dims
        self.num_res_blocks = num_res_blocks
        self.act = act
        self.normalize = normalize

        self.encoder = ConvNeXtEncoder(
            in_chans=input_channels, pretrained=pretrained, depths=depths,
            dims=dims, drop_path_rate=drop_path_rate, freeze_backbone=freeze_backbone)

        self.decoder = UNetDecoder(
            encoder_dims=dims, num_res_blocks=num_res_blocks,
            dropout=dropout, resamp_with_conv=resamp_with_conv, act=act, normalize=normalize)

        if hasattr(self.encoder, 'use_multiscale') and self.encoder.use_multiscale:
            self.output_proj = nn.Sequential(
                normalize(dims[0]), act,
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
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
print('ConvNeXtUNet Standalone Test')
print('=' * 60)

# Test 1: Encoder multi-scale
print('\n[1] ConvNeXtEncoder multi-scale features')
enc = ConvNeXtEncoder(in_chans=6, pretrained=None, use_multiscale=True)
x = torch.randn(2, 6, 256, 256)
feats = enc(x)
print('  Features from 256x256 input:')
for i, f in enumerate(feats):
    print(f'    Stage {i}: {f.shape}  (ch={enc.dims[i]})')
# Expected: [2, 96, 64, 64], [2, 192, 32, 32], [2, 384, 16, 16], [2, 768, 8, 8]

# Test 2: Decoder channel matching
print('\n[2] UNetDecoder')
dec = UNetDecoder(encoder_dims=[96, 192, 384, 768], num_res_blocks=2)
t = torch.tensor([0.5, 0.5])
temb = dec.temb_net(t)
dec_out = dec(feats, temb)
print(f'  Input: stage0={feats[0].shape}, ..., stage3={feats[3].shape}')
print(f'  Output: {dec_out.shape}')
assert dec_out.shape == (2, 96, 64, 64), f'Decoder output wrong: {dec_out.shape}'
print('  Decoder PASSED!')

# Test 3: Full model
print('\n[3] ConvNeXtUNet (full model)')
model = ConvNeXtUNet(input_channels=6, img_size=256, pretrained=None, freeze_backbone=False)
out = model(x, None, t)
print(f'  Input:  {x.shape}')
print(f'  Output: {out.shape}')
assert out.shape == (2, 3, 256, 256), f'Output wrong: {out.shape}'
print('  Full model PASSED!')

# Test 4: Different sizes
print('\n[4] Different model sizes')
for size, d in [('tiny', [96,192,384,768]), ('base', [128,256,512,1024])]:
    m = ConvNeXtUNet(input_channels=6, pretrained=None, dims=d)
    o = m(x, None, t)
    params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f'  {size} ({d}): output={o.shape}, params={params:.1f}M')
    assert o.shape == (2, 3, 256, 256)

# Test 5: Gradient flow
print('\n[5] Gradient flow (backward pass)')
model = ConvNeXtUNet(input_channels=6, pretrained=None, freeze_backbone=False)
out = model(x, None, t)
loss = out.sum()
loss.backward()
has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
print(f'  Has gradients: {has_grad}')
enc_grad = any(p.grad is not None for p in model.encoder.parameters() if p.requires_grad)
dec_grad = any(p.grad is not None for p in model.decoder.parameters() if p.requires_grad)
print(f'  Encoder has grad: {enc_grad}')
print(f'  Decoder has grad: {dec_grad}')
assert has_grad, 'No gradients!'
print('  Gradient flow PASSED!')

# Test 6: Freeze backbone
print('\n[6] Freeze backbone')
model_frz = ConvNeXtUNet(input_channels=6, pretrained=None, freeze_backbone=True)
out_frz = model_frz(x, None, t)
loss_frz = out_frz.sum()
loss_frz.backward()
# Backbone params should be frozen (no grad)
backbone_frozen = all(p.grad is None for p in model_frz.encoder.backbone.parameters())
print(f'  Backbone frozen (no grad): {backbone_frozen}')
assert backbone_frozen, 'Backbone should be frozen!'
# down_proj (newly added) should have grads
down_proj_trainable = any(p.grad is not None for p in model_frz.encoder.down_proj.parameters())
print(f'  down_proj trainable: {down_proj_trainable}')
assert down_proj_trainable, 'down_proj should be trainable!'
# Decoder should be trainable
dec_trainable = any(p.grad is not None for p in model_frz.decoder.parameters() if p.requires_grad)
print(f'  Decoder trainable: {dec_trainable}')
assert dec_trainable, 'Decoder should be trainable!'
print('  Freeze backbone PASSED!')

print('\n' + '=' * 60)
print('ALL TESTS PASSED!')
print('=' * 60)
