"""Test Dino_Encoder freeze functionality."""
import torch
import sys
sys.path.insert(0, '/home/ybb/Project/BasicDFIR')

from basicsr.archs.MyProposal.MPv2_2_arch import MPv2_2_arch

model = MPv2_2_arch(
        upscale=1, img_size=256, embed_dim=90,
        depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6], window_size=8,
        dim_ffn_td=16, category_size=128, num_tokens=64, reducted_dim=4,
        convffn_kernel_size=5, img_range=1., mlp_ratio=2, upsampler='',
        temb_ch=256, embedding_dim=256, downscale=4,
    )
total = sum(p.numel() for p in model.parameters())
print(f"Params: {total / 1e6:.3f}M")

x = torch.randn(2, 3, 256, 256)
c = torch.randn(2, 3, 64, 64)
temp = torch.rand(2)
out = model(x, c, temp)
print(f"Input: {x.shape} -> Output: {out.shape}")

# Test without timestep
out_no_t = model(x, None, None)
print(f"No-t Input: {x[:, :3].shape} -> Output: {out_no_t.shape}")