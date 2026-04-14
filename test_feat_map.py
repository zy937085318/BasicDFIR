"""Test DinoATD feature mapping layer."""
import torch
import os

# Add project path
import sys
sys.path.insert(0, '/home/ybb/Project/BasicDFIR')

from basicsr.archs.dinoatd_arch import DinoATD

def test_feat_map():
    """Test feature mapping with different embed_dim values."""
    print("=" * 60)
    print("Testing DinoATD Feature Mapping Layer")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 2
    in_chans = 3
    H, W = 64, 64

    # DINO tiny dims: [96, 192, 384, 768]
    # Note: embed_dim must be divisible by num_heads
    test_cases = [
        {"embed_dim": 768, "num_heads": 6, "desc": "embed_dim=768 (matches last stage)"},
        {"embed_dim": 256, "num_heads": 4, "desc": "embed_dim=256 (smaller)"},
        {"embed_dim": 512, "num_heads": 8, "desc": "embed_dim=512 (mid size)"},
        {"embed_dim": 90,  "num_heads": 6, "desc": "embed_dim=90 (original ATD)"},
        {"embed_dim": 180, "num_heads": 6, "desc": "embed_dim=180 (2x original)"},
    ]

    for case in test_cases:
        embed_dim = case["embed_dim"]
        num_heads = case["num_heads"]
        desc = case["desc"]
        print(f"\n{'='*60}")
        print(f"Test: {desc}")
        print(f"{'='*60}")

        model = DinoATD(
            upscale=4,
            img_size=H,
            embed_dim=embed_dim,
            depths=(2, 2, 2, 2),
            num_heads=(num_heads, num_heads, num_heads, num_heads),
            window_size=8,
            dim_ffn_td=16,
            category_size=128,
            num_tokens=64,
            reducted_dim=4,
            convffn_kernel_size=5,
            mlp_ratio=2.,
            img_range=1.,
            dino_size='tiny',
            ds_scale=[2, 2, 0, 0],
            upsampler='pixelshuffle'
        ).to(device)

        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params / 1e6:.2f}M")

        # Check feat_map layers
        print(f"\nFeature mapping layers:")
        dino_dims = model.conv_first.dims
        for i, (dim, layer) in enumerate(zip(dino_dims, model.feat_map)):
            layer_type = type(layer).__name__
            if hasattr(layer, 'in_channels'):
                print(f"  Stage {i}: {dim} -> {layer.out_channels} ({layer_type})")
            else:
                print(f"  Stage {i}: {dim} -> {dim} (Identity)")

        # Forward pass
        x = torch.randn(batch_size, in_chans, H, W).to(device)
        try:
            with torch.no_grad():
                y = model(x)
            print(f"\nInput shape:  {x.shape}")
            print(f"Output shape: {y.shape}")
            assert y.shape == (batch_size, in_chans, H, W)
            print("Test PASSED!")
        except Exception as e:
            print(f"Test FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}")

if __name__ == '__main__':
    test_feat_map()
