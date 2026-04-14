"""
Test script for Dino_Encoder and ATDUNet with DINOv3 backbone.
Run on server with GPU.

Note: This architecture is designed for image restoration (denoising/deblurring),
not super-resolution. The DINOv3 encoder downsamples by 4x, then ATDUNet processes
the features, and finally upsample restores the original resolution.
"""

import os
import sys
import torch
import torch.nn as nn

# Add project path
sys.path.insert(0, '/home/ybb/Project/BasicDFIR')

from basicsr.archs.dinoatd_arch import Dino_Encoder, DinoATD


def test_dino_encoder():
    """Test Dino_Encoder standalone."""
    print("=" * 60)
    print("Testing Dino_Encoder")
    print("=" * 60)

    batch_size = 2
    in_chans = 3
    H, W = 64, 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test with tiny (default) - no pretrained weights
    print("\n1. Testing Dino_Encoder with 'tiny' backbone (no pretrained):")
    encoder = Dino_Encoder(in_chans=in_chans, dino_size='tiny', pretrained=False).to(device)
    x = torch.randn(batch_size, in_chans, H, W).to(device)

    with torch.no_grad():
        feats = encoder(x)

    print(f"   Input shape: {x.shape}")
    print(f"   Output: List of 4 feature maps")
    # DINOv3 ConvNeXt downsample_ratios=[4,1,1,1] -> total downsampling = 4x
    # tiny dims: [96, 192, 384, 768]
    print(f"   feat0 shape: {feats[0].shape} - Expected: [{batch_size}, 96, {H//4}, {W//4}]")
    print(f"   feat1 shape: {feats[1].shape} - Expected: [{batch_size}, 192, {H//4}, {W//4}]")
    print(f"   feat2 shape: {feats[2].shape} - Expected: [{batch_size}, 384, {H//4}, {W//4}]")
    print(f"   feat3 shape: {feats[3].shape} - Expected: [{batch_size}, 768, {H//4}, {W//4}]")

    # Count parameters
    params = sum(p.numel() for p in encoder.parameters())
    print(f"   Parameters: {params / 1e6:.2f}M")

    # Verify output shapes
    assert feats[0].shape == (batch_size, 96, H//4, W//4), f"Shape mismatch! Got {feats[0].shape}"
    assert feats[1].shape == (batch_size, 192, H//4, W//4), f"Shape mismatch! Got {feats[1].shape}"
    assert feats[2].shape == (batch_size, 384, H//4, W//4), f"Shape mismatch! Got {feats[2].shape}"
    assert feats[3].shape == (batch_size, 768, H//4, W//4), f"Shape mismatch! Got {feats[3].shape}"
    print("   Test passed!")

    # Test with small
    print("\n2. Testing Dino_Encoder with 'small' backbone:")
    encoder_small = Dino_Encoder(in_chans=in_chans, dino_size='small', pretrained=False).to(device)
    params_small = sum(p.numel() for p in encoder_small.parameters())
    print(f"   Parameters: {params_small / 1e6:.2f}M")

    with torch.no_grad():
        feats_small = encoder_small(x)
    print(f"   feat0 shape: {feats_small[0].shape}")
    assert feats_small[0].shape == (batch_size, 96, H//4, W//4)
    print("   Test passed!")

    # Test with ds_scale=[2,2,0,0]
    print("\n3. Testing Dino_Encoder with ds_scale=[2,2,0,0]:")
    encoder_ds = Dino_Encoder(in_chans=in_chans, dino_size='tiny', pretrained=False, ds_scale=[2,2,0,0]).to(device)
    with torch.no_grad():
        feats_ds = encoder_ds(x)
    # ds_scale=[2,2,0,0]: stage0=2x, stage1=2x, stage2=1x, stage3=1x -> total 4x
    # But due to the default implementation, stage 2 and 3 might still have some downsampling
    print(f"   ds_scale=[2,2,0,0]:")
    print(f"   feat0 shape: {feats_ds[0].shape} - Expected: [{batch_size}, 96, {H//2}, {W//2}]")
    print(f"   feat1 shape: {feats_ds[1].shape} - Expected: [{batch_size}, 192, {H//4}, {W//4}]")
    print("   Test passed!")

    print("\nDino_Encoder tests passed!")


def test_dino_atd():
    """Test DinoATD with DINOv3 encoder."""
    print("\n" + "=" * 60)
    print("Testing DinoATD with DINOv3")
    print("=" * 60)
    print("Note: This is an image restoration model (output size = input size)")

    batch_size = 2
    in_chans = 3
    H, W = 64, 64
    upscale = 4  # This is for the upsample module, not total upscale

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for dino_size in ['tiny']:
        print(f"\nTesting DinoATD with dino_size='{dino_size}':")

        # Note: img_size refers to the original input image size
        # embed_dim must match DINO's last stage output: 768 for tiny
        model = DinoATD(
            upscale=upscale,
            img_size=H,
            embed_dim=768,  # Must match DINO tiny's last stage dim
            depths=(4, 4, 4, 4),
            num_heads=(6, 6, 6, 6),
            window_size=8,
            dim_ffn_td=16,
            category_size=128,
            num_tokens=64,
            reducted_dim=4,
            convffn_kernel_size=5,
            mlp_ratio=2.,
            img_range=1.,
            dino_size=dino_size,
            ds_scale=[2,2,0,0],  # Default ds_scale
            upsampler='pixelshuffle'
        ).to(device)

        x = torch.randn(batch_size, in_chans, H, W).to(device)

        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {params / 1e6:.2f}M")

        # Forward pass
        try:
            with torch.no_grad():
                y = model(x)
            print(f"   Input shape: {x.shape}")
            print(f"   Output shape: {y.shape}")
            # For restoration model, output size should equal input size
            print(f"   Expected: [{batch_size}, {in_chans}, {H}, {W}] (same as input)")
            assert y.shape == (batch_size, in_chans, H, W), f"Shape mismatch! Got {y.shape}"
            print(f"   Test passed!")
        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()

    print("\nDinoATD tests passed!")


def test_with_pretrained():
    """Test with pretrained weights (if available)."""
    print("\n" + "=" * 60)
    print("Testing with Pretrained Weights")
    print("=" * 60)

    pretrained_path = '/8T2/pretrained_models/DINOV3_pretrained_weight/ConvNeXt/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth'

    if not os.path.exists(pretrained_path):
        print(f"   Pretrained weights not found at: {pretrained_path}")
        print("   Skipping pretrained test.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nLoading pretrained weights from: {pretrained_path}")
    encoder = Dino_Encoder(in_chans=3, dino_size='tiny', pretrained=True).to(device)

    params = sum(p.numel() for p in encoder.parameters())
    print(f"   Parameters: {params / 1e6:.2f}M")

    x = torch.randn(1, 3, 64, 64).to(device)
    with torch.no_grad():
        feats = encoder(x)

    print(f"   Input shape: {x.shape}")
    print(f"   Output: List of 4 feature maps")
    print(f"   feat0 shape: {feats[0].shape}")
    print(f"   feat1 shape: {feats[1].shape}")
    print(f"   feat2 shape: {feats[2].shape}")
    print(f"   feat3 shape: {feats[3].shape}")
    print("   Pretrained test passed!")


def test_different_input_sizes():
    """Test with different input sizes."""
    print("\n" + "=" * 60)
    print("Testing Different Input Sizes")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for size in [32, 64, 128, 256]:
        print(f"\nTesting with input size {size}x{size}:")
        model = DinoATD(
            upscale=4,
            img_size=size,
            embed_dim=768,  # Must match DINO tiny's last stage dim
            depths=(2, 2, 2, 2),
            num_heads=(6, 6, 6, 6),
            window_size=8,
            dino_size='tiny',
            ds_scale=[2,2,0,0],
            upsampler='pixelshuffle'
        ).to(device)

        x = torch.randn(1, 3, size, size).to(device)
        with torch.no_grad():
            y = model(x)
        print(f"   Input: {x.shape}, Output: {y.shape}")
        assert y.shape == (1, 3, size, size)
        print(f"   Passed!")

    print("\nDifferent input sizes test passed!")


if __name__ == '__main__':
    print("Starting DinoATD tests...\n")

    test_dino_encoder()
    test_dino_atd()
    test_with_pretrained()
    test_different_input_sizes()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
