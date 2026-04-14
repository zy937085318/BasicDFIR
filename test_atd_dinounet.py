"""
Quick test script to verify ATD-integrated UNetDecoder works.
Run locally or on server.
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/Users/ybb/code zoo/BasicDFIR')

# Force reimport
import importlib
import basicsr.archs.dinounet_atd_arch
importlib.reload(basicsr.archs.dinounet_atd_arch)
from basicsr.archs.dinounet_atd_arch import ConvNeXtUNet, UNetDecoder, ATDDecoderBlock

def test_atd_decoder_block():
    """Test ATDDecoderBlock standalone."""
    print("=" * 60)
    print("Testing ATDDecoderBlock...")
    b, c, h, w = 2, 96, 32, 32
    x = torch.randn(b, c, h, w)
    x_size = (h, w)

    block = ATDDecoderBlock(
        dim=c,
        num_tokens=64,
        reducted_dim=10,
        num_heads=4,
        category_size=128,
        dim_ffn_td=16,
    )

    out = block(x, x_size)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print("  ATDDecoderBlock PASSED")

def test_unet_decoder_without_atd():
    """Test UNetDecoder without ATD (should work as before)."""
    print("=" * 60)
    print("Testing UNetDecoder (no ATD)...")
    encoder_dims = [96, 192, 384, 768]
    decoder = UNetDecoder(encoder_dims=encoder_dims, num_res_blocks=2, use_atd=False)

    # Simulate encoder features (4 stages)
    feats = [torch.randn(2, d, 32 // (2**i), 32 // (2**i)) for i, d in enumerate(encoder_dims)]
    temb = torch.randn(2, encoder_dims[-1] * 4)

    out = decoder(feats, temb)
    print(f"  Output: {out.shape}")
    print("  UNetDecoder (no ATD) PASSED")

def test_unet_decoder_with_atd():
    """Test UNetDecoder with ATD."""
    print("=" * 60)
    print("Testing UNetDecoder (with ATD)...")
    encoder_dims = [96, 192, 384, 768]
    decoder = UNetDecoder(
        encoder_dims=encoder_dims,
        num_res_blocks=2,
        use_atd=True,
        atd_num_tokens=64,
        atd_reducted_dim=10,
        atd_num_heads=4,
        atd_category_size=128,
        atd_dim_ffn_td=16,
    )

    # Simulate encoder features (4 stages)
    feats = [torch.randn(2, d, 32 // (2**i), 32 // (2**i)) for i, d in enumerate(encoder_dims)]
    temb = torch.randn(2, encoder_dims[-1] * 4)

    out = decoder(feats, temb)
    print(f"  Output: {out.shape}")
    print("  UNetDecoder (with ATD) PASSED")

def test_convnext_unet_with_atd():
    """Test full ConvNeXtUNet with ATD."""
    print("=" * 60)
    print("Testing ConvNeXtUNet (with ATD)...")
    model = ConvNeXtUNet(
        input_channels=6,
        encoder_type='tiny',
        use_atd=True,
        atd_num_tokens=64,
        atd_reducted_dim=10,
        atd_num_heads=4,
        atd_category_size=128,
        atd_dim_ffn_td=16,
    )

    x = torch.randn(2, 6, 64, 64)
    temp = torch.tensor([0.5, 0.5])

    out = model(x, temp=temp)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    assert out.shape[2:] == x.shape[2:], f"Output spatial size mismatch"
    print("  ConvNeXtUNet (with ATD) PASSED")

def test_convnext_unet_without_atd():
    """Test full ConvNeXtUNet without ATD (baseline)."""
    print("=" * 60)
    print("Testing ConvNeXtUNet (no ATD, baseline)...")
    model = ConvNeXtUNet(
        input_channels=6,
        encoder_type='tiny',
        use_atd=False,
    )

    x = torch.randn(2, 6, 64, 64)
    temp = torch.tensor([0.5, 0.5])

    out = model(x, temp=temp)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    assert out.shape[2:] == x.shape[2:], f"Output spatial size mismatch"
    print("  ConvNeXtUNet (no ATD) PASSED")

if __name__ == '__main__':
    torch.manual_seed(42)

    test_atd_decoder_block()
    test_unet_decoder_without_atd()
    test_unet_decoder_with_atd()
    test_convnext_unet_without_atd()
    test_convnext_unet_with_atd()

    print("=" * 60)
    print("ALL TESTS PASSED!")
