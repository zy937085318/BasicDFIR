"""
Test script for Decoder architecture on server with CUDA (GPU5).
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import torch
import sys
sys.path.insert(0, '/home/ybb/Project/BasicDFIR')


def test_decoder_interface():
    B, H = 2, 256
    encoder_features = [
        torch.randn(B, 96, H//4, H//4, device='cuda'),
        torch.randn(B, 192, H//4, H//4, device='cuda'),
        torch.randn(B, 384, H//4, H//4, device='cuda'),
        torch.randn(B, 768, H//4, H//4, device='cuda'),
    ]
    print("Mock ConvNeXt intermediate layers (CUDA):")
    for i, feat in enumerate(encoder_features):
        print(f"  Stage {i}: {feat.shape}")
    return encoder_features


def test_decoder_with_mock(encoder_features):
    from basicsr.archs.decoder_arch import UNetDecoder

    decoder = UNetDecoder(
        encoder_dims=[96, 192, 384, 768],
        img_size=256,
        ch=32,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
    ).cuda()
    decoder.eval()

    with torch.no_grad():
        outputs = decoder(encoder_features)

    print(f"\nDecoder output shapes:")
    for i, out in enumerate(outputs):
        print(f"  x[{i}]: {out.shape}")

    assert isinstance(outputs, list) and len(outputs) == 4
    expected_channels = [3, 64, 128, 256]
    for i, (out, exp_ch) in enumerate(zip(outputs, expected_channels)):
        assert out.shape[1] == exp_ch, f"x[{i}]: expected {exp_ch} channels, got {out.shape[1]}"
    print("\nDecoder mock test PASSED!")


def test_decoder_output_resolutions(encoder_features):
    from basicsr.archs.decoder_arch import UNetDecoder

    B, H = 2, 256
    decoder = UNetDecoder(
        encoder_dims=[96, 192, 384, 768],
        img_size=H,
        ch=32,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
    ).cuda()
    decoder.eval()

    with torch.no_grad():
        outputs = decoder(encoder_features)

    expected_resolutions = [
        (H, H),            # x[0]
        (H // 2, H // 2),  # x[1]
        (H // 4, H // 4),  # x[2]
        (H // 8, H // 8),  # x[3]
    ]

    print(f"\nDecoder output resolutions:")
    for i, (out, (exp_h, exp_w)) in enumerate(zip(outputs, expected_resolutions)):
        actual_h, actual_w = out.shape[2], out.shape[3]
        print(f"  x[{i}]: {out.shape[2:]} (expected {exp_h}x{exp_w})")
        assert actual_h == exp_h and actual_w == exp_w
    print("\nDecoder resolution test PASSED!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Decoder interface expectations...")
    print("=" * 60)
    encoder_features = test_decoder_interface()

    print("\n" + "=" * 60)
    print("Testing Decoder with mock implementation (CUDA)...")
    print("=" * 60)
    test_decoder_with_mock(encoder_features)

    print("\n" + "=" * 60)
    print("Testing Decoder output resolutions...")
    print("=" * 60)
    test_decoder_output_resolutions(encoder_features)

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
