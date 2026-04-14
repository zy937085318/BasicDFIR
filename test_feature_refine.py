"""
Test script for modified FeatureRefine module.
Verifies that FeatureRefine only processes the stage1 feature.
"""

import torch
import sys
sys.path.insert(0, '/home/ybb/Project/BasicDFIR')

from basicsr.archs.dinounet_atd_arch import Dino_ATD_UNet

def test_feature_refine():
    print("=" * 60)
    print("Testing FeatureRefine: Stage1-only refinement")
    print("=" * 60)

    # Model config
    config = {
        'input_channels': 6,
        'img_size': 256,
        'output_channels': 3,
        'encoder_type': 'tiny',
        'num_res_blocks': 1,  # Reduced for testing
        'use_atd': False,  # Disable decoder ATD for cleaner test
        'use_feature_refine': True,  # Enable FeatureRefine
        'fr_depths': [2, 2, 2, 2],  # Reduced depth for testing
        'fr_num_heads': [4, 4, 4, 4],
        'freeze_backbone': True,
    }

    print("\n1. Creating model with FeatureRefine enabled...")
    model = Dino_ATD_UNet(**config)

    # Count parameters in FeatureRefine
    fr_params = sum(p.numel() for p in model.feature_refine.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   FeatureRefine params: {fr_params:,}")
    print(f"   Total model params: {total_params:,}")
    print(f"   FR ratio: {fr_params/total_params*100:.2f}%")

    # Test input
    B, H, W = 2, 256, 256
    x = torch.randn(B, 6, H, W)
    temp = torch.ones(B)

    print("\n2. Testing forward pass...")
    print(f"   Input shape: {x.shape}")

    with torch.no_grad():
        output = model(x, temp=temp)

    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Verify FeatureRefine behavior
    print("\n3. Verifying FeatureRefine internal behavior...")

    # Get encoder output
    encoder_features = model.encoder(x)
    print(f"   Encoder outputs:")
    for i, feat in enumerate(encoder_features):
        print(f"     stage{i}: {feat.shape}")

    # Get FeatureRefine output
    refined_features = model.feature_refine(encoder_features)
    print(f"   FeatureRefine outputs:")
    for i, feat in enumerate(refined_features):
        orig = encoder_features[i]
        is_same = torch.allclose(orig, feat, atol=1e-6)
        print(f"     stage{i}: {feat.shape} - {'PASSTHROUGH' if is_same else 'REFINED'}")

    # Verify stages 0,2,3 are unchanged, stage1 is refined
    for i in [0, 2, 3]:
        assert torch.allclose(encoder_features[i], refined_features[i], atol=1e-6), \
            f"stage{i} should be unchanged!"
    assert not torch.allclose(encoder_features[1], refined_features[1], atol=1e-2), \
        "stage1 should be refined!"

    print("\n4. All tests passed! ✓")
    print("   - Stages 0,2,3: PASSTHROUGH (unchanged)")
    print("   - Stage 1: REFINED (modified by ATDB)")

    return True

if __name__ == '__main__':
    try:
        test_feature_refine()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
