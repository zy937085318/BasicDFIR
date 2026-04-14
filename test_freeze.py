"""Test Dino_Encoder freeze functionality."""
import torch
import sys
sys.path.insert(0, '/home/ybb/Project/BasicDFIR')

from basicsr.archs.dinoatd_arch import Dino_Encoder

def count_trainable_params(model):
    """Count trainable and frozen parameters."""
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()
    return trainable, frozen

def test_freeze():
    print("=" * 60)
    print("Testing Dino_Encoder Freeze Functionality")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Test 1: freeze=False (default)
    print("-" * 60)
    print("Test 1: freeze=False (all parameters trainable)")
    print("-" * 60)
    encoder_unfrozen = Dino_Encoder(
        in_chans=3,
        dino_size='tiny',
        pretrained=False,
        ds_scale=[2, 2, 0, 0],
        freeze=False
    ).to(device)

    trainable_unfrozen, frozen_unfrozen = count_trainable_params(encoder_unfrozen)
    total_unfrozen = trainable_unfrozen + frozen_unfrozen

    print(f"Total parameters: {total_unfrozen / 1e6:.2f}M")
    print(f"Trainable: {trainable_unfrozen / 1e6:.2f}M")
    print(f"Frozen: {frozen_unfrozen / 1e6:.2f}M")

    # Check specific layers
    print("\nLayer requires_grad status:")
    for i, stage in enumerate(encoder_unfrozen.dino.stages):
        all_trainable = all(p.requires_grad for p in stage.parameters())
        print(f"  Stage {i}: requires_grad={all_trainable}")
    norm_trainable = all(p.requires_grad for p in encoder_unfrozen.dino.norm.parameters())
    print(f"  Norm: requires_grad={norm_trainable}")
    for i, downsample in enumerate(encoder_unfrozen.dino.downsample_layers):
        all_trainable = all(p.requires_grad for p in downsample.parameters())
        print(f"  Downsample {i}: requires_grad={all_trainable}")

    # Test 2: freeze=True
    print("\n" + "-" * 60)
    print("Test 2: freeze=True (stages and norm frozen)")
    print("-" * 60)
    encoder_frozen = Dino_Encoder(
        in_chans=3,
        dino_size='tiny',
        pretrained=False,
        ds_scale=[2, 2, 0, 0],
        freeze=True
    ).to(device)

    trainable_frozen, frozen_frozen = count_trainable_params(encoder_frozen)
    total_frozen = trainable_frozen + frozen_frozen

    print(f"Total parameters: {total_frozen / 1e6:.2f}M")
    print(f"Trainable: {trainable_frozen / 1e6:.2f}M")
    print(f"Frozen: {frozen_frozen / 1e6:.2f}M")

    # Check specific layers
    print("\nLayer requires_grad status:")
    for i, stage in enumerate(encoder_frozen.dino.stages):
        all_trainable = all(p.requires_grad for p in stage.parameters())
        all_frozen = all(not p.requires_grad for p in stage.parameters())
        print(f"  Stage {i}: requires_grad={all_trainable}, frozen={all_frozen}")
    norm_frozen = all(not p.requires_grad for p in encoder_frozen.dino.norm.parameters())
    print(f"  Norm: frozen={norm_frozen}")
    for i, downsample in enumerate(encoder_frozen.dino.downsample_layers):
        all_trainable = all(p.requires_grad for p in downsample.parameters())
        print(f"  Downsample {i}: requires_grad={all_trainable}")

    # Verify frozen parameters match
    assert frozen_unfrozen == 0, "Unfrozen model should have 0 frozen parameters"
    assert frozen_frozen > 0, "Frozen model should have frozen parameters"
    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)

if __name__ == '__main__':
    test_freeze()
