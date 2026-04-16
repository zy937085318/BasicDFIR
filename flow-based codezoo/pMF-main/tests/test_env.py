
import pytest
import torch
import torchvision
import timm
import lpips
import accelerate
import numpy as np
import pmf

def test_imports():
    """
    Verify that all required dependencies are installed and importable.
    """
    assert torch.__version__ is not None
    assert torchvision.__version__ is not None
    assert timm.__version__ is not None
    assert lpips is not None
    assert accelerate.__version__ is not None
    assert np.__version__ is not None
    # scipy/sentencepiece removed from explicit deps, skipping check
    # assert scipy.__version__ is not None
    # assert sentencepiece.__version__ is not None

def test_cuda_availability():
    """
    Check CUDA availability. This test is skipped if no CUDA device is present,
    but logs information.
    """
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        x = torch.tensor([1.0]).cuda()
        assert x.device.type == 'cuda'
    else:
        pytest.skip("CUDA not available")

def test_tensor_ops():
    """
    Basic sanity check for tensor operations.
    """
    x = torch.ones(2, 2)
    y = x + x
    assert torch.allclose(y, torch.tensor([[2.0, 2.0], [2.0, 2.0]]))
