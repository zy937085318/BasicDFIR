import pytest
import torch
import torch.nn as nn
from pmf.dit import DiT, DiTBlock, FinalLayer, TimestepEmbedder

@pytest.fixture
def model_config():
    return {
        "input_size": 32, # Smaller size for testing
        "patch_size": 4,
        "in_channels": 3,
        "hidden_size": 32,
        "depth": 2,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "class_dropout_prob": 0.1,
        "num_classes": 10,
        "learn_sigma": False
    }

def test_timestep_embedder():
    hidden_size = 32
    embedder = TimestepEmbedder(hidden_size)
    t = torch.rand(8)
    out = embedder(t)
    assert out.shape == (8, hidden_size)

def test_dit_block_forward():
    hidden_size = 32
    num_heads = 4
    block = DiTBlock(hidden_size, num_heads)
    
    # x: (N, T, D)
    x = torch.randn(2, 16, hidden_size)
    # c: (N, D) - conditioning
    c = torch.randn(2, hidden_size)
    
    out = block(x, c)
    assert out.shape == x.shape

def test_dit_forward(model_config):
    model = DiT(**model_config)
    
    batch_size = 2
    x = torch.randn(batch_size, model_config["in_channels"], model_config["input_size"], model_config["input_size"])
    t = torch.rand(batch_size)
    r = torch.rand(batch_size)
    y = torch.randint(0, model_config["num_classes"], (batch_size,))
    
    out = model(x, t, r, y)
    
    expected_out_channels = model_config["in_channels"] * 2 if model_config["learn_sigma"] else model_config["in_channels"]
    assert out.shape == (batch_size, expected_out_channels, model_config["input_size"], model_config["input_size"])

def test_dit_initialization(model_config):
    model = DiT(**model_config)
    # Check if final layer weights are zero-initialized as per standard DiT practice for stability
    assert torch.allclose(model.final_layer.linear.weight, torch.zeros_like(model.final_layer.linear.weight))
    assert torch.allclose(model.final_layer.linear.bias, torch.zeros_like(model.final_layer.linear.bias))
