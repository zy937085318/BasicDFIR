import pytest
import torch
import torch.nn as nn
from pmf.optimizer import Muon, configure_optimizers
from pmf.config import Config

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 2D weight for Muon
        self.linear = nn.Linear(10, 10, bias=False)
        # 1D bias for AdamW
        self.bias = nn.Parameter(torch.zeros(10))
        
    def forward(self, x):
        return self.linear(x) + self.bias

def test_muon_optimizer_step():
    model = SimpleModel()
    params = list(model.linear.parameters())
    # Ensure gradients
    x = torch.randn(2, 10)
    loss = model(x).sum()
    loss.backward()
    
    optimizer = Muon(params, lr=0.01)
    
    # Check weights before
    w_before = model.linear.weight.clone()
    
    optimizer.step()
    
    # Check weights after
    w_after = model.linear.weight
    assert not torch.allclose(w_before, w_after)

def test_optimizer_configuration():
    config = Config()
    model = SimpleModel()
    
    optimizers = configure_optimizers(model, config)
    assert len(optimizers) == 2
    
    # First should be Muon (hidden params - linear weight is 2D)
    assert isinstance(optimizers[0], Muon)
    # Second should be AdamW (other params - bias is 1D)
    assert isinstance(optimizers[1], torch.optim.AdamW)
