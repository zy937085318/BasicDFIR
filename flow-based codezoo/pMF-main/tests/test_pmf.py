
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from pmf.config import Config
from pmf.pixel_mean_flow import PixelMeanFlow
from pmf.dit import DiT

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        # A simple parameter to check gradients
        self.w = nn.Parameter(torch.tensor(0.5))

    def forward(self, z, t, r, y, w=None, cfg_interval=None):
        # Simple deterministic output: x_pred = w * z
        return self.w * z

@pytest.fixture
def config():
    c = Config()
    c.image_size = 32
    c.patch_size = 4
    c.hidden_size = 32
    c.depth = 1
    c.num_heads = 4
    c.num_classes = 10
    c.micro_batch_size = 2
    c.lambda_perc = 0.0
    c.cfg_training = False
    c.class_dropout_prob = 0.0 # Disable dropout to simplify RNG sync in tests
    return c

@pytest.fixture
def pmf(config):
    # Use a real DiT for shape/integration tests
    model = DiT(
        input_size=config.image_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        num_classes=config.num_classes
    )
    return PixelMeanFlow(model, config)

def test_pmf_jvp_correctness(config):
    """
    Verify Algorithm 1 JVP calculation using a mock model with known derivatives.
    u_fn = (z - net(z))/t
    If net(z) = w * z (w=0.5), then u_fn = (1-w) * z / t = 0.5 * z / t.

    Target velocity for JVP tangents is v = u_fn(z, t, t) = 0.5 * z / t.

    JVP (du/dt_total) = (du/dz)*v + (du/dt)*1 + (du/dr)*0
    du/dz = 0.5 / t
    du/dt = -0.5 * z / t^2

    JVP = (0.5/t) * (0.5 * z / t) + (-0.5 * z / t^2)
        = 0.25 * z / t^2 - 0.5 * z / t^2
        = -0.25 * z / t^2
    """
    mock_model = MockModel()
    pmf_mock = PixelMeanFlow(mock_model, config)

    batch_size = 1
    # Fixed inputs
    z = torch.randn(batch_size, 3, 32, 32)
    t = torch.tensor([0.5])
    r = torch.tensor([0.2]) # Arbitrary r < t
    y = torch.tensor([0])

    # We need to access the internal logic. Since it's inside forward_loss,
    # we can't easily isolate the JVP without replicating the code or modifying the class.
    # However, we can check the result of forward_loss if we know the target.

    # But forward_loss does a lot of random sampling.
    # Let's monkeypatch sample_t_r to return our fixed t, r
    pmf_mock.sample_t_r = lambda b, d: (t.to(d), r.to(d))

    # Also we need to control 'e' (noise) to control 'z' inside forward_loss.
    # z_internal = (1-t)*x + t*e.
    # If we pass x, z is derived.
    # Let's pass x such that we know z.
    # If we set x = 0, then z = t*e.
    # v_target = e - x = e.
    # u_fn = (z - 0.5*z)/t = 0.5 * z / t = 0.5 * t * e / t = 0.5 * e.
    # v (for tangent) = u_fn(z, t, t) = 0.5 * e.
    # JVP calculation:
    # u_fn = 0.5 * z_arg / t_arg.
    # du/dz = 0.5 / t_arg.
    # du/dt = -0.5 * z_arg / t_arg^2.
    # jvp = (0.5/t)*(0.5*e) + (-0.5*z/t^2)*1
    #     = 0.25*e/t - 0.5*(t*e)/t^2
    #     = 0.25*e/t - 0.5*e/t
    #     = -0.25 * e / t.

    # V = u + (t - r) * dudt
    #   = 0.5 * e + (t - r) * (-0.25 * e / t)
    #   = e * (0.5 - 0.25 * (t - r) / t)

    x = torch.zeros(batch_size, 3, 32, 32)
    # To fix 'e', we can seed RNG, but 'e' is generated inside.
    # We can rely on the fact that V and v_target are related.
    # Or we can temporarily patch torch.randn_like.

    torch.manual_seed(42)
    # We can't easily patch randn_like inside a function without global side effects.
    # But we can check the loss.
    # Loss = MSE(V, v_target). v_target = e.
    # Loss = MSE( e * (0.5 - 0.25 * (t-r)/t), e )
    #      = MSE( e * (0.5 - 0.25 * (0.3)/0.5), e )
    #      = MSE( e * (0.5 - 0.15), e )
    #      = MSE( 0.35 * e, e )
    #      = (0.65 * e)^2

    # Let's run it.
    loss, loss_dict = pmf_mock.forward_loss(x, y)

    # We don't know exact 'e', but we know the ratio of Loss/MSE(e).
    # Wait, we can't easily get MSE(e) from outside.

    # Alternative: Subclass PixelMeanFlow to intercept locals? No.
    # Alternative: Trust the math derived above and just run it to see if it doesn't crash,
    # AND verify the 'detach' behavior.

    # Check gradients.
    # V = u + (t-r) * stop_grad(dudt).
    # u depends on w. dudt depends on w.
    # V's grad w.r.t w should only come from u, not dudt.
    # u = 0.5 * z / t = 0.5 * e. Depends on w (0.5 is w). u = w * e.
    # dudt = -0.5 * w * e / t. (Wait, earlier calc was with w=0.5 constant).
    # Let's redo with symbolic w.
    # u_fn = (z - w*z)/t = (1-w)z/t.
    # v_tangent = (1-w)z/t.
    # du/dz = (1-w)/t.
    # du/dt = -(1-w)z/t^2.
    # JVP = ((1-w)/t) * ((1-w)z/t) + (-(1-w)z/t^2) * 1
    #     = (1-w)^2 z / t^2 - (1-w) z / t^2
    #     = [ (1-w)^2 - (1-w) ] z / t^2
    #     = (1-w)(1-w - 1) z / t^2
    #     = (1-w)(-w) z / t^2
    #     = -w(1-w) z / t^2.

    # V = u + (t-r)*dudt
    #   = (1-w)z/t + (t-r)*(-w(1-w)z/t^2)
    #   = (1-w)(z/t) [ 1 - w(t-r)/t ]

    # If we stop_grad(dudt), then for gradient calculation, dudt is constant C.
    # V_train = u + (t-r)*C
    # dV/dw = du/dw = d/dw [ (1-w)z/t ] = -z/t.

    # If we didn't stop_grad:
    # dV/dw = du/dw + (t-r)*d(dudt)/dw
    #       = -z/t + (t-r) * d/dw [ -w(1-w) z/t^2 ]
    #       = -z/t + (t-r) * (z/t^2) * [ -(1-2w) ]
    #       = -z/t - (t-r)(1-2w)z/t^2.

    # So we can check the gradient value on mock_model.w.

    # Setup inputs again
    z_val = torch.randn(batch_size, 3, 32, 32)
    x = torch.zeros_like(z_val) # So z = t*e. But wait, z is created inside.
    # If x=0, z = t*e.
    # We can't control e easily.

    # Let's try to pass z through a side channel or mock forward_loss partially?
    # No, cleaner way:
    # Just run forward_loss. It computes gradients on w.
    # We need to estimate what the gradient *should* be.
    # Since we can't know 'z' (random), we can't predict exact gradient.

    # Idea: Hack the random number generator to be deterministic.
    torch.manual_seed(123)
    loss, _ = pmf_mock.forward_loss(x, y)
    grad_w = torch.autograd.grad(loss, mock_model.w)[0]

    # Now run a "simulation" with the same seed to reconstruct variables
    torch.manual_seed(123)
    # Replicate forward_loss logic locally to verify
    t_ref, r_ref = t, r # We patched sample_t_r
    # Noise
    e_ref = torch.randn_like(x)
    z_ref = (1 - t_ref.view(-1,1,1,1)) * x + t_ref.view(-1,1,1,1) * e_ref

    w_val = mock_model.w.detach() # 0.5
    # Calculate expected V and Loss with stop_grad
    # u = (1-w)*z/t
    # dudt = -w(1-w)z/t^2
    # V = u + (t-r) * dudt.detach() (conceptually)

    # In pytorch graph:
    # u_node = (1 - w_param) * z_ref / t_ref
    # dudt_const = -w_val * (1 - w_val) * z_ref / t_ref**2
    # V_node = u_node + (t_ref - r_ref) * dudt_const
    # v_target = e_ref
    # Loss = MSE(V_node, v_target)

    # Let's compute this gradient using autograd on a local tensor
    w_param = torch.tensor(0.5, requires_grad=True)
    u_local = (1 - w_param) * z_ref / t_ref
    dudt_local = -w_val * (1 - w_val) * z_ref / t_ref**2
    V_local = u_local + (t_ref - r_ref) * dudt_local
    loss_local = F.mse_loss(V_local, e_ref)
    grad_ref = torch.autograd.grad(loss_local, w_param)[0]

    assert torch.allclose(grad_w, grad_ref, atol=1e-5), f"Gradient mismatch! Expected {grad_ref}, got {grad_w}. Stop-grad might be missing."

def test_pmf_shapes(pmf):
    batch_size = 2
    x = torch.randn(batch_size, 3, 32, 32)
    y = torch.randint(0, 10, (batch_size,))

    loss, loss_dict = pmf.forward_loss(x, y)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert "loss_pmf" in loss_dict

def test_pmf_sample(pmf):
    batch_size = 2
    z = torch.randn(batch_size, 3, 32, 32)
    y = torch.randint(0, 10, (batch_size,))

    # CFG = 1.0
    x_out = pmf.sample(z, y, cfg_scale=1.0)
    assert x_out.shape == z.shape

    # CFG != 1.0
    x_out_cfg = pmf.sample(z, y, cfg_scale=2.0)
    assert x_out_cfg.shape == z.shape

    # With Interval
    interval = torch.tensor([[0.1, 0.9]] * batch_size)
    x_out_interval = pmf.sample(z, y, cfg_scale=2.0, cfg_interval=interval)
    assert x_out_interval.shape == z.shape

def test_pmf_alignment_verification():
    """
    Runs the smoke tests originally in verify_alignment.py
    """
    from pmf.config import Config
    # Just a quick integration check
    c = Config()
    c.image_size = 32
    c.hidden_size = 32
    c.depth = 1
    c.num_heads = 4
    c.lambda_perc = 0.0

    model = DiT(
        input_size=c.image_size,
        patch_size=c.patch_size,
        in_channels=c.in_channels,
        hidden_size=c.hidden_size,
        depth=c.depth,
        num_heads=c.num_heads,
        num_classes=c.num_classes
    )
    pmf = PixelMeanFlow(model, c)

    x = torch.randn(2, 3, 32, 32)
    y = torch.randint(0, 9, (2,))
    loss, _ = pmf.forward_loss(x, y)
    assert torch.isfinite(loss)
