"""
ABCFlowv2 Model: ABC Flow enhanced JiT (Flow Matching) for image super-resolution.

Key design choices (fixing issues from ABCFlowv1):
  1. ABC flow is embedded as feature modulation INSIDE the UNet backbone,
     NOT as the output. This preserves JiT's conditional optimal transport path.
  2. ABC parameters are spatially-varying (predicted from features), not global.
  3. Velocity loss instead of x_1 loss: more stable gradients across t.
  4. Increased sample_step (20 vs 5): better ODE integration accuracy.
  5. Inference starts from lq_bicubic + small noise, not pure randn.
  6. Gradient-domain regularization instead of unphysical Beltrami loss.
  7. EMA for stable training.
"""

import torch
import torch.nn.functional as F
from collections import OrderedDict

from basicsr.utils.registry import MODEL_REGISTRY
from .jit_model import JiTModel
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from basicsr.utils.image_split import split_with_overlap, merge_with_padding


@MODEL_REGISTRY.register()
class ABCFlowv2Model(JiTModel):
    """
    ABC Flow enhanced JiT model for super-resolution.

    Inherits the complete JiT training/inference pipeline but uses ABCUNetv2
    as the backbone, which internally applies content-aware ABC flow modulation
    to intermediate features.

    The ABC flow acts as a spatial attention mechanism:
      - Modulates features based on local image structure (gradients)
      - Provides inductive bias for edge/texture enhancement
      - Preserves JiT's temporal flow matching objective
    """

    def __init__(self, opt):
        super(ABCFlowv2Model, self).__init__(opt)

        # Override sample_step with recommended value (20 instead of 5)
        self.sample_step = opt['val'].get('sample_step', 20)

        # ABC flow specific: gradient regularization weight
        self.grad_reg_weight = opt['train'].get('abc_grad_reg_weight', 0.05)

        # Inference: noise scale for starting point
        self.start_noise_scale = opt['train'].get('abc_start_noise_scale', 0.3)

    def optimize_parameters(self, current_iter):
        """
        Training step using velocity loss + gradient regularization.

        Velocity loss: directly supervises the velocity field predicted by the network,
        which is more stable than x_1 loss (avoids division by 1-t near t=1).

        For CondOTScheduler (α_t=t, σ_t=1-t):
          - x_t = (1-t) * x_0 + t * x_1
          - Target velocity: dx_t/dt = x_1 - x_0 (constant along the path)
          - Predicted velocity: v_θ(x_t, lq, t) = (x_1_pred - x_t) / (1-t)
          - Loss: ||v_θ - (x_1 - x_0)||²

        Gradient regularization: encourages the output to preserve image gradients,
        which is a more meaningful spatial constraint than Beltrami loss.
        """
        self.optimizer_g.zero_grad()

        x_1 = self.gt  # target HR image
        B = x_1.shape[0]
        device = x_1.device

        # Sample noise and time step
        x_0 = torch.randn_like(x_1)
        t = torch.rand(B).to(device)

        # Conditional optimal transport path sampling
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
        x_t = path_sample.x_t  # (B, 3, H, W)

        # Network forward: predict x_1 from (x_t, lq_bicubic, t)
        x_1_pred = self.net_g(
            torch.cat([x_t, self.lq_bicubic], dim=1),
            None,
            path_sample.t
        )

        # === Velocity loss (primary) ===
        # Target velocity: dx_t/dt = x_1 - x_0 (constant along straight-line path)
        target_velocity = x_1 - x_0  # (B, 3, H, W)

        # Predicted velocity: from x_1 prediction
        # v_θ = (x_1_pred - x_t) / (1 - t)
        # Use clamp to prevent division by zero when t -> 1
        denom = (1 - t[:, None, None, None]).clamp(min=1e-2)
        pred_velocity = (x_1_pred - x_t) / denom

        # L1 loss on velocity field (more stable than x_1 loss)
        l_velocity = F.l1_loss(pred_velocity, target_velocity)

        # === x_1 prediction loss (auxiliary, kept for direct supervision) ===
        l_pixel = F.l1_loss(x_1_pred, x_1)

        # Combined primary loss
        l_primary = l_velocity + 0.5 * l_pixel
        l_total = l_primary

        loss_dict = {
            'l_velocity': l_velocity,
            'l_pixel': l_pixel,
        }

        # === Gradient regularization (replaces unphysical Beltrami loss) ===
        # Encourages x_1_pred to preserve image structure (edges, textures)
        # This is a meaningful constraint for SR: smooth areas should remain smooth,
        # edges should be sharp.
        if self.grad_reg_weight > 0:
            # First-order gradient (edge-aware)
            grad_x_pred = torch.gradient(x_1_pred, dim=3)[0]
            grad_y_pred = torch.gradient(x_1_pred, dim=2)[0]
            grad_x_gt = torch.gradient(x_1, dim=3)[0]
            grad_y_gt = torch.gradient(x_1, dim=2)[0]

            l_grad = (
                F.l1_loss(grad_x_pred, grad_x_gt) +
                F.l1_loss(grad_y_pred, grad_y_gt)
            ) * 0.5

            l_total = l_total + self.grad_reg_weight * l_grad
            loss_dict['l_grad'] = l_grad

        # === Perceptual loss ===
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(x_1_pred, x_1)
            if l_percep is not None:
                l_total = l_total + l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total = l_total + l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        self.update_ema()

    def sample(self, lq_bicubic, sample_step=None):
        """
        Sample from the flow matching model.

        Uses adaptive step size (20 steps by default) for better ODE integration.
        Starting point: lq_bicubic + small_noise (not pure randn).
        This keeps the generation close to the LR condition, reducing noise.
        """
        if sample_step is None:
            sample_step = self.sample_step

        device = self.lq_bicubic.device
        T = torch.linspace(0, 1, sample_step).to(device)

        # Starting point: LR bicubic + small noise, not pure randn
        # This provides a stronger conditioning signal and reduces stochasticity
        x_0 = lq_bicubic + self.start_noise_scale * torch.randn_like(lq_bicubic)

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            net_g = self.net_g_ema
        else:
            net_g = self.net_g.eval()

        cnf = self._CNFWrapper(net_g, lq_bicubic)

        # Import solver lazily to avoid import errors
        try:
            from flow_matching.solver import ODESolver
            solver = ODESolver(velocity_model=cnf)
            x_1_pred = solver.sample(
                time_grid=T,
                x_init=x_0,
                method='midpoint',
                step_size=sample_step,
                return_intermediates=False
            )
        except ImportError:
            # Fallback: manual Euler integration
            x = x_0
            dt = 1.0 / sample_step
            for step_t in T[1:]:
                with torch.no_grad():
                    x_1_pred_t = net_g(
                        torch.cat([x, lq_bicubic], dim=1),
                        None,
                        step_t.repeat(x.shape[0])
                    )
                    # Velocity: (x_1_pred - x) / (1 - t)
                    v = (x_1_pred_t - x) / max(1e-2, 1 - step_t.item())
                    x = x + v * dt
            x_1_pred = x

        return x_1_pred

    def test(self):
        """Test/inference with split-merge strategy for large images."""
        with torch.no_grad():
            img = self.lq_bicubic
            split_h = self.opt['network_g']['img_size']
            split_w = self.opt['network_g']['img_size']
            split_img, padding_info = split_with_overlap(
                img, split_h, split_w, split_h, split_w, return_padding=True
            )
            [_, n_h, n_w, _, _, _] = split_img.shape
            for i in range(n_h):
                for j in range(n_w):
                    split_img[:, i, j, ...] = self.sample(
                        split_img[:, i, j, ...], self.sample_step
                    )
            merged_img = merge_with_padding(split_img, padding_info)
        self.net_g.train()
        self.output = merged_img.clamp(0, 1)

    class _CNFWrapper(torch.nn.Module):
        """
        Continuous Normalizing Flow wrapper for ODE-based sampling.

        Wraps the network to produce velocity field v(t, x) from x_1 prediction.
        For CondOTScheduler: x_t = (1-t)*x_0 + t*x_1
        Velocity: dx_t/dt = x_1 - x_0
        From network: v = (x_1_pred - x) / (1-t)
        """

        def __init__(self, model, lq_bicubic):
            super().__init__()
            self.model = model
            self.lq_bicubic = lq_bicubic

        def forward(self, t, x):
            """
            Args:
                t: scalar tensor, time in [0, 1]
                x: (B, 3, H, W) current state along the path
            Returns:
                v: (B, 3, H, W) velocity field dx/dt
            """
            B = x.shape[0]
            # Predict x_1 from current state x and conditioning
            x_1_pred = self.model(
                torch.cat([x, self.lq_bicubic], dim=1),
                None,
                t.repeat(B)
            )
            # Convert x_1 prediction to velocity
            # v = (x_1_pred - x) / (1 - t)
            # Use clamp to avoid division by zero at t=1
            denom = max(1e-2, 1 - t.item())
            v = (x_1_pred - x) / denom
            return v
