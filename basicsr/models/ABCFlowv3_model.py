"""
ABCFlowv3 Model: Helmholtz-Beltrami Flow + JiT Flow Matching for image SR.

Key design (基于 Helmholtz 分解):
  1. Velocity field via Helmholtz decomposition:
     v = v_curl_free + v_solenoidal
       = ∇φ (LR引导) + ∇×ψ (深度引导)

  2. Network predicts x_1 directly (compatible with JiT inference)
     but trained with Helmholtz physical constraints.

  3. Losses:
     - Velocity loss: ||v_θ - v_target||² (primary)
     - x_1 L1 loss: ||x_1_pred - x_1_gt|| (auxiliary)
     - Helmholtz losses: ||∇·v||² + ||∇×v||² (regularization)
     - Beltrami soft: ||∇×v - λ·v||² (inductive bias)
     - Depth loss: ||depth_pred - depth_gt|| (if GT depth available)

  4. Inference: same as JiT (ODE solver, sample_step=20)
"""

import torch
import torch.nn.functional as F
from collections import OrderedDict

from basicsr.utils.registry import MODEL_REGISTRY
from .jit_model import JiTModel
from basicsr.utils.image_split import split_with_overlap, merge_with_padding


@MODEL_REGISTRY.register()
class ABCFlowv3Model(JiTModel):
    """
    Helmholtz-Beltrami Flow enhanced JiT model for super-resolution.

    Inherits JiT's complete training/inference pipeline but uses HelmholtzUNetv3
    as the backbone, which decomposes velocity prediction into:
      - Curl-free component: guided by LR condition
      - Solenoidal component: guided by depth estimation

    The depth estimator is trained jointly with the main network.
    """

    def __init__(self, opt):
        super(ABCFlowv3Model, self).__init__(opt)

        # Helmholtz-specific settings
        self.helmholtz_div_weight = opt['train'].get('helmholtz_div_weight', 0.05)
        self.helmholtz_curl_weight = opt['train'].get('helmholtz_curl_weight', 0.05)
        self.helmholtz_beltrami_weight = opt['train'].get('helmholtz_beltrami_weight', 0.01)
        self.helmholtz_lambda = opt['train'].get('helmholtz_lambda', 0.1)

        # Depth loss weight (0 = no GT depth supervision)
        self.depth_weight = opt['train'].get('depth_weight', 0.0)

        # Velocity loss weight (primary)
        self.velocity_weight = opt['train'].get('velocity_weight', 1.0)

        # Inference settings
        self.sample_step = opt['val'].get('sample_step', 20)
        self.start_noise_scale = opt['train'].get('abc_start_noise_scale', 0.3)

    def optimize_parameters(self, current_iter):
        """
        Training step with Helmholtz decomposition losses.

        Losses:
          L = w_v · L_velocity + w_pix · L_pixel + w_div · L_div
              + w_curl · L_curl + w_beltrami · L_beltrami + w_depth · L_depth
        """
        self.optimizer_g.zero_grad()

        x_1 = self.gt
        B = x_1.shape[0]
        device = x_1.device

        # Sample noise and time step (same as JiT)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(B).to(device)

        # Path sampling: x_t = (1-t)·x_0 + t·x_1
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
        x_t = path_sample.x_t

        # Network forward
        x_1_pred = self.net_g(
            torch.cat([x_t, self.lq_bicubic], dim=1),
            None,
            path_sample.t
        )

        # ================================================================
        # Primary losses
        # ================================================================

        # Velocity loss (primary)
        target_velocity = x_1 - x_0
        # Predict velocity from x_1 prediction: v = (x_1_pred - x_t) / (1 - t)
        denom = (1 - t[:, None, None, None]).clamp(min=1e-2)
        pred_velocity = (x_1_pred - x_t) / denom
        l_velocity = F.l1_loss(pred_velocity, target_velocity)

        # x_1 L1 loss (auxiliary)
        l_pixel = F.l1_loss(x_1_pred, x_1)

        l_primary = self.velocity_weight * l_velocity + 0.5 * l_pixel
        l_total = l_primary
        loss_dict = {
            'l_velocity': l_velocity.detach(),
            'l_pixel': l_pixel.detach(),
        }

        # ================================================================
        # Helmholtz regularization losses (from predicted velocity)
        # ================================================================

        # Compute predicted velocity at the current x_t
        # v = ∇φ + ∇×ψ from the network's intermediate features
        # We use (x_1_pred - x_t) / (1-t) as the predicted velocity
        v_pred = pred_velocity.detach()  # (B, 3, H, W) - take only spatial channels

        # Use the velocity magnitude for Helmholtz (average across channels)
        # v_mag = v_pred.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        # For 2D Helmholtz: we need 2-channel velocity, use first 2 channels of spatial displacement
        # Note: x_1_pred has 3 channels (RGB), we use their differences as proxy for velocity
        # This is a simplification - the actual velocity comes from the displacement
        h_feat = self.net_g._last_v if hasattr(self.net_g, '_last_v') else None

        if h_feat is not None and self.helmholtz_div_weight > 0:
            # Use the actual Helmholtz velocity from the network
            v_helm = h_feat  # (B, 2, H, W)

            # Divergence-free loss: penalize ∇·v
            dvx_dx = torch.gradient(v_helm[:, 0:1], dim=3)[0]
            dvy_dy = torch.gradient(v_helm[:, 1:2], dim=2)[0]
            if isinstance(dvx_dx, tuple):
                dvx_dx = dvx_dx[0]
            if isinstance(dvy_dy, tuple):
                dvy_dy = dvy_dy[0]
            div = dvx_dx + dvy_dy
            l_div = (div ** 2).mean()

            # Curl-free loss: penalize ∇×v
            dvy_dx = torch.gradient(v_helm[:, 1:2], dim=3)[0]
            dvx_dy = torch.gradient(v_helm[:, 0:1], dim=2)[0]
            if isinstance(dvy_dx, tuple):
                dvy_dx = dvy_dx[0]
            if isinstance(dvx_dy, tuple):
                dvx_dy = dvx_dy[0]
            curl = dvy_dx - dvx_dy
            l_curl = (curl ** 2).mean()

            # Beltrami soft constraint: ∇×v ≈ λ·|v|
            v_mag = (v_helm ** 2).sum(dim=1, keepdim=True).sqrt()
            l_beltrami = ((curl - self.helmholtz_lambda * v_mag) ** 2).mean()

            l_total = l_total + (
                self.helmholtz_div_weight * l_div +
                self.helmholtz_curl_weight * l_curl +
                self.helmholtz_beltrami_weight * l_beltrami
            )
            loss_dict['l_div'] = l_div.detach()
            loss_dict['l_curl'] = l_curl.detach()
            loss_dict['l_beltrami'] = l_beltrami.detach()

        # ================================================================
        # Depth loss (if GT depth available)
        # ================================================================
        if self.depth_weight > 0 and hasattr(self.net_g, 'depth_estimator'):
            # The depth estimator is trained jointly
            # We don't have GT depth in standard SR datasets, so this is optional
            # For now, use image structure as pseudo-depth (TV regularization on depth)
            if hasattr(self.net_g, '_last_depth'):
                depth = self.net_g._last_depth
                # Total variation on depth: encourage smooth depth
                l_depth_tv = (
                    (depth[:, :, 1:] - depth[:, :, :-1]).abs().mean() +
                    (depth[:, :, :, 1:] - depth[:, :, :, :-1]).abs().mean()
                )
                l_total = l_total + self.depth_weight * l_depth_tv
                loss_dict['l_depth'] = l_depth_tv.detach()

        # ================================================================
        # Perceptual loss
        # ================================================================
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(x_1_pred, x_1)
            if l_percep is not None:
                l_total = l_total + l_percep
                loss_dict['l_percep'] = l_percep.detach()
            if l_style is not None:
                l_total = l_total + l_style
                loss_dict['l_style'] = l_style.detach()

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        self.update_ema()

    def sample(self, lq_bicubic, sample_step=None):
        """Sample from the flow matching model (same as JiT)."""
        if sample_step is None:
            sample_step = self.sample_step

        device = self.lq_bicubic.device
        T = torch.linspace(0, 1, sample_step).to(device)

        # Starting point: lq_bicubic + small noise
        x_0 = lq_bicubic + self.start_noise_scale * torch.randn_like(lq_bicubic)

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            net_g = self.net_g_ema
        else:
            self.net_g = self.net_g.eval()
            net_g = self.net_g

        cnf = self._CNFWrapper(net_g, lq_bicubic)

        try:
            from flow_matching.solver import ODESolver
            solver = ODESolver(velocity_model=cnf)
            x_1_pred = solver.sample(
                time_grid=T, x_init=x_0, method='midpoint',
                step_size=sample_step, return_intermediates=False
            )
        except ImportError:
            # Fallback: manual Euler
            x = x_0
            dt = 1.0 / sample_step
            for step_t in T[1:]:
                with torch.no_grad():
                    x_1_pred_t = net_g(
                        torch.cat([x, lq_bicubic], dim=1), None, step_t.repeat(x.shape[0])
                    )
                    v = (x_1_pred_t - x) / max(1e-2, 1 - step_t.item())
                    x = x + v * dt
            x_1_pred = x

        return x_1_pred

    def test(self):
        """Test with split-merge for large images."""
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
        """CNF wrapper for ODE-based sampling."""

        def __init__(self, model, lq_bicubic):
            super().__init__()
            self.model = model
            self.lq_bicubic = lq_bicubic

        def forward(self, t, x):
            B = x.shape[0]
            x_1_pred = self.model(
                torch.cat([x, self.lq_bicubic], dim=1), None, t.repeat(B)
            )
            denom = max(1e-2, 1 - t.item())
            v = (x_1_pred - x) / denom
            return v
