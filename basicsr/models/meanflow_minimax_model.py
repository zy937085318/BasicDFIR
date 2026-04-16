"""
Meanflow Minimax Model for BasicSR framework.

This model implements MeanFlow sampling with Minimax optimization for image super-resolution.
It uses the meanflow_sampler from the MeanFlow-main project and integrates with BasicSR.

Author: Matrix Agent
"""

import torch
import torch.nn.functional as F
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel
from basicsr.utils import split_with_overlap, merge_with_padding

# Import meanflow_sampler from MeanFlow-main project
import sys
import os

# Add MeanFlow-main path to sys.path for importing
meanflow_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'flow-based codezoo', 'MeanFlow-main')
if meanflow_path not in sys.path:
    sys.path.insert(0, meanflow_path)

try:
    from meanflow_sampler import meanflow_sampler
except ImportError:
    # Fallback: define meanflow_sampler function inline if import fails
    print(f"Warning: Could not import meanflow_sampler from {meanflow_path}, using inline version")

    @torch.no_grad()
    def meanflow_sampler(
        model,
        latents,
        y=None,
        cfg_scale=1.0,
        num_steps=1,
        **kwargs
    ):
        """
        MeanFlow sampler supporting both single-step and multi-step generation.

        Based on Eq.(12): z_r = z_t - (t-r)u(z_t, r, t)
        For single-step: z_0 = z_1 - u(z_1, 0, 1)
        For multi-step: iteratively apply the Eq.(12) with intermediate steps
        """
        batch_size = latents.shape[0]
        device = latents.device

        # Prepare for CFG if needed
        do_cfg = y is not None and cfg_scale > 1.0
        if do_cfg:
            if hasattr(model, 'module'):  # DDP
                num_classes = model.module.num_classes
            else:
                num_classes = model.num_classes
            null_y = torch.full_like(y, num_classes)

        if num_steps == 1:
            r = torch.zeros(batch_size, device=device)
            t = torch.ones(batch_size, device=device)

            if do_cfg:
                z_combined = torch.cat([latents, latents], dim=0)
                r_combined = torch.cat([r, r], dim=0)
                t_combined = torch.cat([t, t], dim=0)
                y_combined = torch.cat([y, null_y], dim=0)

                u_combined = model(z_combined, r_combined, t_combined, y=y_combined)
                u_cond, u_uncond = u_combined.chunk(2, dim=0)

                u = u_uncond + cfg_scale * (u_cond - u_uncond)
            else:
                u = model(latents, r, t, y=y)

            # x_0 = x_1 - u(x_1, 0, 1)
            x0 = latents - u

        else:
            z = latents

            time_steps = torch.linspace(1, 0, num_steps + 1, device=device)

            for i in range(num_steps):
                t_cur = time_steps[i]
                t_next = time_steps[i + 1]

                t = torch.full((batch_size,), t_cur, device=device)
                r = torch.full((batch_size,), t_next, device=device)

                if do_cfg:
                    z_combined = torch.cat([z, z], dim=0)
                    r_combined = torch.cat([r, r], dim=0)
                    t_combined = torch.cat([t, t], dim=0)
                    y_combined = torch.cat([y, null_y], dim=0)

                    u_combined = model(z_combined, r_combined, t_combined, y=y_combined)
                    u_cond, u_uncond = u_combined.chunk(2, dim=0)

                    # Apply CFG
                    u = u_uncond + cfg_scale * (u_cond - u_uncond)
                else:
                    u = model(z, r, t, y=y)

                # Update z: z_r = z_t - (t-r)*u(z_t, r, t)
                z = z - (t_cur - t_next) * u

            x0 = z

        return x0


@MODEL_REGISTRY.register()
class Meanflow_minimax_Model(SRModel):
    """MeanFlow Minimax Model for Super Resolution."""

    def __init__(self, opt):
        super(Meanflow_minimax_Model, self).__init__(opt)
        self.sample_step = opt['val']['sample_step']
        self.cfg_scale = opt['val'].get('cfg_scale', 1.0)
        self.cond = opt['val'].get('cond', None)

    @torch.no_grad()
    def update_ema(self):
        """Update EMA parameters."""
        for p_ema, p_net in zip(self.net_g_ema.parameters(), self.net_g.parameters()):
            p_ema.data.mul_(self.ema_decay).add_(p_net.data, alpha=1 - self.ema_decay)
        for p_ema, p_net in zip(self.net_g_ema.buffers(), self.net_g.buffers()):
            p_ema.data.copy_(p_net.data)

    def feed_data(self, data):
        """Feed data to the model."""
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.lq_bicubic = F.interpolate(self.lq, size=self.gt.shape[-2:], mode='bicubic')

    def optimize_parameters(self, current_iter):
        """Optimize model parameters."""
        self.optimizer_g.zero_grad()

        l_total = 0
        x_1 = self.gt
        x_0 = torch.randn_like(x_1).to(self.device)

        # Sample time steps
        t = torch.rand(x_1.shape[0]).to(self.device)

        # Flow interpolation: xt = (1-t)*x_1 + t*x_0
        xt = (1 - t.view(-1, 1, 1, 1)) * x_1 + t.view(-1, 1, 1, 1) * x_0

        # Model prediction
        vt_pred = self.net_g(torch.cat([xt, self.lq_bicubic], dim=1), None, t)

        # Target velocity: vt = x_0 - x_1
        vt_target = x_0 - x_1

        # Flow loss
        l_flow = F.mixed_exp_l1_loss(vt_pred, vt_target)
        l_total += l_flow

        loss_dict = {'l_flow': l_flow}

# Pixel loss
        if self.cri_pix:
            x_1_pred = xt + vt_pred * (1 - t.view(-1, 1, 1, 1))
            l_pix = self.cri_pix(x_1_pred, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # Perceptual loss
        if self.cri_perceptual:
            x_1_pred = xt + vt_pred * (1 - t.view(-1, 1, 1, 1))
            l_percep, l_style = self.cri_perceptual(x_1_pred, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        self.update_ema()

    def sample(self, lq_bicubic, sample_step=None, cfg_scale=None):
        """Sample one image."""
        if sample_step is None:
            sample_step = self.sample_step
        if cfg_scale is None:
            cfg_scale = self.cfg_scale

        # Determine which model to use
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            net_g = self.net_g_ema
        else:
            self.net_g.eval()
            net_g = self.net_g

        # Initialize latents (noise at t=1)
        x_0 = torch.randn_like(lq_bicubic).to(self.device)

        # Prepare condition (y) if needed
        y = None
        if self.cond is not None:
            y = torch.full((lq_bicubic.shape[0],), self.cond, device=self.device, dtype=torch.long)

        # Use meanflow_sampler for sampling
        with torch.no_grad():
            output = meanflow_sampler(
                model=net_g,
                latents=x_0,
                y=y,
                cfg_scale=cfg_scale,
                num_steps=sample_step
            )

        net_g.train()

        return output

    @torch.no_grad()
    def test(self):
        """Test/inference function."""
        img = self.lq_bicubic
        split_h = self.opt['network_g']['img_size']
        split_w = self.opt['network_g']['img_size']

        split_img, padding_info = split_with_overlap(img, split_h, split_w, split_h, split_w, return_padding=True)
        [_, n_h, n_w, _, _, _] = split_img.shape

        for i in range(n_h):
            for j in range(n_w):
                split_img[:, i, j, ...] = self.sample(split_img[:, i, j, ...], self.sample_step, self.cfg_scale)

        merged_img = merge_with_padding(split_img, padding_info)

        self.net_g.train()
        self.output = merged_img.clamp(0, 1)
