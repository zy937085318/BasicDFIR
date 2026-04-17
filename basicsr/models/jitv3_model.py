'''
JiTv3Model: JiTv2Model extended with Classifier-Free Guidance (CFG).

Key changes from JiTv2Model:
1. Training: randomly drops lq condition (cfg_drop_rate) to learn unconditional distribution
2. Inference: applies CFG formula in ODE solver wrapper for controllable generation quality
3. Fully implemented tile_mode (none/split/auto) for large image inference

Config additions:
  train:
    cfg_drop_rate: 0.1     # probability of dropping lq condition per training step
  val:
    cfg_scale: 1.0         # CFG scale (1.0 = disabled, >1.0 = enhanced)
    tile_mode: auto         # 'none', 'split', 'auto'
'''

from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.path.scheduler import CondOTScheduler
from basicsr.utils.image_split import split_with_overlap, merge_with_padding
import torch
import torch.nn.functional as F


@MODEL_REGISTRY.register()  # type: ignore
class JiTv3Model(SRModel):

    def __init__(self, opt):
        super(JiTv3Model, self).__init__(opt)
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.sample_step = opt['val']['sample_step']

        # CFG inference parameter
        self.cfg_scale = opt['val'].get('cfg_scale', 1.0)

        # Tile mode: 'split', 'none', 'auto'
        val_opt = opt.get('val', {})
        self.downscale = opt['network_g'].get('downscale', 4)
        self.tile_mode = val_opt.get('tile_mode', 'auto')
        self.tile_size = opt['network_g'].get('img_size', 256)

        # CFG training parameter
        if self.is_train:
            self.cfg_drop_rate = opt['train'].get('cfg_drop_rate', 0.1)

    @torch.no_grad()
    def update_ema(self):
        for p_ema, p_net in zip(self.net_g_ema.parameters(), self.net_g.parameters()):
            p_ema.data.mul_(self.ema_decay).add_(p_net.data, alpha=1 - self.ema_decay)
        for p_ema, p_net in zip(self.net_g_ema.buffers(), self.net_g.buffers()):
            p_ema.data.copy_(p_net.data)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        self.lq_bicubic = F.interpolate(self.lq, size=self.gt.shape[-2:], mode='bicubic')

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        l_total = 0
        x_1 = self.gt
        x_0 = torch.randn_like(x_1).to(self.device)
        t = torch.rand(x_1.shape[0]).to(self.device)
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        # CFG dropout: whole-batch level, pass None to use network's c=None fallback
        lq_input = self.lq if torch.rand(1).item() >= self.cfg_drop_rate else None

        x_1_pred = self.net_g(path_sample.x_t, lq_input, path_sample.t)

        # Logging
        mse = torch.mean((x_1_pred - x_1) ** 2)
        psnr = 10. * torch.log10(255. * 255. / mse)

        # Losses
        l_pixel = F.l1_loss(x_1_pred, x_1)
        l_total += l_pixel
        loss_dict = {'l_pix': l_pixel, 'l_psnr': psnr}

        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(x_1_pred, x_1)
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

    def sample(self, lq_bicubic, lq, sample_step=None, cfg_scale=None):
        """ODE sampling with optional CFG."""
        if sample_step is None:
            sample_step = self.sample_step
        if cfg_scale is None:
            cfg_scale = self.cfg_scale

        T = torch.linspace(0, 1, sample_step).to(self.device)
        x_0 = torch.randn_like(lq_bicubic).to(self.device)

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            net_g = self.net_g_ema
        else:
            self.net_g.eval()
            net_g = self.net_g

        wrapper = self.CNFCFGWrapper(net_g, lq, cfg_scale)
        solver = ODESolver(velocity_model=wrapper)
        x_1_pred = solver.sample(time_grid=T, x_init=x_0, method='midpoint',
                                 step_size=sample_step, return_intermediates=False)
        return x_1_pred

    class CNFCFGWrapper(torch.nn.Module):
        """ODE velocity wrapper with Classifier-Free Guidance.

        When cfg_scale > 1.0:
          x_1_cfg = x_1_uncond + cfg_scale * (x_1_cond - x_1_uncond)
          v_cfg = (x_1_cfg - x) / (1 - t)

        When cfg_scale <= 1.0:
          Standard conditional inference (identical to JiTv2Model behavior).
        """

        def __init__(self, model, lq, cfg_scale=1.0):
            super().__init__()
            self.model = model
            self.lq = lq
            self.cfg_scale = cfg_scale

        def forward(self, t, x):
            with torch.no_grad():
                B = x.shape[0]
                t_batch = t.repeat(B)

                # Conditional prediction
                x_1_cond = self.model(x, self.lq, t_batch)

                if self.cfg_scale > 1.0:
                    # Unconditional prediction (c=None triggers bilinear fallback in network)
                    x_1_uncond = self.model(x, None, t_batch)
                    # CFG formula
                    x_1_cfg = x_1_uncond + self.cfg_scale * (x_1_cond - x_1_uncond)
                else:
                    x_1_cfg = x_1_cond

                # Convert data prediction to velocity
                v = (x_1_cfg - x) / (1 - t).clamp(min=1e-2)
            return v

    @torch.no_grad()
    def test(self):
        """Test with tile mode support and CFG."""
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
        else:
            self.net_g.eval()

        H, W = self.lq_bicubic.shape[-2:]

        # Determine whether to split
        need_split = False
        if self.tile_mode == 'split':
            need_split = True
        elif self.tile_mode == 'auto':
            need_split = (H > self.tile_size or W > self.tile_size)

        if need_split:
            # Split lq_bicubic at GT resolution
            lq_bc_tiles, pad_info = split_with_overlap(
                self.lq_bicubic, self.tile_size, self.tile_size,
                self.tile_size, self.tile_size, return_padding=True)
            # Split lq at LR resolution
            lq_ts = self.tile_size // self.downscale
            lq_tiles, _ = split_with_overlap(
                self.lq, lq_ts, lq_ts, lq_ts, lq_ts, return_padding=True)
            _, n_h, n_w, _, _, _ = lq_bc_tiles.shape

            for i in range(n_h):
                for j in range(n_w):
                    lq_bc_tiles[:, i, j, ...] = self.sample(
                        lq_bc_tiles[:, i, j, ...],
                        lq_tiles[:, i, j, ...],
                        self.sample_step,
                        self.cfg_scale
                    ).clamp(0, 1)

            self.output = merge_with_padding(lq_bc_tiles, pad_info).clamp(0, 1)
        else:
            self.output = self.sample(
                self.lq_bicubic, self.lq,
                self.sample_step, self.cfg_scale
            ).clamp(0, 1)

        self.net_g.train()
