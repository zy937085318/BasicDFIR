'''
JiTv2Model: Modified JiTModel with separate x_t and lq inputs + optional image splitting.

Key changes from JiTModel:
1. Network forward: (x_t, lq_bicubic, t) instead of (cat([x_t, lq]), None, t)
   - lq is passed as condition `c` for content-adaptive Token Dictionary
2. Configurable tile_mode in val section:
   - 'split': always split (same as JiTModel)
   - 'none': never split, process full image
   - 'auto': split only when image > img_size (default)
'''

from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.path.scheduler import CondOTScheduler
from basicsr.utils.image_split import split_with_overlap, merge_with_padding
import torch


@MODEL_REGISTRY.register()  # type: ignore
class JiTv2Model(SRModel):
    def __init__(self, opt):
        super(JiTv2Model, self).__init__(opt)
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.sample_step = opt['val']['sample_step']

        # Tile mode: 'split', 'none', 'auto'
        val_opt = opt.get('val', {})
        self.downscale = opt['network_g'].get('downscale', 4)
        self.tile_mode = val_opt.get('tile_mode', 'auto')
        self.tile_size = opt['network_g'].get('img_size', 256)

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
        self.lq_bicubic = torch.nn.functional.interpolate(self.lq, size=self.gt.shape[-2:], mode='bicubic')

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        l_total = 0
        x_1 = self.gt
        x_0 = torch.randn_like(x_1).to(self.device)
        t = torch.rand(x_1.shape[0]).to(self.device)
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        # JiTv2Model: net_g(xt, lq, t) where xt is noisy image from x0-GT interpolation
        x_1_pred = self.net_g(path_sample.x_t, self.lq, path_sample.t)
        v_t_pred = (x_1_pred - path_sample.x_t) / (
            1 - t[:, None, None, None].expand(-1, 3, -1, -1)).clamp_min(5e-2)
        
        mse = torch.mean((x_1_pred - x_1)**2)
        psnr = 10. * torch.log10(255. * 255. / mse)
        l_pixel = torch.nn.functional.l1_loss(x_1_pred, x_1)
        l_total += l_pixel
        loss_dict = {'l_pix': l_pixel, 'l_psnr': psnr}
        # perceptual loss
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

    def sample(self, lq_bicubic, lq, sample_step=5):
        T = torch.linspace(0, 1, sample_step).to(self.device)
        x_0 = torch.randn_like(lq_bicubic).to(self.device)
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            net_g = self.net_g_ema
        else:
            net_g = self.net_g.eval()
        cnf = self.cnf(net_g, lq)
        solver = ODESolver(velocity_model=cnf)
        x_1_pred = solver.sample(time_grid=T, x_init=x_0, method='midpoint',
                                 step_size=sample_step, return_intermediates=False)
        return x_1_pred

    class cnf(torch.nn.Module):
        def __init__(self, model, lq):
            super().__init__()
            self.model = model
            self.lq = lq

        def forward(self, t, x):
            with torch.no_grad():
                # Key change: pass lq as separate condition, not concatenated
                x_1_pred = self.model(x, self.lq, t.repeat(x.shape[0]))
                v_t_pred = (x_1_pred - x) / (1 - t)
            return v_t_pred

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.sample(self.lq_bicubic, self.lq, self.sample_step).clamp(0, 1)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.sample(self.lq_bicubic, self.lq, self.sample_step).clamp(0, 1)
            self.net_g.train()
        self.net_g.train()
