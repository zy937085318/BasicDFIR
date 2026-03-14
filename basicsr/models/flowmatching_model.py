import torch
from torch.nn import functional as F
from collections import OrderedDict
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.path.scheduler import CondOTScheduler
from .utils import *
from basicsr.utils.image_split import split_with_overlap, merge_with_padding


@MODEL_REGISTRY.register()
class FlowMatchingModel(SRModel):
    def __init__(self, opt):
        super(FlowMatchingModel, self).__init__(opt)
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.sample_step = opt['val']['sample_step']


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
        vt_pred = self.net_g(torch.cat([path_sample.x_t, self.lq_bicubic], dim=1), None, path_sample.t)
        x_t_pred = vt_pred * t.view(-1,1,1,1) + x_0
        x_1_pred = vt_pred * (1 - t.view(-1,1,1,1)) + x_t_pred
        l_pixel = torch.nn.functional.l1_loss(vt_pred, path_sample.dx_t)
        l_total += l_pixel
        loss_dict = {'l_pixel': l_pixel}
        # perceptual loss
        if self.cri_perceptual:
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

    '''
    采样一张图
    '''
    def sample(self, lq_bicubic, sample_step=5):
        T = torch.linspace(0, 1, sample_step).to(self.device)
        x_0 = torch.randn_like(lq_bicubic).to(self.device)
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            net_g = self.net_g_ema
        else:
            net_g = self.net_g.eval()
        cnf = self.cnf(net_g, lq_bicubic)
        solver = ODESolver(velocity_model=cnf)
        x_1_pred = solver.sample(time_grid=T, x_init=x_0, method='midpoint', step_size=sample_step,
                                 return_intermediates=False)
        return x_1_pred


    def test(self):
        with torch.no_grad():
            img = self.lq_bicubic
            split_h = self.opt['network_g']['img_size']
            split_w = self.opt['network_g']['img_size']
            split_img, padding_info = split_with_overlap(img, split_h, split_w, split_h, split_w, return_padding=True)
            [_, n_h, n_w, _, _, _] = split_img.shape
            for i in range(n_h):
                for j in range(n_w):
                    split_img [:, i, j, ...] = self.sample(split_img [:, i, j, ...], self.sample_step)
            merged_img = merge_with_padding(split_img, padding_info)
        self.net_g.train()
        self.output = merged_img.clamp(0, 1)

    '''
    ODE Solver的包装器
    '''
    class cnf(torch.nn.Module):

        def __init__(self, model, lr_bicubic):
            super().__init__()
            self.model = model
            self.lr_bicubic = lr_bicubic

        def forward(self, t, x):
            with torch.no_grad():
                z = self.model(torch.cat([x, self.lr_bicubic],dim=1), None, t.repeat(x.shape[0]))
            return z