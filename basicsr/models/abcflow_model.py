from asyncio.constants import SSL_SHUTDOWN_TIMEOUT

from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.path.scheduler import CondOTScheduler
from .utils import *
from basicsr.utils.image_split import split_with_overlap, merge_with_padding
import torch


@MODEL_REGISTRY.register() # type: ignore
class ABCFlowModel(SRModel):
    def __init__(self, opt):
        super(ABCFlowModel, self).__init__(opt)
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.sample_step = opt['val']['sample_step']#采样步数


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
        t = torch.rand(x_1.shape[0]).to(self.device)#[8],net的输入要求
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
        x_1_pred = self.net_g(torch.cat([path_sample.x_t, self.lq_bicubic], dim=1), None, path_sample.t)
        v_t_pred = (x_1_pred - path_sample.x_t) / (1 - t[:, None, None, None].expand(-1, 3, -1, -1)).clamp_min(5e-2)
        l_pixel = torch.nn.functional.l1_loss(x_1_pred, x_1)
        l_total += l_pixel
        loss_dict = {'l_pixel': l_pixel}
        beltrami_w=0.1
        if beltrami_w > 0:
                # ABC流满足 ∇×v = v (Beltrami条件)
                # 简化: 用散度和旋度来约束
                vx, vy, vz = v_t_pred[:, 0], v_t_pred[:, 1], v_t_pred[:, 2]
                
                # 2D散度 (忽略通道方向)
                dvx_dx = torch.gradient(vx, dim=2)[0]
                dvy_dy = torch.gradient(vy, dim=1)[0]
                div = dvx_dx + dvy_dy
                
                # 2D旋度
                curl = torch.gradient(vy, dim=2)[0] - torch.gradient(vx, dim=1)[0]
                
                loss_b = (div ** 2).mean() + 0.5 * (curl ** 2).mean()
                
                # ABC参数正则 (让参数接近1.0)
                loss_abc = ((self.net_g.A - 1.0) ** 2 + (self.net_g.B - 1.0) ** 2 + (self.net_g.C - 1.0) ** 2)
                
                ABCloss = beltrami_w * loss_b + 0.01 * loss_abc
                l_total += ABCloss
                loss_dict = {'ABCLoss': ABCloss}
        
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

    '''
        ODE Solver的包装器
    '''

    class cnf(torch.nn.Module):

        def __init__(self, model, lr_bicubic):
            super().__init__()
            self.model = model
            self.lr_bicubic = lr_bicubic

        def forward(self, t, x):#此处t为一个size=[]的tensor
            with torch.no_grad():
                x_1_pred = self.model(torch.cat([x, self.lr_bicubic], dim=1), None, t.repeat(x.shape[0]))
                # print(t.size())
                # print(t.dtype)
                # print(t)
                # if t.size == []:
                #     print('hhhh')
                #     t = t.unsqueeze(0)
                # t_ = t[:, None, None, None].expand(-1, 3, -1, -1)
                v_t_pred = (x_1_pred - x)/(1-t) #.clamp_min(5e-2))
            return v_t_pred

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