import torch
import torch.nn.functional as F
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img, split_with_overlap, merge_with_padding
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class MeanFlowModel(SRModel):
    """Mean Flow Model for Super Resolution."""

    def __init__(self, opt):
        super(MeanFlowModel, self).__init__(opt)
        self.sample_step = opt['val']['sample_step']#采样步数


    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()
        # define losses
        if train_opt.get('flow_opt'):
            self.flow_loss = build_loss(train_opt['flow_opt']).to(self.device)
        else:
            self.flow_loss = None
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.lq_bicubic = torch.nn.functional.interpolate(self.lq, size=self.gt.shape[-2:], mode='bicubic')

    '''
    define the interpolation processing of flow-based method.
    input: time step t, optional r
    '''
    def flow_interpolation(self, t, r=None):
        if self.gt is None:
            raise ValueError('GT is required for flow interpolation')

        h, w = self.gt.shape[-2:]
        # Upsample LQ to GT size
        lq_up = self.lq_bicubic
        batch_size = self.lq.shape[0]
        device = self.device
        # Sample time steps
        time_sampler = getattr(self, 'time_sampler', 'logit_normal')
        time_sigma = getattr(self, 'time_sigma', 1.0)
        time_mu = getattr(self, 'time_mu', 0.0)
        ratio_r_not_equal_t = getattr(self, 'ratio_r_not_equal_t', 0.5)

        if time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
        elif time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, 2, device=device)
            normal_samples = normal_samples * time_sigma + time_mu
            time_samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {time_sampler}")

        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r_samples, t_samples = sorted_samples[:, 0], sorted_samples[:, 1]
        fraction_equal = 1.0 - ratio_r_not_equal_t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal
        r_samples = torch.where(equal_mask, t_samples, r_samples)

        self.t = t_samples
        self.r = r_samples

        t_map = t_samples.view(batch_size, 1, 1, 1)
        alpha_t = 1 - t_map
        sigma_t = t_map
        d_alpha_t = -1
        d_sigma_t = 1
        self.x_0 = torch.randn_like(self.gt).to(self.device)
        self.xt = alpha_t * self.gt + sigma_t * self.x_0 #lq_up
        self.vt_tar = d_alpha_t * self.gt + d_sigma_t * self.x_0

        return self.xt

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.flow_process()
        t_map = self.t.view(-1, 1, 1, 1)
        r_map = self.r.view(-1, 1, 1, 1)
        # MeanFlow uses JVP for training
        jvp_args = (
            lambda xt, t, r: self.net_g(torch.cat([xt, self.lq_bicubic],dim=1), None, torch.cat([t_map, r_map], dim=1)),
            (self.xt, self.t, self.r),
            (self.vt_tar, torch.ones_like(self.t), torch.zeros_like(self.r)),
        )

        u, dudt = torch.autograd.functional.jvp(*jvp_args, create_graph=True)
        u_tgt = self.vt_tar - (t_map - r_map) * dudt

        error = u - u_tgt.detach()
        delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
        # gamma=0.5, c=1e-3 are default parameters
        w = 1.0 / (delta_sq + 1e-3).pow(0.5)
        l_flow = (w.detach() * delta_sq).mean()

        l_total = 0
        loss_dict = OrderedDict()

        # flow loss
        if self.flow_loss:
            l_flow_weighted = self.flow_loss(u, u_tgt.detach())
            l_total += l_flow_weighted
            loss_dict['l_flow'] = l_flow_weighted
        else:
            l_total += l_flow
            loss_dict['l_flow'] = l_flow

        # pixel loss
        if self.cri_pix:
            # Compute predicted data for pixel loss
            pred_data = self.xt + u * (1. - self.t.view(-1, 1, 1, 1))
            l_pix = self.cri_pix(pred_data, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            pred_data = self.xt + u * (1. - self.t.view(-1, 1, 1, 1))
            l_percep, l_style = self.cri_perceptual(pred_data, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    '''
    define the timestep sampling method.
    '''
    def sample_timestep(self):
        # For MeanFlow, time sampling is done in flow_interpolation
        # This method is kept for compatibility with flow_model interface
        return self.t

    '''
    main processing of flow.
    '''
    def flow_process(self):
        # For MeanFlow, flow_interpolation handles both time sampling and interpolation
        # We call it with None to let it sample time steps internally
        self.xt = self.flow_interpolation(None, None)

    '''
    sample image with flow-based ODE.
    '''
    def sample(self, lq_bicubic=None, sample_step=5):
        """Sample image using MeanFlow ODE solver"""
        # # Determine which model to use
        # if model is not None:
        #     model_to_use = model
        # elif ema and hasattr(self, 'net_g_ema'):
        #     model_to_use = self.net_g_ema
        # else:
        #     model_to_use = self.net_g

        # Get target size from scale

        # Upsample LQ to target size as initial state (t=1)
        x_0 = torch.randn_like(lq_bicubic).to(self.device)
        z = x_0
        # MeanFlow sampling loop
        batch_size = z.shape[0]
        device = self.device
        # Generate time steps: [1.0, 0.9, ..., 0.0]
        time_steps = torch.linspace(1.0, 0.0, sample_step + 1, device=device)

        for i in range(sample_step):
            t_cur = time_steps[i]
            t_next = time_steps[i + 1]

            # Construct time tensors [B]
            t_tensor = torch.full((batch_size,), t_cur, device=device)
            r_tensor = torch.full((batch_size,), t_cur, device=device)  # r = t for sampling

            # Call model to predict velocity v
            with torch.no_grad():
                v = self.net_g(torch.cat([z, lq_bicubic], dim=1), torch.cat([t_tensor, r_tensor], dim=1))

            # Euler update: z_{next} = z_{curr} - dt * v
            dt = t_cur - t_next
            z = z - dt * v

        return z

    '''
    Add flow-based loss function.
    '''


    @torch.no_grad()
    def test(self):
        # 1. 确定使用哪一个模型权重进行测试
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            test_model = self.net_g_ema
        else:
            self.net_g.eval()
            test_model = self.net_g

        img = self.lq_bicubic
        split_h = self.opt['network_g']['img_size']
        split_w = self.opt['network_g']['img_size']
        split_img, padding_info = split_with_overlap(img, split_h, split_w, split_h, split_w, return_padding=True)
        [_, n_h, n_w, _, _, _] = split_img.shape
        for i in range(n_h):
            for j in range(n_w):
                split_img[:, i, j, ...] = self.sample(split_img[:, i, j, ...], self.sample_step)
        merged_img = merge_with_padding(split_img, padding_info)

        self.net_g.train()
        self.output = merged_img.clamp(0, 1)

