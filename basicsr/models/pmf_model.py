import torch
import time
import copy
import os
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from basicsr.metrics import calculate_metric
import torch.nn.functional as F
from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img, makeborder
import numpy as np
from torch.func import functional_call, jvp


'''
基于JiT方法的图像重建Model V2版.
探索以噪声为输入，lr为condition的重建方式。

重点注意：
（1）全部去掉了输入输出图像的正则化，例如 self.output = (self.output + 1.0) / 2.0
（2）feed_data方法中对输入LR图像进行了GT比例缩放


'''

@MODEL_REGISTRY.register()
class PixelMeanFlow_Model(SRModel):
    def __init__(self, opt):
        # 1. 初始化父类
        super(PixelMeanFlow_Model, self).__init__(opt)

        # 2. 扩散/流模型参数
        self.scale = opt['network_g'].get('upscale', 4)
        self.P_mean = opt.get('P_mean', -1.2)
        self.P_std = opt.get('P_std', 1.2)
        self.sampling_steps = opt.get('num_sampling_steps', 5)
        self.t_eps = 5e-2

        # 3. EMA 模型初始化
        if self.is_train:
            self.ema_decay = opt['train'].get('ema_decay', 0.9999)

            # 剥离 DDP 外壳
            if hasattr(self.net_g, 'module'):
                net_g_bare = self.net_g.module
            else:
                net_g_bare = self.net_g

            # 深拷贝
            self.net_g_ema = copy.deepcopy(net_g_bare)
            self.net_g_ema.eval()

            # 冻结梯度
            for p in self.net_g_ema.parameters():
                p.requires_grad = False

            # 放入设备
            self.net_g_ema = self.net_g_ema.to(self.device)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # 手动定义优化器
        optim_opt = train_opt['optim_g']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        self.optimizer_g = torch.optim.AdamW(
            optim_params,
            lr=optim_opt['lr'],
            weight_decay=train_opt.get('weight_decay', 0),
            betas=train_opt.get('betas', (0.9, 0.999))
        )

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

        # self.optimizers.append(self.optimizer_g)
        # # [关键修复] 必须显式初始化调度器，否则 optimizer 里没有 initial_lr-孙锦泱
        # self.setup_schedulers()
        self.setup_optimizers()
        self.setup_schedulers()


    @torch.no_grad()
    def update_ema(self):
        for p_ema, p_net in zip(self.net_g_ema.parameters(), self.net_g.parameters()):
            p_ema.data.mul_(self.ema_decay).add_(p_net.data, alpha=1 - self.ema_decay)
        for p_ema, p_net in zip(self.net_g_ema.buffers(), self.net_g.buffers()):
            p_ema.data.copy_(p_net.data)

    def feed_data(self, data):
        self.gt = data['gt'].to(self.device)# * 2.0 - 1.0
        self.lq = data['lq'].to(self.device)# * 2.0 - 1.0
        if self.lq.shape[-2:] != self.gt.shape[-2:]:
            self.lq_up = torch.nn.functional.interpolate(self.lq, self.gt.shape[-2:], mode='bicubic', align_corners=False)

    def sample_t_r(self):
        """
        Sample t and r.
        Strategy depends on config:
        - logit_normal: t ~ LogitNormal(loc, scale) with mixture of uniform
        - uniform: t ~ U[0, 1]
        """
        sampling_dist = self.opt['sampling_dist']
        if sampling_dist == 'logit_normal':
            loc = getattr(self.config, 'logit_normal_loc', 0.0)
            scale = getattr(self.config, 'logit_normal_scale', 0.8)
            uniform_prob = getattr(self.config, 'uniform_prob', 0.1)
            # TODO(pMF): 论文 Appendix A 写明“以 10% 概率均匀采样 (t,r) 以获得更平滑分布”，但未明确是否为三角域 0<=r<=t 的均匀；当前实现为 t 的 mixture + r|t ~ U[0,t]，需与官方实现核对。
            u = torch.rand(self.batch, device=self.device)
            # LogitNormal sampling for t
            t_ln = torch.sigmoid(torch.randn(self.batch, device=self.device) * scale + loc)
            # Uniform sampling for t
            t_unif = torch.rand(self.batch, device=self.device)

            mask = (u < uniform_prob).float()
            t = mask * t_unif + (1 - mask) * t_ln

        elif sampling_dist == 'uniform':
            t = torch.rand(self.batch, device=self.device)
        else:
            # Fallback to default
            t = torch.rand(self.batch, device=self.device)

        # r ~ U[0, t]
        r = torch.rand(self.batch, device=self.device) * t

        # Avoid numerical issues at t=0
        t = torch.clamp(t, min=1e-5)

        return t, r

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        gt = self.gt
        lq = self.lq_up
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        # lq = (lq - self.mean) * self.img_range
        self.batch = gt.size(0)
        '''
        JiT扩散过程，初始状态为LR，后期可考虑函数化。
        '''
        t, r = self.sample_t_r()
        t_b = t.view(self.batch, 1, 1, 1)
        r_b = r.view(self.batch, 1, 1, 1)
        # Noise
        e = lq
        z = (1 - t_b) * gt + t_b * e

        # Target velocity v
        # v = z' = -x + e = e - x
        v_target = e - gt

        # Functional wrapper for JVP
        params = dict(self.net_g.named_parameters())
        buffers = dict(self.net_g.named_buffers())
        params_and_buffers = {**params, **buffers}

        def net_call(z_arg, r_arg, t_arg):
            return functional_call(self.net_g, params_and_buffers, (z_arg, t_arg, r_arg))

        def u_fn(z_arg, r_arg, t_arg):
            x_pred_arg = net_call(z_arg, r_arg, t_arg)
            t_view = t_arg.view(-1, 1, 1, 1)
            return (z_arg - x_pred_arg) / t_view

        v = u_fn(z, t, t)
        func = lambda z_, r_, t_: u_fn(z_, r_, t_)
        primals = (z, r, t)
        tangents = (v, torch.zeros_like(r), torch.ones_like(t))
        u_out, dudt_out = jvp(func, primals, tangents)
        v_loss_target = v_target
        V = u_out + (t_b - r_b) * dudt_out.detach()
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(v_loss_target, V)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        x_pred = z - t_b * u_out
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(x_pred, gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            # if l_style is not None:
            #     l_total += l_style
            #     loss_dict['l_style'] = l_style
        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)
        self.update_ema()

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            test_model = self.net_g_ema
        else:
            self.net_g.eval()
            test_model = self.net_g

        with torch.no_grad():
            img = self.lq_up
            B, C, h, w = img.size()
            scale = self.scale

            net_opt = self.opt['network_g']
            hr_tile_h = net_opt.get('img_size', 256) * scale
            hr_tile_w = hr_tile_h
            lr_tile_h = hr_tile_h // scale
            lr_tile_w = hr_tile_w // scale

            pad_h = (lr_tile_h - h % lr_tile_h) % lr_tile_h
            pad_w = (lr_tile_w - w % lr_tile_w) % lr_tile_w

            img_padded = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
            _, _, H_pad, W_pad = img_padded.size()

            output_full = torch.zeros(B, C, H_pad * scale, W_pad * scale, device=self.device)

            for r in range(0, H_pad, lr_tile_h):
                for c in range(0, W_pad, lr_tile_w):
                    lq_tile = img_padded[:, :, r: r + lr_tile_h, c: c + lr_tile_w]
                    sr_tile = self.sample_image(lq_tile, num_step=self.sampling_steps, model=test_model)
                    if sr_tile.dim() == 3:
                        sr_tile = sr_tile.unsqueeze(0)
                    r_hr = r * scale
                    c_hr = c * scale
                    output_full[:, :, r_hr: r_hr + hr_tile_h, c_hr: c_hr + hr_tile_w] = sr_tile

            final_h = h * scale
            final_w = w * scale
            self.output = output_full[:, :, :final_h, :final_w]

        self.net_g.train()
        # self.output = (self.output + 1.0) / 2.0
        self.output = self.output.clamp(0, 1)


    @torch.no_grad()
    def sample_image(self, lq, num_step=5, model=None, direct=False):
        b = lq.shape[0]
        # t=1, r=0. We predict x(z_1, 0, 1).
        # z_1 is noise (z).
        t = torch.ones(b, device=self.device)
        r = torch.zeros(b, device=self.device)
        # x_pred = net(z, t, r, y)
        x_pred = self.net_g_ema(lq, t, r)
        return x_pred


    #     if model is None:
    #         model = self.net_g
    #     steps = num_step
    #     timesteps = torch.linspace(0.0, 1.0, steps + 1, device=self.device)
    #     b,c,h,w = lq.size()
    #     noise = torch.rand([b, c, h*self.scale, w*self.scale],device=self.device)# * self.noise_scale
    #     z = noise#lq:#[b,3,32,32], noise:[b,1,64,64]
    #     direct_sample_list = []
    #     for i in range(steps):
    #         t_curr = timesteps[i]
    #         t_next = timesteps[i + 1]
    #         t_curr = torch.full((b,), t_curr, device=self.device)
    #         t_next = torch.full((b,), t_next, device=self.device)
    #         if direct is False:
    #             z = self._euler_step(z=z, lq=lq, t=t_curr, t_next=t_next, model=model)
    #         else:
    #             direct_sample_list.append(self._direct_sample(z=z, lq=lq, t=t_curr, t_next=t_next, model=model))
    #             z = direct_sample_list[-1]
    #     return z.clamp(0, 1.0)
    #
    # @torch.no_grad()
    # def _euler_step(self, z, lq, t, t_next, model):
    #     x_pred = model(z, lq, t=t)
    #     v_next = (x_pred - z) / (1.0 - t).clamp_min(self.t_eps)
    #     z_next = z + (t_next - t) * v_next
    #     return z_next
    # @torch.no_grad()
    # def _direct_sample(self, z, lq, t, t_next, model):
    #     x_pred = model(z, lq, t=t)
    #     return x_pred






                                                                                     
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0 for metric in self.metric_results}

        pbar = tqdm(total=len(dataloader), unit='image') if use_pbar else None

        total_inference_time = 0.0
        total_images = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            self.test()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            total_inference_time += (end_time - start_time)
            total_images += val_data['lq'].size(0)

            visuals = self.get_current_visuals()
            sr_img = tensor2img(visuals['result'])
            gt_img = tensor2img(visuals['gt'])
            metric_data = {'img': sr_img, 'img2': gt_img}

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                    lq_img = tensor2img(torch.nn.functional.interpolate(self.lq_up, sr_img.shape[:2], mode='bicubic'))
                    sr_img = np.hstack((makeborder(lq_img), makeborder(sr_img), makeborder(gt_img)))
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}.png')

                os.makedirs(osp.dirname(save_img_path), exist_ok=True)
                imwrite(sr_img, save_img_path)
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            if pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if pbar:
            pbar.close()

        if total_images > 0:
            avg_time = total_inference_time / total_images
            logger = get_root_logger()
            logger.info(
                f'\n\t [Validation Speed] Total: {total_images}, Time: {total_inference_time:.2f}s, Avg: {avg_time:.4f}s \n')

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        if hasattr(self, 'net_g_ema'):
            self.save_network(self.net_g_ema, 'net_g_ema', current_iter)
        self.save_training_state(epoch, current_iter)

    # @torch.no_grad()
    # def _heun_step(self, z, t, t_next, noise, model):
    #     v_pred_t = model(z, noise, t)
    #     z_next_euler = z + (t_next - t) * v_pred_t
    #     v_pred_t_next = model(lq=z_next_euler, noise=noise, t=t_next)
    #
    #     v_pred = 0.5 * (v_pred_t + v_pred_t_next)
    #     z_next = z + (t_next - t) * v_pred
    #     return z_next