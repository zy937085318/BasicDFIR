import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp
try:
    from lpips import LPIPS
except Exception:
    LPIPS = None

class PixelMeanFlow(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        LPIPS = None
        if LPIPS is None:
            self.lpips = None
        else:
            self.lpips = LPIPS(net='vgg').eval()
            for p in self.lpips.parameters():
                p.requires_grad = False
        # TODO(pMF): 论文 Appendix A 讨论 ConvNeXt-V2 变体的 perceptual loss；当前仅实现 VGG-LPIPS，需要对齐引用实现后再补全。

    def _maybe_move_lpips(self, device):
        if self.lpips is None:
            return
        try:
            p = next(self.lpips.parameters())
        except StopIteration:
            return
        if p.device != device:
            self.lpips.to(device)

    def _random_crop_and_resize_224(self, x):
        if x.ndim != 4:
            raise ValueError("Expected x with shape (N, C, H, W).")
        n, c, h, w = x.shape
        if h == 224 and w == 224:
            return x
        crop_size = min(224, h, w)
        if crop_size <= 0:
            raise ValueError("Invalid spatial size for perceptual crop.")
        if h == crop_size:
            top = torch.zeros(n, device=x.device, dtype=torch.long)
        else:
            top = torch.randint(0, h - crop_size + 1, (n,), device=x.device)
        if w == crop_size:
            left = torch.zeros(n, device=x.device, dtype=torch.long)
        else:
            left = torch.randint(0, w - crop_size + 1, (n,), device=x.device)
        crops = []
        for i in range(n):
            crops.append(x[i:i+1, :, top[i]:top[i] + crop_size, left[i]:left[i] + crop_size])
        x_crop = torch.cat(crops, dim=0)
        if x_crop.shape[-2:] != (224, 224):
            x_crop = F.interpolate(x_crop, size=(224, 224), mode='bilinear', align_corners=False)
        return x_crop

    def sample_t_r(self, batch_size, device):
        """
        Sample t and r.
        Strategy depends on config:
        - logit_normal: t ~ LogitNormal(loc, scale) with mixture of uniform
        - uniform: t ~ U[0, 1]
        """
        sampling_dist = getattr(self.config, 'sampling_dist', 'logit_normal')

        if sampling_dist == 'logit_normal':
            loc = getattr(self.config, 'logit_normal_loc', 0.0)
            scale = getattr(self.config, 'logit_normal_scale', 0.8)
            uniform_prob = getattr(self.config, 'uniform_prob', 0.1)
            # TODO(pMF): 论文 Appendix A 写明“以 10% 概率均匀采样 (t,r) 以获得更平滑分布”，但未明确是否为三角域 0<=r<=t 的均匀；当前实现为 t 的 mixture + r|t ~ U[0,t]，需与官方实现核对。

            u = torch.rand(batch_size, device=device)
            # LogitNormal sampling for t
            t_ln = torch.sigmoid(torch.randn(batch_size, device=device) * scale + loc)
            # Uniform sampling for t
            t_unif = torch.rand(batch_size, device=device)

            mask = (u < uniform_prob).float()
            t = mask * t_unif + (1 - mask) * t_ln

        elif sampling_dist == 'uniform':
            t = torch.rand(batch_size, device=device)
        else:
            # Fallback to default
            t = torch.rand(batch_size, device=device)

        # r ~ U[0, t]
        r = torch.rand(batch_size, device=device) * t

        # Avoid numerical issues at t=0
        t = torch.clamp(t, min=1e-5)

        return t, r

    def sample_t_r_cfg(self, batch_size, device):
        t, r = self.sample_t_r(batch_size, device)
        cfg_scale_min = getattr(self.config, "cfg_scale_min", 1.0)
        cfg_scale_max = getattr(self.config, "cfg_scale_max", 7.0)
        if cfg_scale_min <= 0:
            raise ValueError("cfg_scale_min must be > 0.")
        if cfg_scale_max < cfg_scale_min:
            raise ValueError("cfg_scale_max must be >= cfg_scale_min.")
        w = torch.empty(batch_size, device=device).uniform_(cfg_scale_min, cfg_scale_max)
        # TODO(pMF): 论文 Appendix A 表述“condition on CFG scale interval”，但未公开 interval 训练时的采样分布；此处用三角区域均匀采样作为占位实现，需与 iMF 官方实现对齐。
        a = torch.rand(batch_size, device=device)
        b = a + torch.rand(batch_size, device=device) * (1 - a)
        cfg_interval = torch.stack([a, b], dim=1)
        return t, r, w, cfg_interval

    def forward_loss(self, x, y):
        """
        x: lq
        y: gt
        """
        b, c, h, w = x.shape
        device = x.device
        y_in=None

        use_cfg_training = False
        t, r = self.sample_t_r(b, device)
        w_scale = None
        cfg_interval = None

        # Add dimensions for broadcasting
        t_b = t.view(b, 1, 1, 1)
        r_b = r.view(b, 1, 1, 1)

        # Noise
        e = x #torch.randn_like(x) #lq
        z = (1 - t_b) * y + t_b * e #y->gt, which means z=gt while t=0

        # Target velocity v
        # v = z' = -x + e = e - x
        v_target = e - y

        # Functional wrapper for JVP
        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        params_and_buffers = {**params, **buffers}

        def net_call(z_arg, t_arg, r_arg, y_arg, w_arg, interval_arg):
            return functional_call(self.model, params_and_buffers, (z_arg, t_arg, r_arg, y_arg, w_arg, interval_arg))

        def u_fn(z_arg, t_arg, r_arg, y_arg, w_arg, interval_arg):
            x_pred_arg = net_call(z_arg, t_arg, r_arg, y_arg, w_arg, interval_arg)
            t_view = t_arg.view(-1, 1, 1, 1)
            return (z_arg - x_pred_arg) / t_view


        v = u_fn(z, t, t, y_in, w_scale, cfg_interval)
        func = lambda z_, t_, r_: u_fn(z_, t_, r_, y_in, w_scale, cfg_interval)
        primals = (z, t, r)
        tangents = (v, torch.ones_like(t), torch.zeros_like(r))
        u_out, dudt_out = jvp(func, primals, tangents)#u dudt

        # V = u + (t - r) * stop_grad(dudt)
        V = v_target - (t_b - r_b) * dudt_out#v_hat in meanflow git code

        # Loss
        loss_pmf = F.mse_loss(V.detach(), u_out) #v_loss_target)

        # Perceptual Loss
        x_pred = z - (t_b-0) * u_out

        loss_perc = torch.tensor(0.0, device=device)

        if self.config.lambda_perc > 0:
            if self.lpips is None:
                raise RuntimeError("lambda_perc > 0 but lpips is not installed.")
            mask_perc = (t <= self.config.perc_threshold)
            if mask_perc.any():
                x_sub = x[mask_perc]
                x_pred_sub = x_pred[mask_perc]

                self._maybe_move_lpips(device)
                x_sub_224 = self._random_crop_and_resize_224(x_sub)
                x_pred_sub_224 = self._random_crop_and_resize_224(x_pred_sub)

                l_perc = self.lpips(x_pred_sub_224, x_sub_224).mean()
                loss_perc = l_perc

        total_loss = loss_pmf + self.config.lambda_perc * loss_perc

        return total_loss, {
            "loss_pmf": loss_pmf.item(),
            "loss_perc": loss_perc.item() if isinstance(loss_perc, torch.Tensor) else loss_perc,
            "loss_total": total_loss.item(),
        }, x_pred

    @torch.no_grad()
    def sample(self, z, y=None, cfg_scale=1.0, cfg_interval=None):
        """
        One-step generation.
        z: (N, C, H, W) noise
        y: (N,) class labels
        """
        b = z.shape[0]
        device = z.device

        # t=1, r=0. We predict x(z_1, 0, 1).
        # z_1 is noise (z).
        t = torch.ones(b, device=device)
        r = torch.zeros(b, device=device)

        # Unconditional input
        # Note: We assume config.num_classes is correct. Training uses (num_classes - 1) as null.
        y_null = None #torch.full_like(y, self.config.num_classes - 1)

        # Forward pass
        # We need x_pred.
        # x_pred = net(z, t, r, y)

        x_out = self.model(z, t, r, y=None, w=None, cfg_interval=cfg_interval)

        return x_out
