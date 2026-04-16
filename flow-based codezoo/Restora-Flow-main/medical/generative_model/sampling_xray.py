# -----------------------------------------------------------------------------
# This file includes components adapted from the following open-source projects:
#
# - GaussianDiffusion implementation based on:
#       https://github.com/mobaidoctor/med-ddpm
#
# - RePaint implementation adapted from:
#       https://github.com/andreas128/RePaint
#
# - DDNM implementation adapted from:
#       https://github.com/wyhuai/DDNM
#
# - PnP-Flow, D-Flow, OT-ODE, Flow-Priors adapted from:
#       https://github.com/annegnx/PnP-Flow
#
# Modifications were made to integrate these methods into a unified framework.
# -----------------------------------------------------------------------------

import os
import warnings
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint
from tqdm import tqdm

from generative_model.scheduler import get_schedule_jump
from utils import image_metrics
from utils import occlusion_simulation


warnings.filterwarnings("ignore", category=UserWarning)


try:
    from apex import amp

    APEX_AVAILABLE = True
    print("APEX: ON")
except:
    APEX_AVAILABLE = False
    print("APEX: OFF")


def exists(x):
    return x is not None


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


def linear_beta_schedule(timesteps):
    # Linear schedule from Ho et al, extended to work for any number of
    # diffusion steps.
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(
        beta_start, beta_end, timesteps, dtype=np.float64
    )


def save_image(img, output_folder, filename):
    from PIL import Image
    png_image_path = os.path.join(output_folder, filename)
    img_uint8 = (((img - img.min()) /
                  (img.max() - img.min())) * 255.9).astype(np.uint8)
    img = Image.fromarray(img_uint8)
    img.save(png_image_path)


def compute_metrics(gt_img, degraded_img, restored_img, problem, H_adj, image_id):
    psnr_rec, psnr_noisy = image_metrics.compute_psnr(
        gt_img[:, 0].unsqueeze(0),
        degraded_img[:, 0].unsqueeze(0),
        restored_img[:, 0].unsqueeze(0),
        problem, H_adj
    )
    print(f"image_id {image_id}: psnr_rec={psnr_rec}, psnr_noisy={psnr_noisy}")

    ssim_rec, ssim_noisy = image_metrics.compute_ssim(
        gt_img[:, 0].unsqueeze(0),
        degraded_img[:, 0].unsqueeze(0),
        restored_img[:, 0].unsqueeze(0),
        problem, H_adj
    )
    print(f"image_id {image_id}: ssim_rec={ssim_rec}, ssim_noisy={ssim_noisy}")

    lpips_rec, lpips_noisy = image_metrics.compute_lpips(
        gt_img[:, 0].unsqueeze(0),
        degraded_img[:, 0].unsqueeze(0),
        restored_img[:, 0].unsqueeze(0),
        problem, H_adj
    )
    print(f"image_id {image_id}: lpips_rec={lpips_rec}, lpips_noisy={lpips_noisy}")

    return psnr_rec, ssim_rec, lpips_rec


class Cnf(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def model_forward(self, x, t):
        return self.model(x, t)

    def forward(self, t, x):
        with torch.no_grad():
            z = self.model_forward(x, t.repeat(x.shape[0]))
        return z


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            *,
            image_size,
            channels=1,
            timesteps=1000,
            betas=None,
            schedule='cosine'
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.device = "cuda"

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            if schedule == 'cosine':
                betas = cosine_beta_schedule(timesteps)
            else:
                betas = linear_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.to_torch = to_torch
        self.alphas = to_torch(alphas)
        self.sqrt_alphas = to_torch(np.sqrt(alphas))
        self.sqrt_one_minus_alphas = to_torch(np.sqrt(1. - alphas))

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef3', to_torch(
            1. - (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(
            self,
            x_t,
            t,
            noise
    ):
        x_hat = 0.
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise -
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_hat
        )

    def q_posterior(
            self,
            x_start,
            x_t,
            t
    ):
        x_hat = 0.
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t +
                extract(self.posterior_mean_coef3, t, x_t.shape) * x_hat
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self,
            x,
            t,
            clip_denoised: bool
    ):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    def undo(
            self,
            image_before_step,
            img_after_model,
            est_x_0,
            t,
            debug=False
    ):
        return self._undo(img_after_model, t)

    def _undo(
            self,
            img_out,
            t
    ):
        # beta = _extract_into_tensor(self.betas, t, img_out.shape)
        beta = extract(self.betas, t, img_out.shape)
        img_in_est = torch.sqrt(1 - beta) * img_out + torch.sqrt(beta) * torch.randn_like(img_out)

        return img_in_est

    @torch.no_grad()
    def p_sample(
            self,
            x,
            t,
            clip_denoised=True,
            repeat_noise=False
    ):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, _ = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))  # no noise when t == 0
        sample = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        return sample

    @torch.no_grad()
    def p_sample_loop(
            self,
            shape
    ):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    @torch.no_grad()
    def sample(
            self,
            batch_size=1
    ):
        return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size))

    def p_sample_repaint(
            self,
            x0,
            mask,
            x,
            t,
            clip_denoised=True,
            pred_xstart=None,
            output_folder=None
    ):
        noise = torch.randn_like(x)
        gt = x0

        if pred_xstart is not None:
            gt_keep_mask = mask
            alpha_cumprod = extract(self.alphas_cumprod, t, x.shape)
            gt_weight = torch.sqrt(alpha_cumprod)
            gt_part = gt_weight * gt

            noise_weight = torch.sqrt((1 - alpha_cumprod))
            noise_part = noise_weight * torch.randn_like(x)

            weighed_gt = gt_part + noise_part

            x = gt_keep_mask * weighed_gt + (1 - gt_keep_mask) * x

            if output_folder is not None:
                known_numpy = (gt_keep_mask * weighed_gt).cpu().numpy().squeeze(0)
                save_image(known_numpy[0], output_folder, 'known.png')
                unknown_numpy = ((1 - gt_keep_mask) * x).cpu().numpy().squeeze(0)
                save_image(unknown_numpy[0], output_folder, 'unknown.png')

        model_mean, model_variance, model_log_variance, pred_xstart = self.p_mean_variance(x=x, t=t,
                                                                                           clip_denoised=clip_denoised)
        out = {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )

        sample = out["mean"] + nonzero_mask * \
                 torch.exp(0.5 * out["log_variance"]) * noise

        result = {
            "sample": sample,
            "pred_xstart": out["pred_xstart"],
            "gt": gt,
        }

        return result

    def p_sample_loop_progressive_repaint(
            self,
            x0,
            mask,
            shape,
            params,
            noise=None,
            clip_denoised=True,
            device=None,
            progress=False,
            output_folder=None
    ):
        if device is None:
            device = next(self.denoise_fn.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            image_after_step = noise
        else:
            image_after_step = torch.randn(*shape, device=device)

        timesteps, jump_length, jump_n_sample = params

        pred_xstart = None
        idx_wall = -1
        sample_idxs = defaultdict(lambda: 0)
        n_sample = shape[0]

        times = get_schedule_jump(
            t_T=timesteps,
            n_sample=n_sample,
            jump_length=jump_length,
            jump_n_sample=jump_n_sample
        )

        time_pairs = list(zip(times[:-1], times[1:]))
        if progress:
            from tqdm.auto import tqdm
            time_pairs = tqdm(time_pairs)

        out = None
        for t_last, t_cur in time_pairs:
            idx_wall += 1
            t_last_t = torch.tensor([t_last] * shape[0], device=device)

            if t_cur < t_last:
                with torch.no_grad():
                    image_before_step = image_after_step.clone()
                    out = self.p_sample_repaint(x0, mask, image_after_step, t_last_t,
                                                output_folder=output_folder,
                                                clip_denoised=clip_denoised,
                                                pred_xstart=pred_xstart
                                                )
                    image_after_step = out["sample"]
                    pred_xstart = out["pred_xstart"]
                    sample_idxs[t_cur] += 1
            else:
                # t_shift = conf.get('inpa_inj_time_shift', 1)
                t_shift = 1
                image_before_step = image_after_step.clone()
                image_after_step = self.undo(image_before_step, image_after_step, est_x_0=out['pred_xstart'],
                                             t=t_last_t + t_shift, debug=False)
                pred_xstart = out["pred_xstart"]

        return out["sample"]

    def simplified_ddnm_plus(
            self,
            y,
            A,
            Ap,
            sigma_y,
            jump_length=1,
            jump_n_sample=1,
            sampling_steps=50
    ):
        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
            return a

        eta = 0.85  # as in DDNM paper
        sigma_y = 2 * sigma_y  # to account for scaling to [-1, 1]

        x = torch.randn(y.shape[0], self.channels, self.image_size, self.image_size, device="cuda")

        with torch.no_grad():
            skip = self.num_timesteps // sampling_steps
            n = x.size(0)
            x0_preds = []
            xs = [x]

            times = get_schedule_jump(
                t_T=sampling_steps,  # DDIM sampling steps
                n_sample=1,
                jump_length=jump_length,
                jump_n_sample=jump_n_sample
            )

            time_pairs = list(zip(times[:-1], times[1:]))
            from tqdm.auto import tqdm
            time_pairs = tqdm(time_pairs)

            # reverse diffusion sampling
            for i, j in time_pairs:
                i, j = i * skip, j * skip
                if j < 0: j = -1

                if j < i:  # normal sampling
                    t = (torch.ones(n) * i).to(x.device)
                    next_t = (torch.ones(n) * j).to(x.device)
                    at = compute_alpha(self.betas, t.long())
                    at_next = compute_alpha(self.betas, next_t.long())
                    sigma_t = (1 - at_next ** 2).sqrt()
                    xt = xs[-1].to('cuda')

                    et = self.denoise_fn(xt, t)

                    if et.size(1) == 6:
                        et = et[:, :3]

                    # Eq. 12
                    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                    # Eq. 19 (batch size = 1)
                    '''if sigma_t >= at_next * sigma_y:
                        lambda_t = 1.
                        gamma_t = (sigma_t ** 2 - (at_next * sigma_y) ** 2).sqrt()
                    else:
                        lambda_t = (sigma_t) / (at_next * sigma_y)
                        gamma_t = 0.'''

                    # Eq. 19 for batch size > 1
                    at_next_sigma_y = at_next * sigma_y
                    lambda_t = torch.where(sigma_t >= at_next_sigma_y, torch.ones_like(sigma_t),
                                           sigma_t / at_next_sigma_y)

                    gamma_t = torch.where(sigma_t >= at_next_sigma_y,
                                          torch.sqrt(sigma_t ** 2 - at_next_sigma_y ** 2),
                                          torch.zeros_like(sigma_t))

                    # Eq. 17
                    x0_t_hat = x0_t - lambda_t * Ap(A(x0_t) - y)
                    c1 = (1 - at_next).sqrt() * eta
                    c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

                    xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * et)

                    x0_preds.append(x0_t.to('cpu'))
                    xs.append(xt_next.to('cpu'))
                else:  # time-travel back
                    next_t = (torch.ones(n) * j).to(x.device)
                    at_next = compute_alpha(self.betas, next_t.long())
                    x0_t = x0_preds[-1].to('cuda')

                    xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                    xs.append(xt_next.to('cpu'))

            x = xs[-1]

        return x

    def solve_ip_ddnm(
            self,
            test_dataset,
            problem: str,
            degradation,
            sigma_noise: float,
            params: list,
            output_folder: str = None
    ):
        H, H_adj = degradation.H, degradation.H_adj
        psnrs, ssims, lpips = [], [], []

        for counter, item in enumerate(test_dataset):
            gt_img, img_id, landmarks = item
            gt_img = gt_img.unsqueeze(0).expand(1, -1, -1, -1)
            save_image(gt_img.cpu().numpy()[0][0], output_folder, f'{img_id}_clean_img.png')

            degraded_img = H(gt_img.clone().to(self.device))
            torch.manual_seed(img_id)
            degraded_img += torch.randn_like(degraded_img) * sigma_noise
            degraded_img, gt_img = degraded_img.to(self.device), gt_img.to('cpu')

            # Initialize the image with the adjoint operator
            x = H_adj(torch.ones_like(degraded_img)).to(self.device)

            if problem == 'occlusion_removal':
                occluded_image, occlusion_mask = occlusion_simulation.occlusion_simulation.simulate(degraded_img, landmarks[:, 1:])
                degraded_img[:, 0, :, :] = occluded_image.clone()

                if torch.all(occlusion_mask == 0):
                    # skip the mage if  no occlusions were added
                    continue

            save_image(degraded_img.cpu().numpy()[0][0], output_folder, f'{img_id}_input_img.png')

            with torch.no_grad():
                output = self.simplified_ddnm_plus(y=degraded_img, A=H, Ap=H_adj,
                                                    sigma_y=sigma_noise,
                                                    sampling_steps=params[0],
                                                    jump_length=params[1],
                                                    jump_n_sample=params[2])

            # Save results
            restored_img = output.detach().clone()
            save_image(restored_img.cpu().numpy()[0][0], output_folder, f'{img_id}_reconstructed.png')

            # Compute metrics
            psnr_rec, ssim_rec, lpips_rec = compute_metrics(gt_img, degraded_img, restored_img, problem, H_adj, img_id)
            psnrs.append(psnr_rec)
            ssims.append(ssim_rec)
            lpips.append(lpips_rec)

        return psnrs, ssims, lpips

    def solve_ip_repaint(
            self,
            test_dataset,
            problem: str,
            degradation,
            sigma_noise: float,
            params: list,
            output_folder: str = None
    ):
        H, H_adj = degradation.H, degradation.H_adj
        psnrs, ssims, lpips = [], [], []

        # Main loop
        for counter, item in enumerate(test_dataset):
            gt_img, img_id, landmarks = item
            gt_img = gt_img.unsqueeze(0).expand(1, -1, -1, -1)
            save_image(gt_img.cpu().numpy()[0][0], output_folder, f'{img_id}_gt_img.png')

            degraded_img = H(gt_img.clone().to(self.device))
            torch.manual_seed(img_id)
            degraded_img += torch.randn_like(degraded_img) * sigma_noise
            degraded_img, gt_img = degraded_img.to(self.device), gt_img.to('cpu')

            # Create a mask
            x = H_adj(torch.ones_like(degraded_img)).to(self.device)

            if problem == 'occlusion_removal':
                occluded_image, occlusion_mask = occlusion_simulation.simulate(degraded_img, landmarks[:, 1:])
                degraded_img[:, 0, :, :] = occluded_image.clone()

                if torch.all(occlusion_mask == 0):
                    # skip images where no occlusions were added
                    continue

            save_image(degraded_img.cpu().numpy()[0][0], output_folder, f'{img_id}_input_img.png')

            # Iterative reconstruction
            with torch.no_grad():
                if problem == 'superresolution':
                    # input for superresolution with RePaint
                    superresolution_input = gt_img.to(self.device) * x + torch.randn_like(
                        gt_img.to(self.device)) * sigma_noise
                    output = self.p_sample_loop_progressive_repaint(x0=superresolution_input, mask=x,
                                                                    shape=gt_img.shape, params=params)
                else:
                    output = self.p_sample_loop_progressive_repaint(x0=degraded_img, mask=x, shape=degraded_img.shape,
                                                                    params=params)

            # Save results
            restored_img = output.detach().clone()
            save_image(restored_img.cpu().numpy()[0][0], output_folder, f'{img_id}_reconstructed.png')

            # Compute metrics
            psnr_rec, ssim_rec, lpips_rec = compute_metrics(gt_img, degraded_img, restored_img, problem, H_adj, img_id)
            psnrs.append(psnr_rec)
            ssims.append(ssim_rec)
            lpips.append(lpips_rec)

        return psnrs, ssims, lpips


class Flow(nn.Module):
    def __init__(
            self,
            denoise_fn,
            *,
            image_size,
            channels=1,
            ode_steps=32,
            device='cuda'
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.ode_steps = ode_steps
        self.device = device

        # DDPM buffers: not required when training flow models from scratch, but must be initialized (e.g., with dummy
        # values) to load pretrained flow models used in the paper. Without them, checkpoint loading will fail due to
        # missing keys.
        self.register_buffer('betas', torch.tensor(0))
        self.register_buffer('alphas_cumprod', torch.tensor(0))
        self.register_buffer('alphas_cumprod_prev', torch.tensor(0))
        self.register_buffer('sqrt_alphas_cumprod', torch.tensor(0))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.tensor(0))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.tensor(0))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.tensor(0))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.tensor(0))
        self.register_buffer('posterior_variance', torch.tensor(0))
        self.register_buffer('posterior_log_variance_clipped', torch.tensor(0))
        self.register_buffer('posterior_mean_coef1', torch.tensor(0))
        self.register_buffer('posterior_mean_coef2', torch.tensor(0))
        self.register_buffer('posterior_mean_coef3', torch.tensor(0))

    @torch.no_grad()
    def p_sample_loop(
            self,
            shape
    ):
        def func_conditional(t, x):
            t_curr = torch.ones(x.shape[0], device=self.device) * t
            x_ipt = (1 - (1 - sigma_min) * t) * x0 + t * x  # conditional flow, eq (22)
            return self.denoise_fn(x_ipt, t_curr)

        sigma_min = 1e-5
        x0 = torch.randn(shape, device=self.device)
        torch_linspace = torch.linspace(0, 1, self.ode_steps, device=self.device)

        traj = odeint(
            func_conditional,
            y0=x0,
            t=torch_linspace,
            method='euler',
            atol=1e-5,
            rtol=1e-5
        )

        return traj[-1]

    @torch.no_grad()
    def sample(
            self,
            batch_size=1
    ):
        return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size))

    def sample_restora_flow_denoising(
            self,
            input_img,
            sigma_noise,
            ode_steps
    ):
        x = torch.randn_like(input_img, device=self.device)  # initialize x with noise
        x_obs = input_img * (1 - sigma_noise)

        torch_linspace = torch.linspace(0, 1, int(ode_steps), device=self.device)
        delta_t = 1 / len(torch_linspace)

        for t in torch_linspace:
            mask = torch.ones(input_img.shape, device=self.device)

            if t < (1 - sigma_noise):
                x = mask * x_obs + (1 - mask) * x
            else:
                x = x + delta_t * self.denoise_fn(x, torch.tensor(t, device=self.device).repeat(x.shape[0]))

        return x

    def sample_restora_flow_mask_guided(
            self,
            input_img,
            mask,
            ode_steps,
            correction_steps,
            progress=False,
            output_folder=None,
            sample_id=0
    ):
        if output_folder is not None:
            save_image(input_img.cpu().numpy()[0][0], output_folder, f'{sample_id}_input_img.png')
            save_image(mask.cpu().numpy()[0][0], output_folder, f'{sample_id}_mask.png')

        batch_size = input_img.shape[0]
        x = torch.randn_like(input_img, device=self.device)  # initialize x with noise
        pred_x_start = None

        times = get_schedule_jump(
            t_T=ode_steps,
            n_sample=1,
            jump_length=1,
            jump_n_sample=correction_steps + 1
        )

        # normalize
        times = [((x - min(times)) / (max(times) - min(times))) for x in times]
        times.reverse()
        time_pairs = list(zip(times[:-1], times[1:]))

        if progress:
            from tqdm.auto import tqdm
            time_pairs = tqdm(time_pairs)

        for t_last, t_cur in time_pairs:
            t_last_t = torch.tensor([t_last] * batch_size, device=self.device).view(batch_size, 1, 1, 1)
            t_cur_t = torch.tensor([t_cur] * batch_size, device=self.device).view(batch_size, 1, 1, 1)

            if t_last < t_cur:
                with torch.no_grad():
                    if pred_x_start is not None:
                        # mask-based update
                        eps = torch.randn_like(x)
                        z_prim = t_last_t * input_img + (1 - t_last_t) * eps
                        x = mask * z_prim + (1 - mask) * x

                        if output_folder is not None:
                            known = (mask * z_prim)
                            save_image(known.cpu().numpy()[0][0], output_folder, f'{sample_id}_known.png')

                            unknown = (1 - mask) * x
                            save_image(unknown.cpu().numpy()[0][0], output_folder, f'{sample_id}_unknown.png')

                    # flow update
                    delta_t = t_cur_t - t_last_t    # equivalent to 1 / ode_steps

                    x = x + delta_t * self.denoise_fn(x, torch.tensor(t_last, device=self.device).repeat(batch_size))
                    out_sample = x.clone()

                    if output_folder is not None:
                        save_image(out_sample.cpu().numpy()[0][0], output_folder, f'out_sample.png')

                    pred_x_start = True
            else:
                # trajectory correction
                x_1_prim = x + (1 - t_last_t) * self.denoise_fn(x, torch.tensor(t_last, device=self.device).repeat(batch_size))
                x = t_cur_t * x_1_prim + (1 - t_cur_t) * torch.randn_like(x)

        return out_sample

    def solve_ip_restora_flow(
            self,
            test_dataset,
            problem: str,
            degradation,
            sigma_noise: float,
            params: list,
            output_folder: str = None
    ):
        ode_steps, correction_steps = params

        H, H_adj = degradation.H, degradation.H_adj
        psnrs, ssims, lpips = [], [], []

        # Main loop
        for item in test_dataset:
            gt_img, img_id, landmarks = item
            gt_img = gt_img.unsqueeze(0).expand(1, -1, -1, -1)
            save_image(gt_img.cpu().numpy()[0][0], output_folder, f'{img_id}_gt_img.png')

            degraded_img = H(gt_img.clone().to(self.device))
            torch.manual_seed(img_id)
            degraded_img += torch.randn_like(degraded_img) * sigma_noise
            degraded_img, gt_img = degraded_img.to(self.device), gt_img.to('cpu')

            # Create a mask
            x = H_adj(torch.ones_like(degraded_img)).to(self.device)

            if problem == 'occlusion_removal':
                occluded_image, occlusions_mask = occlusion_simulation.simulate(degraded_img, landmarks[:, 1:])
                degraded_img[:, 0, :, :] = occluded_image.clone()

                if torch.all(occlusions_mask == 0):
                    # skip the image if no occlusions were added
                    continue

            save_image(degraded_img.cpu().numpy()[0][0], output_folder, f'{img_id}_input_img.png')

            # Iterative reconstruction
            with torch.no_grad():
                if problem == 'denoising':
                    output = self.sample_restora_flow_denoising(input_img=degraded_img,
                                                                sigma_noise=sigma_noise,
                                                                ode_steps=ode_steps)
                elif problem == 'superresolution':
                    # input for superresolution with Restora-Flow
                    superresolution_input = gt_img.to(self.device) * x + torch.randn_like(
                        gt_img.to(self.device)) * sigma_noise
                    # save_image(superresolution_input.cpu().numpy()[0][0], output_folder, 'superresolution_input.png')
                    output = self.sample_restora_flow_mask_guided(input_img=superresolution_input,
                                                                  mask=x,
                                                                  ode_steps=ode_steps,
                                                                  correction_steps=correction_steps,
                                                                  sample_id=img_id)
                else:  # inpainting
                    output = self.sample_restora_flow_mask_guided(input_img=degraded_img,
                                                                  mask=x,
                                                                  ode_steps=ode_steps,
                                                                  correction_steps=correction_steps,
                                                                  sample_id=img_id)

            # Save results
            restored_img = output.detach().clone()
            save_image(restored_img.cpu().numpy()[0][0], output_folder, f'{img_id}_reconstructed.png')

            # Compute metrics
            psnr_rec, ssim_rec, lpips_rec = compute_metrics(gt_img, degraded_img, restored_img, problem, H_adj, img_id)
            psnrs.append(psnr_rec)
            ssims.append(ssim_rec)
            lpips.append(lpips_rec)

        return psnrs, ssims, lpips

    def solve_ip_ot_ode(
            self,
            test_dataset,
            problem: str,
            degradation,
            sigma_noise: float,
            params: list,
            output_folder: str = None
    ):
        def initialization(noisy_img, t0):
            return t0 * noisy_img + (1 - t0) * torch.randn_like(noisy_img)

        # initialization
        steps_ode, start_time, gamma = params
        steps, delta = steps_ode, 1 / steps_ode

        H, H_adj = degradation.H, degradation.H_adj
        psnrs, ssims, lpips = [], [], []

        # iterate over images in test set
        for item in test_dataset:
            gt_img, img_id, landmarks = item
            gt_img = gt_img.unsqueeze(0).expand(1, -1, -1, -1)
            save_image(gt_img.cpu().numpy()[0][0], output_folder, f'{img_id}_gt_img.png')

            degraded_img = H(gt_img.clone().to(self.device))
            torch.manual_seed(img_id)
            degraded_img += torch.randn_like(degraded_img) * sigma_noise
            degraded_img, gt_img = degraded_img.to(self.device), gt_img.to('cpu')

            # initialize the image with the adjoint operator
            # x = H_adj(noisy_img.clone()).to(self.device)
            x = initialization(H_adj(degraded_img.clone()), start_time)

            if problem == 'occlusion_removal':
                occluded_image, occlusion_mask = occlusion_simulation.simulate(degraded_img, landmarks[:, 1:])
                degraded_img[:, 0, :, :] = occluded_image.clone()

                if torch.all(occlusion_mask == 0):
                    # skip the image if no occlusions were added
                    continue

            save_image(degraded_img.cpu().numpy()[0][0], output_folder, f'{img_id}_input_img.png')

            for iteration in range(int(steps * start_time), int(steps)):
                with ((torch.no_grad())):
                    t1 = torch.ones(len(x), device=self.device) * delta * iteration
                    vt = self.denoise_fn(x, t1)
                    rt_squared = ((1-t1)**2 / ((1-t1)**2 + t1**2)).view(-1, 1, 1, 1)
                    x1_hat = x + (1-t1.view(-1, 1, 1, 1)) * vt

                    # solve linear problem Cx=d
                    d = degraded_img - H(x1_hat)
                    sol = torch.zeros_like(d)

                    for i in range(d.shape[0]):
                        sol_tmp = 1 / (H(torch.ones_like(x))[i] * rt_squared[i] + sigma_noise**2) * d[i]
                        sol[i] = sol_tmp.reshape(d[i].shape)

                    vec = H_adj(sol)

                # do vector jacobian product
                t = t1.view(-1, 1, 1, 1)
                if gamma == "constant":
                    gamma = 1
                elif gamma == "gamma_t":
                    gamma = torch.sqrt(t / (t**2 + (1 - t)**2))
                g = torch.autograd.functional.vjp(lambda z:  self.denoise_fn(z, t1), inputs=x, v=vec)[1]

                with torch.no_grad():
                    g = vec + (1-t1.view(-1, 1, 1, 1)) * g
                    ratio = (1-t1.view(-1, 1, 1, 1)) / t1.view(-1, 1, 1, 1)
                    v_adapted = vt + ratio * gamma * g
                    x_new = x + delta * v_adapted
                    x = x_new

            # Save results
            restored_img = x.detach().clone()
            save_image(restored_img.cpu().numpy()[0][0], output_folder, f'{img_id}_reconstructed.png')

            # Compute metrics
            psnr_rec, ssim_rec, lpips_rec = compute_metrics(gt_img, degraded_img, restored_img, problem, H_adj, img_id)
            psnrs.append(psnr_rec)
            ssims.append(ssim_rec)
            lpips.append(lpips_rec)

        return psnrs, ssims, lpips

    def solve_ip_flow_priors(
            self,
            test_dataset,
            problem: str,
            degradation,
            sigma_noise: float,
            params: list,
            output_folder: str = None
    ):
        def hut_estimator(NO_test, v, inp, t):
            batch_size = inp.shape[0]

            def fn(x):
                x = x.reshape(batch_size * NO_test, inp.shape[1], inp.shape[2],
                              inp.shape[3])
                return v(x, torch.tensor([t]).repeat(x.shape[0]).to('cuda')).reshape(NO_test, batch_size, inp.shape[1],
                                                                                     inp.shape[2], inp.shape[3])

            inp_new = inp.repeat(NO_test, 1, 1, 1, 1).clone()
            eps = ((torch.rand(NO_test, batch_size, inp.shape[1],
                               inp.shape[2], inp.shape[3], device='cuda') < 0.5)) * 2 - 1
            prod = torch.autograd.functional.jvp(fn, inp_new, eps, create_graph=True)[1]

            prod = (prod * eps).sum(dim=(2, 3, 4)).mean(0)
            return prod

        # Initialization
        lmbda, eta = params
        N, K = 100, 1
        start_time = 0.0

        H, H_adj = degradation.H, degradation.H_adj
        psnrs, ssims, lpips = [], [], []

        # Main loop
        torch.cuda.empty_cache()
        for item in test_dataset:
            gt_img, img_id, landmarks = item
            gt_img = gt_img.unsqueeze(0).expand(1, -1, -1, -1)
            save_image(gt_img.cpu().numpy()[0][0], output_folder, f'{img_id}_gt_img.png')

            degraded_img = H(gt_img.clone().to(self.device))
            torch.manual_seed(img_id)
            degraded_img += torch.randn_like(degraded_img) * sigma_noise
            degraded_img, gt_img = degraded_img.to(self.device), gt_img.to('cpu')

            if problem == 'occlusion_removal':
                occluded_image, occlusion_mask = occlusion_simulation.simulate(degraded_img, landmarks[:, 1:])
                degraded_img[:, 0, :, :] = occluded_image.clone()

                if torch.all(occlusion_mask == 0):
                    # skip the image if no occlusions were added
                    continue

            save_image(degraded_img.cpu().numpy()[0][0], output_folder, f'{img_id}_input_img.png')

            x_init = torch.randn(gt_img.shape).to(self.device)
            x = x_init.clone()
            x.requires_grad_(True)

            if start_time > 0.0:
                eps = 1 * start_time
                dt = (1 - eps) / N
            else:
                # Uniform
                dt = 1. / N
                eps = 1e-3

            # Iterative reconstruction
            for iteration in range(N):
                num_t = iteration / N * (1 - eps) + eps
                t1 = torch.ones(len(x), device=self.device) * num_t
                t = t1.view(-1, 1, 1, 1)

                x = x.detach().clone()
                x.requires_grad = True
                optim_img = torch.optim.Adam([x], lr=eta)

                if iteration == 0:
                    for k in range(K):
                        x_next = x + self.denoise_fn(x, t1) * dt

                        y_next = (t + dt) * degraded_img + (1 - (t + dt)) * H(x_init)
                        trace_term = hut_estimator(1, self.denoise_fn, x, num_t)
                        loss = lmbda * torch.sum((H(x_next) - y_next) ** 2, dim=(1, 2, 3))

                        loss += 0.5 * torch.sum(x ** 2, dim=(1, 2, 3)) + trace_term * dt
                        loss = loss.sum()
                        optim_img.zero_grad()
                        grad = torch.autograd.grad(loss, x, create_graph=False)[0]
                        x.grad = grad
                        optim_img.step()
                else:
                    for k in range(K):
                        pred = self.denoise_fn(x, t1)
                        x_next = x + pred * dt
                        y_next = (t + dt) * degraded_img + (1 - (t + dt)) * H(x_init)

                        trace_term = hut_estimator(1, self.denoise_fn, x, num_t)
                        loss = lmbda * torch.sum((H(x_next) - y_next) ** 2, dim=(1, 2, 3))
                        loss += trace_term * dt
                        loss = loss.sum()

                        optim_img.zero_grad()
                        grad = torch.autograd.grad(loss, x, create_graph=False)[0]
                        grad_xt_lik = (- 1.0 / (1.0 - num_t) * (-x + num_t * pred.detach())).detach()
                        x.grad = grad + grad_xt_lik
                        optim_img.step()

                x = x + self.denoise_fn(x, t1) * dt
                torch.cuda.empty_cache()
                del optim_img

            # Save results
            restored_img = x.detach().clone()
            save_image(restored_img.cpu().numpy()[0][0], output_folder, f'{img_id}_reconstructed.png')

            # Compute metrics
            psnr_rec, ssim_rec, lpips_rec = compute_metrics(gt_img, degraded_img, restored_img, problem, H_adj, img_id)
            psnrs.append(psnr_rec)
            ssims.append(ssim_rec)
            lpips.append(lpips_rec)

            del degraded_img, gt_img, x
            torch.cuda.empty_cache()

        return psnrs, ssims, lpips

    def solve_ip_d_flow(
            self,
            test_dataset,
            problem: str,
            degradation,
            sigma_noise: float,
            params: list,
            output_folder: str = None
    ):
        def model_forward(x, t):
            return self.denoise_fn(x, t)

        def gaussian(img):
            if img.ndim != 4:
                raise RuntimeError(
                    f"Expected input `img` to be an 4D tensor, but got {img.shape}")
            return (img ** 2).sum([1, 2, 3]) * 0.5

        def forward_flow_matching(z):
            steps = steps_euler
            delta = (1 - start_time) / (steps - 1)
            for i in range(steps - 1):
                t1 = torch.ones(len(z), device=self.device) * \
                     delta * i + start_time
                z = z + delta * model_forward(z + delta /
                                                   2 * model_forward(z, t1), t1 + delta / 2)
            return z

        def inverse_flow_matching(z):
            flow_class = Cnf(self.denoise_fn)
            z_t = odeint(flow_class, z,
                         torch.tensor([1.0, 0.0]).to(self.device),
                         atol=1e-5,
                         rtol=1e-5,
                         method='dopri5',
                         )
            x = z_t[-1].detach()
            return x

        # Initialization
        lmbda, alpha, LBFGS_iter = params
        steps_euler = 6
        start_time = 0.0
        max_iter = 20

        H = degradation.H
        H_adj = degradation.H_adj
        psnrs, ssims, lpips = [], [], []

        # Main loop
        for item in test_dataset:
            gt_img, img_id, landmarks = item
            gt_img = gt_img.unsqueeze(0).expand(1, -1, -1, -1)
            save_image(gt_img.cpu().numpy()[0][0], output_folder,f'{img_id}_gt_img.png')

            degraded_img = H(gt_img.clone().to(self.device))
            torch.manual_seed(img_id)
            degraded_img += torch.randn_like(degraded_img) * sigma_noise
            degraded_img, gt_img = degraded_img.to(self.device), gt_img.to('cpu')

            if problem == 'occlusion_removal':
                occluded_image, occlusion_mask = occlusion_simulation.simulate(degraded_img, landmarks[:, 1:])
                degraded_img[:, 0, :, :] = occluded_image.clone()

                if torch.all(occlusion_mask == 0):
                    # skip the image if no occlusions were added
                    continue

            save_image(degraded_img.cpu().numpy()[0][0], output_folder,f'{img_id}_input_img.png')

            x = H_adj(degraded_img.clone()).to(self.device)
            z = inverse_flow_matching(x).to(self.device)

            # Blend initialization as in the D-Flow paper
            z = np.sqrt(alpha) * z + np.sqrt(1 - alpha) * torch.randn_like(z)
            z = z.detach().requires_grad_(True)

            # Start the gradient descent
            optim_img = torch.optim.LBFGS([z], max_iter=LBFGS_iter, history_size=100, line_search_fn='strong_wolfe')
            d = z.shape[1] * z.shape[2] * z.shape[3]

            for iteration in range(max_iter):
                def closure():
                    optim_img.zero_grad()  # Reset gradients
                    reg = - torch.clamp(gaussian(z), min=-1e6, max=1e6) + (
                            d - 1) * torch.log(torch.sqrt(torch.sum(z ** 2, dim=(1, 2, 3))) + 1e-5)

                    loss = (torch.sum((H(forward_flow_matching(z)) -
                                       degraded_img) ** 2, dim=(1, 2, 3)) + lmbda * reg).sum()
                    loss.backward()  # Compute gradients
                    return loss

                optim_img.step(closure)
                restored_img = forward_flow_matching(z.detach())
                del restored_img

            z = z.detach().requires_grad_(False)

            # Save results
            restored_img = forward_flow_matching(z.detach())
            save_image(restored_img.detach().cpu().numpy()[0][0], output_folder, f'{img_id}_reconstructed.png')

            # Compute metrics
            psnr_rec, ssim_rec, lpips_rec = compute_metrics(gt_img, degraded_img, restored_img, problem, H_adj, img_id)
            psnrs.append(psnr_rec)
            ssims.append(ssim_rec)
            lpips.append(lpips_rec)

            del restored_img
            del degraded_img, gt_img, x, z

        return psnrs, ssims, lpips

    def solve_ip_pnp_flow(
            self,
            test_dataset,
            problem: str,
            degradation,
            sigma_noise: float,
            params: list,
            output_folder: str = None
    ):
        def learning_rate_strat(lr, t):
            t = t.view(-1, 1, 1, 1)
            gamma_styles = {
                '1_minus_t': lambda lr, t: lr * (1 - t),
                'sqrt_1_minus_t': lambda lr, t: lr * torch.sqrt(1 - t),
                'constant': lambda lr, t: lr,
                'alpha_1_minus_t': lambda lr, t: lr * (1 - t) ** alpha,
            }
            return gamma_styles.get('alpha_1_minus_t', lambda lr, t: lr)(lr, t)

        def grad_datafit(x, y, H, H_adj):
            return H_adj(H(x) - y) / (sigma_noise ** 2)

        def interpolation_step(x, t):
            return t * x + torch.randn_like(x) * (1 - t)

        def denoiser(x, t):
            v = self.denoise_fn(x, t)
            return x + (1 - t.view(-1, 1, 1, 1)) * v

        # Initialization
        steps_pnp, lr_pnp, alpha = params
        num_samples = 5
        lr = sigma_noise ** 2 * lr_pnp
        steps, delta = steps_pnp, 1 / steps_pnp

        H, H_adj = degradation.H, degradation.H_adj
        psnrs, ssims, lpips = [], [], []

        # Main loop
        for item in test_dataset:
            gt_img, img_id, landmarks = item
            gt_img = gt_img.unsqueeze(0).expand(1, -1, -1, -1)
            save_image(gt_img.cpu().numpy()[0][0], output_folder, f'{img_id}_gt_img.png')

            degraded_img = H(gt_img.clone().to(self.device))
            torch.manual_seed(img_id)
            degraded_img += torch.randn_like(degraded_img) * sigma_noise
            degraded_img, gt_img = degraded_img.to(self.device), gt_img.to('cpu')

            # Initialize x using the adjoint operator
            x = H_adj(torch.ones_like(degraded_img)).to(self.device)

            # Occlusion removal
            if problem == 'occlusion_removal':
                occluded_image, occlusion_mask = occlusion_simulation.simulate(degraded_img, landmarks[:, 1:])
                degraded_img[:, 0, :, :] = occluded_image.clone()

                if torch.all(occlusion_mask == 0):
                    continue  # skip images with no occlusions

            # Save input image
            save_image(degraded_img.cpu().numpy()[0][0], output_folder, f'{img_id}_input_img.png')

            # Iterative reconstruction
            with torch.no_grad():
                for iteration in range(int(steps)):
                    t = torch.ones(len(x), device=self.device) * delta * iteration
                    lr_t = learning_rate_strat(lr, t)

                    z = x - lr_t * grad_datafit(x, degraded_img, H, H_adj)

                    x_new = torch.zeros_like(x)
                    for _ in range(num_samples):
                        z_tilde = interpolation_step(z, t.view(-1, 1, 1, 1))
                        x_new += denoiser(z_tilde, t)

                    x = x_new / num_samples

            # Save results
            restored_img = x.detach().clone()
            save_image(restored_img.cpu().numpy()[0][0], output_folder, f'{img_id}_reconstructed.png')

            # Compute metrics
            psnr_rec, ssim_rec, lpips_rec = compute_metrics(gt_img, degraded_img, restored_img, problem, H_adj, img_id)
            psnrs.append(psnr_rec)
            ssims.append(ssim_rec)
            lpips.append(lpips_rec)

        return psnrs, ssims, lpips
