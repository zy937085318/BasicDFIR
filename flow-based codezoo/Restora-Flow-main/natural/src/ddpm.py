# Parts of this file are adapted from https://github.com/mobaidoctor/med-ddpm
# RePaint implementation adapted from https://github.com/andreas128/RePaint
# DDNM implementation adapted from https://github.com/wyhuai/DDNM

import torch
from torch import nn
from functools import partial
import numpy as np
from tqdm import tqdm
import warnings
from collections import defaultdict
from utils import scheduler
import helpers as helpers

warnings.filterwarnings("ignore", category=UserWarning)


# Helper functions
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


# Diffusion class
class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            *,
            image_size,
            channels=1,
            timesteps=1000,
            loss_type='l1',
            betas=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

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

    def predict_start_from_noise(self, x_t, t, noise):
        x_hat = 0.
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise -
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_hat
        )

    def q_posterior(self, x_start, x_t, t):
        x_hat = 0.
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t +
                extract(self.posterior_mean_coef3, t, x_t.shape) * x_hat
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    def undo(self, image_before_step, img_after_model, est_x_0, t, debug=False):
        return self._undo(img_after_model, t)

    def _undo(self, img_out, t):
        # beta = _extract_into_tensor(self.betas, t, img_out.shape)
        beta = extract(self.betas, t, img_out.shape)

        img_in_est = torch.sqrt(1 - beta) * img_out + torch.sqrt(beta) * torch.randn_like(img_out)

        return img_in_est

    def p_sample_repaint(self, x0, mask, x, t, clip_denoised=False, cond_fn=None, model_kwargs=None, pred_xstart=None):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param cond_fn: if not None, this is a gradient function that acts similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to pass to the model. This can be used for
        conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        noise = torch.randn_like(x, device="cuda")

        # if conf.inpa_inj_sched_prev:

        gt = x0

        if pred_xstart is not None:
            # gt_keep_mask = model_kwargs.get('gt_keep_mask')
            gt_keep_mask = mask

            # if gt_keep_mask is None:
            #     gt_keep_mask = conf.get_inpa_mask(x)

            # gt = model_kwargs['gt']

            # alpha_cumprod = _extract_into_tensor(
            #     self.alphas_cumprod, t, x.shape)
            alpha_cumprod = extract(self.alphas_cumprod, t, x.shape)

            # if conf.inpa_inj_sched_prev_cumnoise:
            #     weighed_gt = self.get_gt_noised(gt, int(t[0].item()))
            # else:
            # ---
            gt_weight = torch.sqrt(alpha_cumprod)
            gt_part = gt_weight * gt

            noise_weight = torch.sqrt((1 - alpha_cumprod))
            noise_part = noise_weight * torch.randn_like(x)

            weighed_gt = gt_part + noise_part
            # ---

            x = (gt_keep_mask * weighed_gt + (1 - gt_keep_mask) * x)

        # out = self.p_mean_variance(
        #     model,
        #     x,
        #     t,
        #     clip_denoised=clip_denoised,
        #     denoised_fn=denoised_fn,
        #     model_kwargs=model_kwargs,
        # )
        model_mean, model_variance, model_log_variance, pred_xstart = (
            self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised))
        out = {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )

        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )

        sample = out["mean"] + nonzero_mask * \
                 torch.exp(0.5 * out["log_variance"]) * noise

        result = {
            "sample": sample,
            "pred_xstart": out["pred_xstart"],
            "gt": gt,
        }

        return result

    def p_sample_loop_progressive_repaint(self, x0, mask, shape, output_folder, noise=None, clip_denoised=False,
                                          jump_length=10, jump_n_sample=10, cond_fn=None, model_kwargs=None,
                                          device=None, progress=False):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """

        if device is None:
            device = next(self.denoise_fn.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            image_after_step = noise
        else:
            image_after_step = torch.randn(*shape, device=device)

        # helpers.save_image(x0, output_folder, 'input_img.png')
        # helpers.save_image(mask, output_folder, 'mask.png')

        self.gt_noises = None  # reset for next image
        pred_xstart = None

        idx_wall = -1
        sample_idxs = defaultdict(lambda: 0)
        n_sample = shape[0]

        # if conf.schedule_jump_params:
        times = scheduler.get_schedule_jump(
            t_T=self.num_timesteps,
            n_sample=n_sample,
            jump_length=jump_length,
            jump_n_sample=jump_n_sample
        )

        time_pairs = list(zip(times[:-1], times[1:]))
        if progress:
            from tqdm.auto import tqdm
            time_pairs = tqdm(time_pairs)

        for t_last, t_cur in time_pairs:
            idx_wall += 1
            t_last_t = torch.tensor([t_last] * shape[0],  # pylint: disable=not-callable
                                    device=device)

            # helpers.save_image(out["sample"], output_folder, 'output_img.png')

            if t_cur < t_last:  # reverse
                with torch.no_grad():
                    image_before_step = image_after_step.clone()
                    out = self.p_sample_repaint(x0, mask, image_after_step, t_last_t, clip_denoised=clip_denoised,
                                                cond_fn=cond_fn, model_kwargs=model_kwargs, pred_xstart=pred_xstart)
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

        helpers.save_image(out["sample"], output_folder, 'output_img_final.png')
        return out["sample"]

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device

        model_mean, _, model_log_variance, _ = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))  # no noise when t == 0
        sample = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return sample

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    @torch.no_grad()
    def sample(self, batch_size=1):
        return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size))

    def simplified_ddnm_plus(self, y, A, Ap, sigma_y, jump_length=1, jump_n_sample=1, sampling_steps=100):
        eta = 0.85  # as in DDNM paper
        sigma_y = 2 * sigma_y  # to account for scaling to [-1, 1]
        sigma_y = sigma_y

        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
            return a

        # init x_T
        x = torch.randn(y.shape[0], self.channels, self.image_size, self.image_size, device="cuda")

        with torch.no_grad():
            skip = self.num_timesteps // sampling_steps
            n = x.size(0)
            x0_preds = []
            xs = [x]

            times = scheduler.get_schedule_jump(
                t_T=sampling_steps,  # DDIM sampling steps
                n_sample=1,
                jump_length=jump_length,    # jump length
                jump_n_sample=jump_n_sample  # resampling steps
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

                    # Eq. 19 for a batch size > 1
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

                    # different from the paper, we use DDIM here instead of DDPM
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
