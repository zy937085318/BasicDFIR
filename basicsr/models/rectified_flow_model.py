"""
Rectified Flow Model for Super Resolution
"""
from __future__ import annotations

import math
from collections import OrderedDict, namedtuple
from typing import Literal, Callable

import torch
from torch import Tensor
from torch import nn, pi
from torch.nn import Module
from torch.distributions import Normal
import torch.nn.functional as F

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("Warning: torchdiffeq not available. Sampling will not work.")

try:
    from einops import rearrange, repeat
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False
    def rearrange(tensor, pattern, **kwargs):
        return tensor
    def repeat(tensor, pattern, **kwargs):
        return tensor

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def from_numpy(arr):
    return torch.from_numpy(arr)

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .flow_model import FlowModel


# helpers

def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def identity(t):
    return t


# tensor helpers

def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))


# normalizing helpers

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# noise schedules

def cosmap(t):
    # Algorithm 21 in https://arxiv.org/abs/2403.03206
    return 1. - (1. / (torch.tan(pi / 2 * t) + 1))


# loss breakdown

LossBreakdown = namedtuple('LossBreakdown', ['total', 'main', 'data_match', 'velocity_match'])


# EMA helper (simplified version)

class EMA:
    """Exponential Moving Average"""
    def __init__(self, model, beta=0.9999, update_after_step=100, include_online_model=False):
        self.model = model
        self.ema_model = self._copy_model(model)
        self.beta = beta
        self.update_after_step = update_after_step
        self.step = 0
        self.include_online_model = include_online_model

    def _copy_model(self, model):
        import copy
        return copy.deepcopy(model)

    def update(self):
        self.step += 1
        if self.step < self.update_after_step:
            return

        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.beta).add_(param.data, alpha=1 - self.beta)


@MODEL_REGISTRY.register()
class RectifiedFlowModel(FlowModel):
    """Rectified Flow Model for Super Resolution"""

    def __init__(self, opt):
        # Get model configuration before calling parent init
        model_opt = opt.get('rectified_flow', {})

        # Call parent FlowModel __init__ which will build network
        super(RectifiedFlowModel, self).__init__(opt)

        # Rectified Flow specific settings (set after parent init)
        self.time_cond_kwarg = model_opt.get('time_cond_kwarg', 'times')
        self.predict = model_opt.get('predict', 'flow')  # 'flow' or 'noise'
        self.mean_variance_net = model_opt.get('mean_variance_net', False)

        # Noise schedule
        noise_schedule = model_opt.get('noise_schedule', 'cosmap')
        if noise_schedule == 'cosmap':
            self.noise_schedule = cosmap
        elif callable(noise_schedule):
            self.noise_schedule = noise_schedule
        else:
            self.noise_schedule = identity

        # ODE solver settings
        self.odeint_kwargs = model_opt.get('odeint_kwargs', dict(
            atol=1e-5,
            rtol=1e-5,
            method='midpoint'
        ))

        # Clipping settings
        self.clip_during_sampling = model_opt.get('clip_during_sampling', False)
        self.clip_flow_during_sampling = model_opt.get('clip_flow_during_sampling', None)
        self.clip_values = model_opt.get('clip_values', (-1., 1.))
        self.clip_flow_values = model_opt.get('clip_flow_values', (-3., 3))

        # Consistency flow matching
        self.use_consistency = model_opt.get('use_consistency', False)
        self.consistency_decay = model_opt.get('consistency_decay', 0.9999)
        self.consistency_velocity_match_alpha = model_opt.get('consistency_velocity_match_alpha', 1e-5)
        self.consistency_delta_time = model_opt.get('consistency_delta_time', 1e-3)
        self.consistency_loss_weight = model_opt.get('consistency_loss_weight', 1.0)

        # Data normalization
        self.data_normalize_fn = model_opt.get('data_normalize_fn', normalize_to_neg_one_to_one)
        self.data_unnormalize_fn = model_opt.get('data_unnormalize_fn', unnormalize_to_zero_to_one)

        # Immiscible diffusion
        self.immiscible = model_opt.get('immiscible', False)

        # Data augmentation: add small Gaussian noise to x_0 and gt during training
        self.train_noise_std = model_opt.get('train_noise_std', 0.01)  # Default: 0.01 (1% noise)

        # Data shape (will be set during first forward)
        self.data_shape = None

        # EMA model for consistency
        if self.use_consistency:
            ema_update_after_step = model_opt.get('ema_update_after_step', 100)
            ema_kwargs = model_opt.get('ema_kwargs', {})
            self.ema_model = EMA(
                self.net_g,
                beta=self.consistency_decay,
                update_after_step=ema_update_after_step,
                **ema_kwargs
            )

        # Note: init_training_settings is already called by parent FlowModel.__init__
        # and will use this class's overridden method

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0 and not self.use_consistency:
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
            # Default to MSE loss
            self.flow_loss = build_loss({'type': 'MSELoss', 'loss_weight': 1.0}).to(self.device)
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        if self.flow_loss is None and self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('All losses (flow, pixel, perceptual) are None.')

        # Cache whether flow_loss accepts additional arguments (performance optimization)
        self._flow_loss_accepts_extra_args = False
        if hasattr(self.flow_loss, 'forward'):
            try:
                import inspect
                sig = inspect.signature(self.flow_loss.forward)
                self._flow_loss_accepts_extra_args = 'pred_data' in sig.parameters
            except:
                # Fallback: check co_varnames only once
                if hasattr(self.flow_loss.forward, '__code__'):
                    self._flow_loss_accepts_extra_args = 'pred_data' in self.flow_loss.forward.__code__.co_varnames

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def predict_flow(self, model: Module, noised, *, times, eps=1e-10, **model_kwargs):
        """Returns the model output as well as the derived flow"""
        batch = noised.shape[0]

        # Prepare time conditioning
        time_kwarg = self.time_cond_kwarg
        if exists(time_kwarg):
            if EINOPS_AVAILABLE:
                times = rearrange(times, '... -> (...)')
            else:
                times = times.flatten()

            if times.numel() == 1:
                if EINOPS_AVAILABLE:
                    times = repeat(times, '1 -> b', b=batch)
                else:
                    times = times.expand(batch)

            model_kwargs.update(**{time_kwarg: times})

        output = model(noised, **model_kwargs)

        # Derive flow depending on objective
        if self.predict == 'flow':
            flow = output
        elif self.predict == 'noise':
            noise = output
            padded_times = append_dims(times, noised.ndim - 1)
            flow = (noised - noise) / padded_times.clamp(min=eps)
        else:
            raise ValueError(f'unknown objective {self.predict}')

        return output, flow

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        # Clear cache when new data arrives (will be recomputed in flow_interpolation)
        if hasattr(self, '_cached_x_0_base'):
            delattr(self, '_cached_x_0_base')
        if hasattr(self, '_cached_x_1_base'):
            delattr(self, '_cached_x_1_base')
        if hasattr(self, '_cached_lq_id'):
            delattr(self, '_cached_lq_id')
        if hasattr(self, '_cached_gt_id'):
            delattr(self, '_cached_gt_id')

    '''
    define the interpolation processing of flow-based method. Should be instanced for different flow-based method.
    input: time step t
    '''
    def flow_interpolation(self, t, r=None):
        """Rectified Flow interpolation: x_t = x_0 + t * (x_1 - x_0)"""
        batch = self.lq.shape[0]

        # Cache normalized and upsampled data to avoid recomputing every iteration
        # Use data pointer/id to check if data actually changed (more reliable than shape)
        current_lq_id = id(self.lq) if hasattr(self, 'gt') and self.gt is not None else id(self.lq)
        current_gt_id = id(self.gt) if hasattr(self, 'gt') and self.gt is not None else None

        if (not hasattr(self, '_cached_x_0') or
            not hasattr(self, '_cached_lq_id') or
            self._cached_lq_id != current_lq_id or
            (current_gt_id is not None and (not hasattr(self, '_cached_gt_id') or self._cached_gt_id != current_gt_id))):
            # Normalize LQ and GT
            lq_norm = self.data_normalize_fn(self.lq)

            # Determine target shape
            if hasattr(self, 'gt') and self.gt is not None:
                gt_norm = self.data_normalize_fn(self.gt)
                target_h, target_w = self.gt.shape[-2], self.gt.shape[-1]
                # Upsample LQ to GT size
                x_0_base = F.interpolate(lq_norm, size=(target_h, target_w), mode='bilinear', align_corners=False)
                x_1_base = gt_norm
            else:
                # If no GT, use LQ as both start and end
                x_0_base = lq_norm
                x_1_base = lq_norm

            # Cache base values (without noise) for reuse
            self._cached_x_0_base = x_0_base
            self._cached_x_1_base = x_1_base
            self._cached_lq_id = current_lq_id
            if current_gt_id is not None:
                self._cached_gt_id = current_gt_id
        else:
            # Reuse cached base values
            x_0_base = self._cached_x_0_base
            x_1_base = self._cached_x_1_base

        # Add small Gaussian noise during training for data augmentation
        # Add noise every time (not cached) to ensure randomness in each iteration
        if self.opt.get('is_train', False) and self.train_noise_std > 0:
            # Generate noise with same shape as x_0 and x_1
            noise_x0 = torch.randn_like(x_0_base) * self.train_noise_std
            noise_x1 = torch.randn_like(x_1_base) * self.train_noise_std
            x_0 = x_0_base + noise_x0
            x_1 = x_1_base + noise_x1
            # Clip to valid range after adding noise
            if hasattr(self, 'clip_values'):
                x_0 = torch.clamp(x_0, self.clip_values[0], self.clip_values[1])
                x_1 = torch.clamp(x_1, self.clip_values[0], self.clip_values[1])
            else:
                x_0 = torch.clamp(x_0, -1.0, 1.0)
                x_1 = torch.clamp(x_1, -1.0, 1.0)
        else:
            # No noise during testing
            x_0 = x_0_base
            x_1 = x_1_base

        # Apply noise schedule to time
        t_scheduled = self.noise_schedule(t)

        # Linear interpolation: x_t = x_0 + t * (x_1 - x_0)
        if isinstance(t_scheduled, torch.Tensor):
            if t_scheduled.dim() == 0:
                t_scheduled = t_scheduled.unsqueeze(0)
            if t_scheduled.shape[0] == 1 and batch > 1:
                t_scheduled = t_scheduled.expand(batch)
            t_scheduled = append_dims(t_scheduled, x_0.ndim - 1)

        x_t = x_0.lerp(x_1, t_scheduled)

        # Store for later use
        self.x_0 = x_0
        self.x_1 = x_1
        self.t_scheduled = t_scheduled

        return x_t

    '''
    define the timestep sampling method.
    '''
    def sample_timestep(self):
        """Sample random time step for training"""
        batch = self.lq.shape[0]
        self.t = torch.rand(batch, device=self.device)
        # Adjust times for consistency loss
        if self.use_consistency:
            self.t = self.t * (1. - self.consistency_delta_time)
        return self.t

    '''
    main processing of flow.
    '''
    def flow_process(self):
        """Main flow processing: sample timestep and interpolate"""
        self.sample_timestep()
        self.xt = self.flow_interpolation(self.t)

    '''
    sample image with flow-based ODE.
    '''
    def sample_image(self, lq=None, model=None, ema=False):
        """Sample image using flow-based ODE solver"""
        if lq is None:
            lq = self.lq

        # Determine which model to use
        if model is not None:
            model_to_use = model
        elif ema and hasattr(self, 'net_g_ema') and not self.use_consistency:
            model_to_use = self.net_g_ema
        elif self.use_consistency and hasattr(self, 'ema_model'):
            model_to_use = self.ema_model.ema_model
        else:
            model_to_use = self.net_g

        # For super-resolution, use direct forward pass for stability
        # (ODE sampling can be unstable during early training)
        if hasattr(self, 'gt') and self.gt is not None:
            gt_shape = self.gt.shape[1:]
        else:
            # Estimate GT shape from scale
            scale = self.opt.get('scale', 4)
            gt_shape = (
                lq.shape[1],  # channels
                int(lq.shape[2] * scale),  # height
                int(lq.shape[3] * scale)   # width
            )

        # Upsample LQ to target size
        lq_upsampled = F.interpolate(
            lq,
            size=(gt_shape[-2], gt_shape[-1]),
            mode='bilinear',
            align_corners=False
        )

        # Normalize
        lq_norm = self.data_normalize_fn(lq_upsampled)

        # Pad to be divisible by 8
        lq_padded, (original_h, original_w) = self._pad_to_divisible(lq_norm, divisor=8)

        # Use direct forward pass: predict flow at t=1
        times = torch.ones(lq.shape[0], device=self.device)
        _, flow = self.predict_flow(model_to_use, lq_padded, times=times)

        # Clip flow
        if hasattr(self, 'clip_flow_values'):
            flow = torch.clamp(flow, self.clip_flow_values[0], self.clip_flow_values[1])
        else:
            flow = torch.clamp(flow, -2.0, 2.0)

        # Apply flow: x_1 = x_0 + flow
        output_padded = lq_padded + flow

        # Clip output
        if hasattr(self, 'clip_values'):
            output_padded = torch.clamp(output_padded, self.clip_values[0], self.clip_values[1])
        else:
            output_padded = torch.clamp(output_padded, -1.0, 1.0)

        # Crop back
        output = self._crop_from_padded(output_padded, original_h, original_w)

        # Unnormalize
        output = self.data_unnormalize_fn(output)
        output = torch.clamp(output, 0.0, 1.0)

        return output

    '''
    Add flow-based loss function.
    '''
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.flow_process()

        # Predict flow using network
        # Prepare time conditioning for network
        times = self.t
        if exists(self.time_cond_kwarg):
            if EINOPS_AVAILABLE:
                times_flat = rearrange(times, '... -> (...)')
            else:
                times_flat = times.flatten()

            if times_flat.numel() == 1:
                if EINOPS_AVAILABLE:
                    times_expanded = repeat(times_flat, '1 -> b', b=self.xt.shape[0])
                else:
                    times_expanded = times_flat.expand(self.xt.shape[0])
            else:
                times_expanded = times_flat

            model_kwargs = {self.time_cond_kwarg: times_expanded}
        else:
            model_kwargs = {}

        # Network forward pass
        model_output = self.net_g(self.xt, **model_kwargs)

        # Derive flow from model output
        if self.predict == 'flow':
            self.vt_pre = model_output
        elif self.predict == 'noise':
            noise = model_output
            padded_times = append_dims(times_expanded, self.xt.ndim - 1)
            self.vt_pre = (self.xt - noise) / padded_times.clamp(min=1e-10)
        else:
            raise ValueError(f'unknown objective {self.predict}')

        # Handle mean-variance network
        if self.mean_variance_net:
            mean, variance = model_output
            self.vt_pre = torch.normal(mean, variance)

        # Compute target flow: v_t = x_1 - x_0
        self.vt_tar = self.x_1 - self.x_0

        # Compute predicted data for loss functions that need it
        pred_data = self.xt + self.vt_pre * (1. - self.t_scheduled)

        # Set output for loss computation
        self.output = pred_data

        l_total = 0
        loss_dict = OrderedDict()

        # flow loss
        if self.flow_loss:
            # Use cached flag instead of checking every iteration
            if self._flow_loss_accepts_extra_args:
                l_flow = self.flow_loss(model_output, self.vt_tar, pred_data=pred_data,
                                       times=times, data=self.x_1)
            else:
                l_flow = self.flow_loss(model_output, self.vt_tar)
            l_total += l_flow
            loss_dict['l_flow'] = l_flow

        # pixel loss (if available)
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.x_1)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss (if available)
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.x_1)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        # Consistency loss (if using consistency flow matching)
        if self.use_consistency:
            # Get EMA model prediction
            delta_t = self.consistency_delta_time
            ema_times = self.t + delta_t
            if EINOPS_AVAILABLE:
                ema_times_flat = rearrange(ema_times, '... -> (...)')
            else:
                ema_times_flat = ema_times.flatten()

            if ema_times_flat.numel() == 1:
                if EINOPS_AVAILABLE:
                    ema_times_expanded = repeat(ema_times_flat, '1 -> b', b=self.xt.shape[0])
                else:
                    ema_times_expanded = ema_times_flat.expand(self.xt.shape[0])
            else:
                ema_times_expanded = ema_times_flat

            ema_model_kwargs = {self.time_cond_kwarg: ema_times_expanded}
            ema_output = self.ema_model.ema_model(self.xt, **ema_model_kwargs)

            if self.predict == 'flow':
                ema_vt_pre = ema_output
            elif self.predict == 'noise':
                ema_noise = ema_output
                ema_padded_times = append_dims(ema_times_expanded, self.xt.ndim - 1)
                ema_vt_pre = (self.xt - ema_noise) / ema_padded_times.clamp(min=1e-10)

            if self.mean_variance_net:
                ema_mean, ema_variance = ema_output
                ema_vt_pre = torch.normal(ema_mean, ema_variance)

            ema_pred_data = self.xt + ema_vt_pre * (1. - (self.t_scheduled + delta_t))

            data_match_loss = F.mse_loss(pred_data, ema_pred_data)
            velocity_match_loss = F.mse_loss(self.vt_pre, ema_vt_pre)
            consistency_loss = data_match_loss + velocity_match_loss * self.consistency_velocity_match_alpha

            l_total += consistency_loss * self.consistency_loss_weight
            loss_dict['l_data_match'] = data_match_loss
            loss_dict['l_velocity_match'] = velocity_match_loss
            loss_dict['l_consistency'] = consistency_loss

        l_total.backward()
        self.optimizer_g.step()

        # Print loss every 100 iterations
        if current_iter % 100 == 0:
            logger = get_root_logger()
            loss_str = f'[Iter {current_iter}] Losses: '
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    loss_str += f'{k}: {v.item():.6f} '
                else:
                    loss_str += f'{k}: {v:.6f} '
            logger.info(loss_str)

        # Don't log loss values to avoid printing in regular logs
        self.log_dict = OrderedDict()

        # Update EMA
        if self.ema_decay > 0 and not self.use_consistency:
            self.model_ema(decay=self.ema_decay)

        if self.use_consistency:
            self.ema_model.update()


    @torch.no_grad()
    def sample(
            self,
            batch_size=1,
            steps=16,
            noise=None,
            data_shape=None,
            temperature: float = 1.,
            use_ema: bool = False,
            **model_kwargs
    ):
        """Sample from the rectified flow model"""
        if not TORCHDIFFEQ_AVAILABLE:
            raise ImportError("torchdiffeq is required for sampling")

        use_ema = default(use_ema, self.use_consistency)
        assert not (use_ema and not self.use_consistency), \
            'in order to sample from an ema model, you must have `use_consistency` turned on'

        model = self.ema_model.ema_model if use_ema else self.net_g

        # Note: BaseModel doesn't have training attribute, so we track it manually
        was_training = getattr(self, '_is_training', True)
        if hasattr(self, 'net_g'):
            self.net_g.eval()

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'you need to either pass in a `data_shape` or have trained at least with one forward'

        # Clipping functions
        maybe_clip = (lambda t: t.clamp_(*self.clip_values)) if self.clip_during_sampling else identity
        maybe_clip_flow = (lambda t: t.clamp_(*self.clip_flow_values)) if self.clip_flow_during_sampling else identity

        # ODE step function
        def ode_fn(t, x):
            x = maybe_clip(x)

            # Ensure times is a tensor with correct shape
            if isinstance(t, torch.Tensor):
                if t.dim() == 0:
                    t = t.unsqueeze(0)
                # Expand times to match batch size
                if t.shape[0] == 1 and x.shape[0] > 1:
                    t = t.expand(x.shape[0])
            else:
                t = torch.tensor([t], device=x.device, dtype=x.dtype)
                if x.shape[0] > 1:
                    t = t.expand(x.shape[0])

            _, output = self.predict_flow(model, x, times=t, **model_kwargs)

            flow = output

            if self.mean_variance_net:
                mean, variance = output
                std = variance.clamp(min=1e-5).sqrt()
                flow = torch.normal(mean, std * temperature)

            flow = maybe_clip_flow(flow)

            return flow

        # Start with random gaussian noise
        noise = default(noise, torch.randn((batch_size, *data_shape), device=self.device))

        # Time steps
        times = torch.linspace(0., 1., steps, device=self.device)

        # Solve ODE
        trajectory = odeint(ode_fn, noise, times, **self.odeint_kwargs)

        sampled_data = trajectory[-1]

        # Restore training mode
        if was_training and hasattr(self, 'net_g'):
            self.net_g.train()
        self._is_training = was_training

        return self.data_unnormalize_fn(sampled_data)

    def _pad_to_divisible(self, x, divisor=8):
        """Pad input tensor to be divisible by divisor

        Returns:
            padded_tensor: Padded tensor
            (original_h, original_w): Original dimensions before padding
        """
        b, c, h, w = x.shape
        original_h, original_w = h, w
        pad_h = (divisor - h % divisor) % divisor
        pad_w = (divisor - w % divisor) % divisor

        if pad_h > 0 or pad_w > 0:
            # Pad: (pad_left, pad_right, pad_top, pad_bottom)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            return x, (original_h, original_w)
        return x, (original_h, original_w)

    def _crop_from_padded(self, x, original_h, original_w):
        """Crop output tensor to remove padding

        Args:
            x: Padded tensor
            original_h, original_w: Original dimensions before padding
        """
        h, w = x.shape[-2:]
        if h > original_h or w > original_w:
            x = x[..., :original_h, :original_w]
        return x

    '''
    self.output is reconstructed image from sample_image method.
    '''
    def test(self):
        if hasattr(self, 'net_g_ema') and not self.use_consistency:
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.sample_image(self.lq, model=self.net_g_ema)
        elif self.use_consistency and hasattr(self, 'ema_model'):
            self.ema_model.ema_model.eval()
            with torch.no_grad():
                self.output = self.sample_image(self.lq, model=self.ema_model.ema_model)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.sample_image(self.lq, model=self.net_g)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            from tqdm import tqdm
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            from os import path as osp
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save_training_results(self, current_iter, result_dir):
        """Save training results (LQ, SR, GT comparison) during training"""
        import os
        import numpy as np
        from torchvision.utils import make_grid

        os.makedirs(result_dir, exist_ok=True)

        # Get visuals
        visuals = self.get_current_visuals()

        # Convert to images
        lq_img = visuals['lq']
        sr_img = visuals['result']

        # Handle different tensor shapes - take first image in batch
        if lq_img.dim() == 4:
            lq_img = lq_img[0]  # Take first image in batch
        if sr_img.dim() == 4:
            sr_img = sr_img[0]

        # Determine target size (use SR size as reference)
        target_h, target_w = sr_img.shape[-2:]

        # Debug: print image statistics
        logger = get_root_logger()
        logger.debug(f'LQ img stats: min={lq_img.min():.4f}, max={lq_img.max():.4f}, mean={lq_img.mean():.4f}')
        logger.debug(f'SR img stats: min={sr_img.min():.4f}, max={sr_img.max():.4f}, mean={sr_img.mean():.4f}')

        # Check for NaN or Inf in SR image
        if torch.isnan(sr_img).any() or torch.isinf(sr_img).any():
            logger.warning(f'SR image contains NaN or Inf. Replacing with upsampled LQ.')
            # Replace with upsampled LQ
            if lq_img.shape[-2:] != (target_h, target_w):
                sr_img = F.interpolate(
                    lq_img.unsqueeze(0),
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            else:
                sr_img = lq_img.clone()

        # Ensure images are in [0, 1] range
        # If images are normalized to [-1, 1], convert to [0, 1]
        if lq_img.min() < 0:
            lq_img = (lq_img + 1) / 2.0
        if sr_img.min() < 0:
            sr_img = (sr_img + 1) / 2.0

        # Normalize to [0, 1] if needed (images might be in [0, 255] range)
        if lq_img.max() > 1.0:
            lq_img = lq_img / 255.0
        if sr_img.max() > 1.0:
            sr_img = sr_img / 255.0

        # Clamp to [0, 1] and ensure values are valid
        lq_img = torch.clamp(lq_img, 0, 1)
        sr_img = torch.clamp(sr_img, 0, 1)

        # Additional check: if SR image is all zeros or very dark, use upsampled LQ
        if sr_img.mean() < 0.01 or sr_img.max() < 0.01:
            logger.warning(f'SR image appears to be very dark (mean={sr_img.mean():.4f}, max={sr_img.max():.4f}). Using upsampled LQ instead.')
            # Replace with upsampled LQ
            if lq_img.shape[-2:] != (target_h, target_w):
                sr_img = F.interpolate(
                    lq_img.unsqueeze(0),
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            else:
                sr_img = lq_img.clone()

        # Prepare images for grid - ensure all have same size
        images_to_show = []

        # Add LQ (upsampled to match SR size for better comparison)
        if lq_img.shape[-2:] != (target_h, target_w):
            lq_upsampled = F.interpolate(
                lq_img.unsqueeze(0),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            images_to_show.append(lq_upsampled)
        else:
            images_to_show.append(lq_img)

        # Add SR result (ensure it matches target size)
        if sr_img.shape[-2:] != (target_h, target_w):
            sr_img = F.interpolate(
                sr_img.unsqueeze(0),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        images_to_show.append(sr_img)

        # Add GT if available (resize to match target size)
        if 'gt' in visuals:
            gt_img = visuals['gt']
            if gt_img.dim() == 4:
                gt_img = gt_img[0]
            # Handle normalization
            if gt_img.min() < 0:
                gt_img = (gt_img + 1) / 2.0
            if gt_img.max() > 1.0:
                gt_img = gt_img / 255.0
            gt_img = torch.clamp(gt_img, 0, 1)
            # Resize GT to match target size
            if gt_img.shape[-2:] != (target_h, target_w):
                gt_img = F.interpolate(
                    gt_img.unsqueeze(0),
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            images_to_show.append(gt_img)

        # Ensure all images have the same size before creating grid
        for i, img in enumerate(images_to_show):
            if img.shape[-2:] != (target_h, target_w):
                images_to_show[i] = F.interpolate(
                    img.unsqueeze(0),
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

        # Create grid (horizontal layout: LQ | SR | GT)
        grid = make_grid(images_to_show, nrow=len(images_to_show), padding=10, pad_value=1.0)

        # Convert to numpy and save
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        grid_np = (grid_np * 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        if grid_np.shape[2] == 3:
            grid_bgr = grid_np[:, :, ::-1]
        else:
            grid_bgr = grid_np

        # Save image
        save_path = os.path.join(result_dir, f'iter_{current_iter:06d}.png')
        imwrite(grid_bgr, save_path)

        logger = get_root_logger()
        logger.info(f'Saved training result comparison to {save_path}')

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema') and not self.use_consistency:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter,
                            param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

