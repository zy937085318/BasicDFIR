import torch
import torch.nn.functional as F
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .flow_model import FlowModel


@MODEL_REGISTRY.register()
class MeanFlowModel(FlowModel):
    """Mean Flow Model for Super Resolution."""

    def __init__(self, opt):
        super(MeanFlowModel, self).__init__(opt)

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

    '''
    define the interpolation processing of flow-based method.
    input: time step t, optional r
    '''
    def flow_interpolation(self, t, r=None):
        if self.gt is None:
            raise ValueError('GT is required for flow interpolation')

        h, w = self.gt.shape[-2:]
        # Upsample LQ to GT size
        lq_up = F.interpolate(
            self.lq, size=(h, w), mode='bicubic', align_corners=False
        )

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

        self.xt = alpha_t * self.gt + sigma_t * lq_up
        self.vt_tar = d_alpha_t * self.gt + d_sigma_t * lq_up

        return self.xt

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
    def sample_image(self, lq=None, model=None, ema=False):
        """Sample image using MeanFlow ODE solver"""
        if lq is None:
            lq = self.lq

        # Determine which model to use
        if model is not None:
            model_to_use = model
        elif ema and hasattr(self, 'net_g_ema'):
            model_to_use = self.net_g_ema
        else:
            model_to_use = self.net_g

        # Get target size from scale
        scale = self.opt.get('scale', 4)
        h, w = lq.shape[-2:]
        target_size = (h * scale, w * scale)

        # Upsample LQ to target size as initial state (t=1)
        z = F.interpolate(
            lq, size=target_size, mode='bicubic', align_corners=False
        )

        # Handle fixed input size if required
        model_input_size = self.opt['network_g'].get('input_size', None)
        if model_input_size is not None:
            if z.shape[2] != model_input_size or z.shape[3] != model_input_size:
                start_h = (z.shape[2] - model_input_size) // 2
                start_w = (z.shape[3] - model_input_size) // 2
                end_h = start_h + model_input_size
                end_w = start_w + model_input_size
                z = z[:, :, start_h:end_h, start_w:end_w]

        # MeanFlow sampling loop
        batch_size = z.shape[0]
        device = self.device
        num_steps = getattr(self, 'num_sampling_steps', 10)  # Default 10 steps

        # Generate time steps: [1.0, 0.9, ..., 0.0]
        time_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        for i in range(num_steps):
            t_cur = time_steps[i]
            t_next = time_steps[i + 1]

            # Construct time tensors [B]
            t_tensor = torch.full((batch_size,), t_cur, device=device)
            r_tensor = torch.full((batch_size,), t_cur, device=device)  # r = t for sampling

            # Call model to predict velocity v
            model_input = (z, t_tensor, r_tensor)
            v = model_to_use(model_input)

            # Euler update: z_{next} = z_{curr} - dt * v
            dt = t_cur - t_next
            z = z - dt * v

        return z

    '''
    Add flow-based loss function.
    '''
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.flow_process()

        # MeanFlow uses JVP for training
        jvp_args = (
            lambda z, t, r: self.net_g((z, t, r)),
            (self.xt, self.t, self.r),
            (self.vt_tar, torch.ones_like(self.t), torch.zeros_like(self.r)),
        )

        u, dudt = torch.autograd.functional.jvp(*jvp_args, create_graph=True)
        t_map = self.t.view(-1, 1, 1, 1)
        r_map = self.r.view(-1, 1, 1, 1)

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

        # Debug output
        if current_iter % 100 == 0:
            logger = get_root_logger()
            for name, param in self.net_g.named_parameters():
                if param.grad is None:
                    continue
                if 'x_embedder' in name and 'weight' in name:
                    grad_mean = param.grad.abs().mean().item()
                    weight_mean = param.data.abs().mean().item()
                    logger.info(f'[Iter {current_iter}] Layer: {name} | Weight Mean: {weight_mean:.8f} | Grad Mean: {grad_mean:.8f}')
                    break



    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
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

            # tentative for out of GPU memory
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
                # calculate metrics
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
                # update the best metric result
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

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
