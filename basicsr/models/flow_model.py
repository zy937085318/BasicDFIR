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
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class FlowModel(SRModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(FlowModel, self).__init__(opt)
        self.t = None
        self.xt = None
        self.vt_pre = None
        self.vt_tar = None

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

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    '''
    define the interpolation processing of flow-based method. Should be instanced for differnet flow-based method.
    input:time step t
    '''
    def flow_interpolation(self, t, r=None):
        """Default flow interpolation implementation.
        This is a simple linear interpolation: x_t = (1-t) * x_1 + t * x_0
        Subclasses should override this method for specific flow-based methods.
        """
        if self.gt is None:
            raise ValueError('GT is required for flow interpolation')

        batch_size = self.lq.shape[0]
        device = self.device

        # Sample time step if not provided
        if t is None or (isinstance(t, torch.Tensor) and t.numel() == 0):
            # Uniform sampling in [0, 1]
            self.t = torch.rand(batch_size, device=device)
        else:
            self.t = t if isinstance(t, torch.Tensor) else torch.tensor(t, device=device)
            if self.t.numel() == 1:
                self.t = self.t.expand(batch_size)

        # Upsample LQ to GT size
        h, w = self.gt.shape[-2:]
        lq_up = F.interpolate(
            self.lq, size=(h, w), mode='bicubic', align_corners=False
        )

        # Linear interpolation: x_t = (1-t) * x_1 + t * x_0
        # where x_0 = lq_up (low quality), x_1 = gt (high quality)
        t_map = self.t.view(batch_size, 1, 1, 1)
        self.xt = (1 - t_map) * self.gt + t_map * lq_up

        # Target velocity: v_t = x_1 - x_0 = gt - lq_up
        self.vt_tar = self.gt - lq_up

        return self.xt

    '''
    define the timestep sampling method.
    '''
    def sample_timestep(self):
        """Sample time step. Returns self.t if already set, otherwise None."""
        return self.t

    '''
    main processing of flow.
    '''
    def flow_process(self):
        """Main flow processing: sample timestep and perform interpolation."""
        # flow_interpolation will handle time sampling if self.t is None
        self.xt = self.flow_interpolation(self.t)

    '''
    sample image with flow-based ODE.
    '''
    def sample_image(self, lq=None, model=None, ema=False):
        """Sample image using flow-based forward pass"""
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

        # Upsample LQ to target size
        lq_upsampled = F.interpolate(
            lq, size=target_size, mode='bicubic', align_corners=False
        )

        # Get downsample_factor from model (default 8 for FlowUNet)
        if hasattr(model_to_use, 'downsample_factor'):
            downsample_factor = model_to_use.downsample_factor
        else:
            downsample_factor = 8

        # Pad to be divisible by downsample_factor
        _, _, h_up, w_up = lq_upsampled.shape
        pad_h = (downsample_factor - h_up % downsample_factor) % downsample_factor
        pad_w = (downsample_factor - w_up % downsample_factor) % downsample_factor

        if pad_h > 0 or pad_w > 0:
            lq_padded = F.pad(lq_upsampled, (0, pad_w, 0, pad_h), mode='reflect')
            original_h, original_w = h_up, w_up
        else:
            lq_padded = lq_upsampled
            original_h, original_w = h_up, w_up

        # Use t=1 to predict flow at final time step
        batch_size = lq_padded.shape[0]
        times = torch.ones(batch_size, device=self.device)

        # Predict flow: v = net_g(x_t, t)
        vt_pre = model_to_use(lq_padded, times)

        # Apply flow: x_1 = x_0 + v
        output_padded = lq_padded + vt_pre

        # Crop back to original size if padding was applied
        if pad_h > 0 or pad_w > 0:
            output = output_padded[:, :, :original_h, :original_w]
        else:
            output = output_padded

        # Clamp to valid range [0, 1]
        output = torch.clamp(output, 0.0, 1.0)

        return output

    '''
    Add flow-based loss function.
    '''
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.flow_process()

        # Prepare time conditioning for network
        # FlowUNet requires times as the second positional argument
        times = self.t
        if times is not None:
            # Ensure times is 1D tensor with shape [batch_size]
            if times.dim() > 1:
                times = times.flatten()
            if times.numel() == 1:
                # Expand to batch size if single value
                batch_size = self.xt.shape[0]
                times = times.expand(batch_size)
            self.vt_pre = self.net_g(self.xt, times)
        else:
            raise ValueError('Time step self.t is None. flow_process() should set self.t.')

        # Compute predicted data for loss computation
        # self.output is the predicted high-resolution image
        # Formula: output = xt + vt_pre * (1 - t)
        # Reshape times to [batch_size, 1, 1, 1] for broadcasting
        if times.dim() == 1:
            t_view = times.view(-1, 1, 1, 1)
        else:
            t_view = times.flatten().view(-1, 1, 1, 1)
        self.output = self.xt + self.vt_pre * (1. - t_view)

        l_total = 0
        loss_dict = OrderedDict()
        # flow loss
        if self.flow_loss:
            l_flow = self.flow_loss(self.output, self.gt)
            l_total += l_flow
            loss_dict['l_flow'] = l_flow
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
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
    self.output is reconstructed image from sample_image method.
    Tile-based processing for large images to manage memory.
    '''
    def test(self):
        _, C, h, w = self.lq.size()
        scale = self.opt.get('scale', 4)

        # Check if model has fixed input size (e.g., MeanFlow with DiT)
        model_input_size = self.opt['network_g'].get('input_size', None)

        # For DiT-like architectures (has x_embedder), avoid outer tiling/cropping.
        # Delegate to sample_image, which handles DiT fixed input size and padding.
        if hasattr(self.net_g, 'x_embedder'):
            if hasattr(self, 'net_g_ema'):
                model_to_use = self.net_g_ema
                model_to_use.eval()
                with torch.no_grad():
                    self.output = self.sample_image(self.lq, model=model_to_use)
                model_to_use.train()
            else:
                self.net_g.eval()
                with torch.no_grad():
                    self.output = self.sample_image(self.lq, model=self.net_g)
                self.net_g.train()
            return

        if model_input_size is not None:
            # For models with fixed input size, adjust chunk size to match
            # For scale=4 and input_size=128, each LR chunk should be 32x32
            chunk_size_lr = model_input_size // scale
            split_token_h = max(1, h // chunk_size_lr)
            split_token_w = max(1, w // chunk_size_lr)
        else:
            # Default chunk size
            split_token_h = h // 200 + 1  # number of horizontal cut sections
            split_token_w = w // 200 + 1  # number of vertical cut sections

        # padding
        mod_pad_h, mod_pad_w = 0, 0
        if h % split_token_h != 0:
            mod_pad_h = split_token_h - h % split_token_h
        if w % split_token_w != 0:
            mod_pad_w = split_token_w - w % split_token_w

        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, H, W = img.size()

        if model_input_size is not None:
            # Use fixed chunk size for models with fixed input
            split_h = chunk_size_lr
            split_w = chunk_size_lr
            # No overlapping for fixed input size models to ensure exact size match
            shave_h = 0
            shave_w = 0
        else:
            split_h = H // split_token_h  # height of each partition
            split_w = W // split_token_w  # width of each partition
            # overlapping
            shave_h = split_h // 10
            shave_w = split_w // 10

        ral = H // split_h
        row = W // split_w

        slices = []  # list of partition borders
        for i in range(ral):
            for j in range(row):
                if i == 0 and i == ral - 1:
                    top = slice(i * split_h, (i + 1) * split_h)
                elif i == 0:
                    top = slice(i*split_h, (i+1)*split_h+shave_h)
                elif i == ral - 1:
                    top = slice(i*split_h-shave_h, (i+1)*split_h)
                else:
                    top = slice(i*split_h-shave_h, (i+1)*split_h+shave_h)

                if j == 0 and j == row - 1:
                    left = slice(j*split_w, (j+1)*split_w)
                elif j == 0:
                    left = slice(j*split_w, (j+1)*split_w+shave_w)
                elif j == row - 1:
                    left = slice(j*split_w-shave_w, (j+1)*split_w)
                else:
                    left = slice(j*split_w-shave_w, (j+1)*split_w+shave_w)

                temp = (top, left)
                slices.append(temp)

        img_chops = []  # list of partitions
        for temp in slices:
            top, left = temp
            img_chops.append(img[..., top, left])

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    # Use sample_image for flow-based models
                    out = self.sample_image(chop, model=self.net_g_ema)
                    # Ensure output has batch dimension [B, C, H, W]
                    if out.dim() == 3:
                        out = out.unsqueeze(0)
                    outputs.append(out)

                _img = torch.zeros(1, C, H * scale, W * scale, device=self.device)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h*scale, (shave_h+split_h)*scale)
                        if j == 0:
                            _left = slice(0, split_w*scale)
                        else:
                            _left = slice(shave_w*scale, (shave_w+split_w)*scale)

                        output_chunk = outputs[i * row + j]
                        # Ensure output_chunk has correct shape [1, C, H, W]
                        if output_chunk.dim() == 4 and output_chunk.shape[0] == 1:
                            _img[..., top, left] = output_chunk[..., _top, _left]
                        else:
                            # Handle unexpected shapes
                            if output_chunk.dim() == 3:
                                output_chunk = output_chunk.unsqueeze(0)
                            _img[..., top, left] = output_chunk[0:1, ..., _top, _left]

                self.output = _img
        else:
            self.net_g.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    # Use sample_image for flow-based models
                    out = self.sample_image(chop, model=self.net_g)
                    # Ensure output has batch dimension [B, C, H, W]
                    if out.dim() == 3:
                        out = out.unsqueeze(0)
                    outputs.append(out)

                _img = torch.zeros(1, C, H * scale, W * scale, device=self.device)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                        if j == 0:
                            _left = slice(0, split_w * scale)
                        else:
                            _left = slice(shave_w * scale, (shave_w + split_w) * scale)

                        output_chunk = outputs[i * row + j]
                        # Ensure output_chunk has correct shape [1, C, H, W]
                        if output_chunk.dim() == 4 and output_chunk.shape[0] == 1:
                            _img[..., top, left] = output_chunk[..., _top, _left]
                        else:
                            # Handle unexpected shapes
                            if output_chunk.dim() == 3:
                                output_chunk = output_chunk.unsqueeze(0)
                            _img[..., top, left] = output_chunk[0:1, ..., _top, _left]

                self.output = _img
            self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    # def test_selfensemble(self):
    #     # TODO: to be tested
    #     # 8 augmentations
    #     # modified from https://github.com/thstkdgus35/EDSR-PyTorch
    #
    #     def _transform(v, op):
    #         # if self.precision != 'single': v = v.float()
    #         v2np = v.data.cpu().numpy()
    #         if op == 'v':
    #             tfnp = v2np[:, :, :, ::-1].copy()
    #         elif op == 'h':
    #             tfnp = v2np[:, :, ::-1, :].copy()
    #         elif op == 't':
    #             tfnp = v2np.transpose((0, 1, 3, 2)).copy()
    #
    #         ret = torch.Tensor(tfnp).to(self.device)
    #         # if self.precision == 'half': ret = ret.half()
    #
    #         return ret
    #
    #     # prepare augmented data
    #     lq_list = [self.lq]
    #     for tf in 'v', 'h', 't':
    #         lq_list.extend([_transform(t, tf) for t in lq_list])
    #
    #     # inference
    #     if hasattr(self, 'net_g_ema'):
    #         self.net_g_ema.eval()
    #         with torch.no_grad():
    #             out_list = [self.net_g_ema(aug) for aug in lq_list]
    #     else:
    #         self.net_g.eval()
    #         with torch.no_grad():
    #             out_list = [self.net_g_ema(aug) for aug in lq_list]
    #         self.net_g.train()
    #
    #     # merge results
    #     for i in range(len(out_list)):
    #         if i > 3:
    #             out_list[i] = _transform(out_list[i], 't')
    #         if i % 4 > 1:
    #             out_list[i] = _transform(out_list[i], 'h')
    #         if (i % 4) % 2 == 1:
    #             out_list[i] = _transform(out_list[i], 'v')
    #     output = torch.cat(out_list, dim=0)
    #
    #     self.output = output.mean(dim=0, keepdim=True)

    '''
    added code to delete flow-based attribution self.v_pred et.al.
    '''
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
                if hasattr(self, 'gt'):
                    del self.gt

            # tentative for out of GPU memory
            if hasattr(self, 'xt'):
                del self.xt
            if hasattr(self, 'vt_pre'):
                del self.vt_pre
            if hasattr(self, 'vt_tar'):
                del self.vt_tar
            if hasattr(self, 'lq'):
                del self.lq
            if hasattr(self, 'output'):
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
