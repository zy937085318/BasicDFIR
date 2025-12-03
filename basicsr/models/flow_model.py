import inspect
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
        pass

    '''
    define the timestep sampling method.
    '''
    def sample_timestep(self):
        return self.t

    '''
    main processing of flow.
    '''
    def flow_process(self):
        self.sample_timestep()
        self.xt = self.flow_interpolation(self.t)

    '''
    sample image with flow-based ODE.
    '''
    def sample_image(self, ema=False):
        srimage = None
        return srimage

    '''
    Add flow-based loss function.
    '''
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.flow_process()
        self.vt_pre = self.net_g(self.xt)
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
    '''
    def test(self):
        """Test method for flow-based models with tile-based processing.

        This method processes images in tiles to handle large images efficiently,
        especially suitable for SET14 dataset. It handles different EMA scenarios:
        1. Regular EMA model (net_g_ema)
        2. Consistency flow matching EMA model (ema_model.ema_model)
        3. Standard model (net_g)

        The tile size is optimized for SET14 dataset (200 pixels per tile).
        """
        _, C, h, w = self.lq.size()

        # Tile size optimized for SET14 dataset
        # SET14 images are typically 200-800 pixels, so 200 pixels per tile is appropriate
        tile_size = 200
        split_token_h = h // tile_size + 1  # number of horizontal cut sections
        split_token_w = w // tile_size + 1  # number of vertical cut sections

        # Padding
        mod_pad_h, mod_pad_w = 0, 0
        if h % split_token_h != 0:
            mod_pad_h = split_token_h - h % split_token_h
        if w % split_token_w != 0:
            mod_pad_w = split_token_w - w % split_token_w

        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, H, W = img.size()

        split_h = H // split_token_h  # height of each partition
        split_w = W // split_token_w  # width of each partition

        # Overlapping to avoid boundary artifacts
        shave_h = split_h // 10
        shave_w = split_w // 10

        scale = self.opt.get('scale', 1)
        ral = H // split_h
        row = W // split_w

        # Generate slice indices for each tile
        slices = []  # list of partition borders
        for i in range(ral):
            for j in range(row):
                if i == 0 and i == ral - 1:
                    top = slice(i * split_h, (i + 1) * split_h)
                elif i == 0:
                    top = slice(i * split_h, (i + 1) * split_h + shave_h)
                elif i == ral - 1:
                    top = slice(i * split_h - shave_h, (i + 1) * split_h)
                else:
                    top = slice(i * split_h - shave_h, (i + 1) * split_h + shave_h)

                if j == 0 and j == row - 1:
                    left = slice(j * split_w, (j + 1) * split_w)
                elif j == 0:
                    left = slice(j * split_w, (j + 1) * split_w + shave_w)
                elif j == row - 1:
                    left = slice(j * split_w - shave_w, (j + 1) * split_w)
                else:
                    left = slice(j * split_w - shave_w, (j + 1) * split_w + shave_w)

                temp = (top, left)
                slices.append(temp)

        # Extract image tiles
        img_chops = []  # list of partitions
        for temp in slices:
            top, left = temp
            img_chops.append(img[..., top, left])

        # Determine which model to use
        # Check if using consistency flow matching with EMA model
        if hasattr(self, 'use_consistency') and self.use_consistency and hasattr(self, 'ema_model'):
            model_to_use = self.ema_model.ema_model
            use_ema = True
        # Check if using regular EMA model
        elif hasattr(self, 'net_g_ema'):
            # Check if consistency is disabled (to avoid using net_g_ema when consistency is enabled)
            if not (hasattr(self, 'use_consistency') and self.use_consistency):
                model_to_use = self.net_g_ema
                use_ema = True
            else:
                # Consistency is enabled, use regular model
                model_to_use = self.net_g
                use_ema = False
        # Use standard model
        else:
            model_to_use = self.net_g
            use_ema = False

        # Set model to eval mode
        model_to_use.eval()

        # Check method signature to determine how to call sample_image
        sample_image_sig = inspect.signature(self.sample_image)
        has_lq_param = 'lq' in sample_image_sig.parameters
        has_model_param = 'model' in sample_image_sig.parameters
        use_sample_image = has_lq_param and has_model_param

        with torch.no_grad():
            outputs = []
            for chop in img_chops:
                # Process each tile
                if use_sample_image:
                    # Use sample_image method (for flow-based models like RectifiedFlow)
                    out = self.sample_image(lq=chop, model=model_to_use)
                else:
                    # Direct network forward pass (for simpler models)
                    out = model_to_use(chop)
                outputs.append(out)

            # Merge tiles
            _img = torch.zeros(1, C, H * scale, W * scale, device=self.device, dtype=outputs[0].dtype)

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

                    _img[..., top, left] = outputs[i * row + j][..., _top, _left]

            self.output = _img

        # Remove padding
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

        # Restore training mode if using regular model
        if not use_ema:
            self.net_g.train()

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
                del self.gt

            # tentative for out of GPU memory
            del self.xt
            del self.vt_pre
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
