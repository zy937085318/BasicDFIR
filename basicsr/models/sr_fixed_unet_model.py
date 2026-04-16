import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import numpy as np
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img, makeborder, merge_with_padding, split_with_overlap
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_unet_model import SRUNetModel


@MODEL_REGISTRY.register()
class SRUFixedNetModel(SRUNetModel):
    """SR model with frozen backbone for single image super-resolution.

    This model freezes the backbone (net_g.backbone) parameters during training,
    allowing only the upsampling head to be trained.
    """

    def __init__(self, opt):
        super(SRUFixedNetModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)

        # Freeze backbone parameters if exists
        if hasattr(self.net_g, 'backbone'):
            logger = get_root_logger()
            # Print backbone weight statistics before freezing
            total_params = sum(p.numel() for p in self.net_g.backbone.parameters())
            mean_val = sum(p.mean().item() * p.numel() for p in self.net_g.backbone.parameters()) / total_params
            logger.info(f'Backbone weight stats before freezing - mean: {mean_val:.6f}, total_params: {total_params}')
            logger.info('Freezing backbone parameters...')
            for param in self.net_g.backbone.parameters():
                param.requires_grad = False

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        # if self.is_train:
        #     self.init_training_settings()

    # def init_training_settings(self):
    #     """Override to avoid parent class calling setup_optimizers before backbone is frozen."""
    #     self.net_g.train()
    #     train_opt = self.opt['train']
    #     self.ema_decay = train_opt.get('ema_decay', 0)
    #     if self.ema_decay > 0:
    #         logger = get_root_logger()
    #         logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
    #         self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
    #         load_path = self.opt['path'].get('pretrain_network_g', None)
    #         if load_path is not None:
    #             self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
    #         else:
    #             self.model_ema(0)
    #         self.net_g_ema.eval()

        # define losses
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
        """Set up optimizers that only optimize non-backbone parameters."""
        train_opt = self.opt['train']
        optim_params = []



        # Get all parameters that require grad (backbone is already frozen)
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # Filter out backbone parameters explicitly (backup safety)
        if hasattr(self.net_g, 'backbone'):
            backbone_params = set(self.net_g.backbone.parameters())
            optim_params = [p for p in optim_params if p not in backbone_params]

        logger = get_root_logger()
        logger.info(f'Number of trainable parameters (excluding backbone): {len(optim_params)}')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        """Optimize only non-backbone parameters."""
        self.optimizer_g.zero_grad()

        # Print backbone weight mean periodically
        # if hasattr(self.net_g, 'backbone') and current_iter % 1 == 0:
        #     backbone_mean = sum(p.mean().item() for p in self.net_g.backbone.parameters()) / sum(1 for _ in self.net_g.backbone.parameters())
        #     logger = get_root_logger()
        #     logger.info(f'Iter {current_iter} - Backbone weight mean: {backbone_mean:.6f}')

        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
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
