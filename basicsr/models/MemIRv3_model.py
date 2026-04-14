from asyncio.constants import SSL_SHUTDOWN_TIMEOUT
from collections import OrderedDict
import copy
from os import path as osp
import numpy as np
from tqdm import tqdm
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils.img_util import imwrite, makeborder, tensor2img
from basicsr.utils.logger import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.archs import build_network
from basicsr.models.base_model import BaseModel#, SRModel
from basicsr.archs.swin_decoder_arch import SwinUNetDecoder #UNetDecoder
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.path.scheduler import CondOTScheduler
from .utils import *
from basicsr.utils.image_split import split_with_overlap, merge_with_padding
import torch
from basicsr.archs.dinov3.convnext_encoder import get_convnext_arch


class united_model(torch.nn.Module):
    def __init__(self, encoder_lr, decoder):
        super(united_model, self).__init__()
        self.encoder_lr = encoder_lr
        self.decoder = decoder

    def forward(self, x):
        feat_lr = self.encoder_lr.get_intermediate_layers(x, n=4, reshape=True, norm=False)
        output = self.decoder(feat_lr)
        # print(f'[x forward] x min: {x.min().item():.6f}, max: {x.max().item():.6f}')
        # print(f'[united_model forward] feat_lr min: {feat_lr.min().item():.6f}, max: {feat_lr.max().item():.6f}')
        # print(f'[united_model forward] output min: {output.min().item():.6f}, max: {output.max().item():.6f}')
        # exit()
        return output
    
@MODEL_REGISTRY.register() # type: ignore
class MemIRv3Model(BaseModel):
    def __init__(self, opt):
        super(MemIRv3Model, self).__init__(opt)
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.sample_step = None #opt['val']['sample_step']#采样步数
        self.decoder = SwinUNetDecoder() #build_network(opt['network_g'])
        self.decoder = self.model_to_device(self.decoder)
        self.encoder_lr = get_convnext_arch(opt['encoder_lr']['model_type'], opt['scale'])()
        self.encoder_lr = self.model_to_device(self.encoder_lr)
        self.encoder_hr = get_convnext_arch(opt['encoder_hr']['model_type'], opt['scale'])()
        self.encoder_hr = self.model_to_device(self.encoder_hr)
        for param in self.encoder_lr.parameters():
                param.requires_grad = False
        for param in self.encoder_hr.parameters():
                param.requires_grad = False  
        self.net_g = united_model(self.encoder_lr, self.decoder)
        if self.is_train:
            self.init_training_settings()
            
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = copy.deepcopy(self.net_g)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()
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
        train_opt = self.opt['train']
        optim_params = []
        print(optim_params)
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
    
        

    @torch.no_grad()
    def update_ema(self):
        for p_ema, p_net in zip(self.net_g_ema.parameters(), self.net_g.parameters()):
            p_ema.data.mul_(self.ema_decay).add_(p_net.data, alpha=1 - self.ema_decay)
        for p_ema, p_net in zip(self.net_g_ema.buffers(), self.net_g.buffers()):
            p_ema.data.copy_(p_net.data)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.lq_bicubic = torch.nn.functional.interpolate(self.lq, size=self.gt.shape[-2:], mode='bicubic').clamp(0, 1)
        self.lq_bicubic = self.lq_bicubic.clamp(0, 1)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.net_g
        self.output = self.net_g(self.lq_bicubic)
        # print(self.net_g.decoder.conv_first.weight.mean())
        # if current_iter % 100 == 0:
        #     gt = self.gt[0,...].permute(1,2,0).cpu().numpy()*255
        #     lq = self.lq_bicubic[0,...].permute(1,2,0).cpu().numpy()*255
        #     sr = self.output[0,...].permute(1,2,0).detach().cpu().numpy()*255
        #     imwrite(np.hstack((makeborder(lq), makeborder(sr), makeborder(gt))), f'./debug/{current_iter}.png')
        
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

    '''
    采样一张图
    '''
    def sample(self, lq_bicubic, sample_step=5):
        # T = torch.linspace(0, 1, sample_step).to(self.device)
        # x_0 = torch.randn_like(lq_bicubic).to(self.device)
        # if hasattr(self, 'net_g_ema'):
        #     self.net_g_ema.eval()
        #     net_g = self.net_g_ema
        # else:
        #     net_g = self.net_g.eval()
        # cnf = self.cnf(net_g, lq_bicubic)
        # solver = ODESolver(velocity_model=cnf)
        # x_1_pred = solver.sample(time_grid=T, x_init=x_0, method='midpoint', step_size=sample_step,
        #                          return_intermediates=False)
        # return x_1_pred
        return self.net_g(lq_bicubic)

    '''
        ODE Solver的包装器
    '''

    class cnf(torch.nn.Module):

        def __init__(self, model, lr_bicubic):
            super().__init__()
            self.model = model
            self.lr_bicubic = lr_bicubic

        def forward(self, t, x):#此处t为一个size=[]的tensor
            with torch.no_grad():
                x_1_pred = self.model(torch.cat([x, self.lr_bicubic], dim=1), None, t.repeat(x.shape[0]))
                # print(t.size())
                # print(t.dtype)
                # print(t)
                # if t.size == []:
                #     print('hhhh')
                #     t = t.unsqueeze(0)
                # t_ = t[:, None, None, None].expand(-1, 3, -1, -1)
                v_t_pred = (x_1_pred - x)/(1-t) #.clamp_min(5e-2))
            return v_t_pred

    def test(self):
        with torch.no_grad():
            img = self.lq_bicubic
            split_h, split_w = 256, 256 #img.shape[-2], img.shape[-1]
            # split_h = self.opt['network_g']['img_size']
            # split_w = self.opt['network_g']['img_size']
            split_img, padding_info = split_with_overlap(img, split_h, split_w, split_h, split_w, return_padding=True)
            [_, n_h, n_w, _, _, _] = split_img.shape
            for i in range(n_h):
                for j in range(n_w):
                    split_img [:, i, j, ...] = self.sample(split_img [:, i, j, ...], self.sample_step)
            merged_img = merge_with_padding(split_img, padding_info)
        self.net_g.train()
        self.output = merged_img.clamp(0, 1)

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
            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, img_name,
                                             f'{img_name}_{current_iter}.png')
                    lq_img = tensor2img(torch.nn.functional.interpolate(self.lq, sr_img.shape[:2], mode='bicubic'))
                    sr_img = np.hstack((makeborder(lq_img), makeborder(sr_img), makeborder(gt_img)))


                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)
            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()
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