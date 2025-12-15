import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .flow_model import FlowModel
from .utils import *


@MODEL_REGISTRY.register()
class FlowmatchingModel(FlowModel):
    """Base SR model for single image super-resolution.
        self.t
        self.xt
        self.vt_pre
        self.vt_tar
        """

    def __init__(self, opt):
        super(FlowmatchingModel, self).__init__(opt)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        self.lq = torch.nn.functional.interpolate(self.lq, size=self.gt.shape[-2:], mode='bicubic')

    '''
    define the interpolation processing of flow-based method. Should be rewrite for differnet flow-based method.
    input:time step t
    '''
    def flow_interpolation(self):
        self.xt = (1 - self.tt) * self.lq + self.tt * self.gt

    '''
    define the timestep sampling method.
    '''
    def sample_timestep(self):
        self.t = torch.rand(self.gt.shape[0]).to(self.device)
        self.tt = expand_tensor_like(self.t, self.gt)


    '''
    main processing of flow.
    '''
    def flow_process(self):
        self.sample_timestep()
        self.flow_interpolation()

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
        print(self.net_g.blocks[0].mlp.fc1.weight.mean())
        self.flow_process()
        self.vt_pre = self.net_g((self.xt, self.t))
        l_total = 0
        loss_dict = OrderedDict()
        l_flow = torch.nn.functional.l1_loss(self.vt_pre, self.gt - self.lq)
        l_total += l_flow
        # flow loss
        # if self.flow_loss:
            # l_flow = self.flow_loss(self.vt_pre, self.gt-self.lq)
            # l_total += l_flow
            # loss_dict['l_flow'] = l_flow
        # # pixel loss
        # if self.cri_pix:
        #     l_pix = self.cri_pix(self.output, self.gt)
        #     l_total += l_pix
        #     loss_dict['l_pix'] = l_pix
        # # perceptual loss
        # if self.cri_perceptual:
        #     l_percep, l_style = self.cri_perceptual(self.output, self.gt)
        #     if l_percep is not None:
        #         l_total += l_percep
        #         loss_dict['l_percep'] = l_percep
        #     if l_style is not None:
        #         l_total += l_style
        #         loss_dict['l_style'] = l_style
        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    '''
    self.output is reconstructed image from sample_image method.
    '''
    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.sample_image(self.lq, model=self.net_g_ema)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.sample_image(self.lq, model=self.net_g)
            self.net_g.train()

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
