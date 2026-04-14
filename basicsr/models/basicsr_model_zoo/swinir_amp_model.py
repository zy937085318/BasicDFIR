import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_amp_model import SRAMPModel


@MODEL_REGISTRY.register()
class SwinIRAMPModel(SRAMPModel):

    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        # choose autocast context when AMP enabled
        if getattr(self, 'use_amp', False):
            autocast_ctx = torch.amp.autocast
        else:
            from contextlib import nullcontext
            autocast_ctx = nullcontext

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                with autocast_ctx('cuda'):
                    self.output = self.net_g_ema(img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                with autocast_ctx('cuda'):
                    self.output = self.net_g(img)
            self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
