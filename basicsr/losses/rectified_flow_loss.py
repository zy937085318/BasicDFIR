"""
Rectified Flow Loss Functions
"""
import torch
from torch import nn
from torch.nn import Module, functional as F

try:
    from einops import reduce
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False
    def reduce(tensor, pattern, reduction='mean'):
        if reduction == 'mean':
            return tensor.mean()
        elif reduction == 'sum':
            return tensor.sum()
        return tensor

try:
    import torchvision
    from torchvision.models import VGG16_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

from basicsr.utils.registry import LOSS_REGISTRY


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


@LOSS_REGISTRY.register()
class LPIPSLoss(Module):
    """LPIPS (Learned Perceptual Image Patch Similarity) Loss"""
    def __init__(
            self,
            vgg: Module | None = None,
            vgg_weights = None,
            loss_weight=1.0,
            reduction='mean',
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for LPIPSLoss")

        if not exists(vgg):
            if vgg_weights is None:
                try:
                    vgg_weights = VGG16_Weights.DEFAULT
                except:
                    vgg_weights = None
            vgg = torchvision.models.vgg16(weights=vgg_weights)
            vgg.classifier = nn.Sequential(*vgg.classifier[:-2])

        self.vgg = [vgg]

    def forward(self, pred_data, data, reduction=None):
        reduction = default(reduction, self.reduction)
        vgg, = self.vgg
        vgg = vgg.to(data.device)

        pred_embed, embed = map(vgg, (pred_data, data))

        loss = F.mse_loss(embed, pred_embed, reduction=reduction)

        if reduction == 'none':
            if EINOPS_AVAILABLE:
                loss = reduce(loss, 'b ... -> b', 'mean')
            else:
                loss = loss.view(loss.shape[0], -1).mean(dim=1)

        return loss * self.loss_weight


@LOSS_REGISTRY.register()
class PseudoHuberLoss(Module):
    """Pseudo Huber Loss"""
    def __init__(self, data_dim: int = 3, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.data_dim = data_dim
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, reduction=None, **kwargs):
        reduction = default(reduction, self.reduction)
        data_dim = default(self.data_dim, kwargs.pop('data_dim', None))

        c = .00054 * self.data_dim
        loss = (F.mse_loss(pred, target, reduction=reduction) + c * c).sqrt() - c

        if reduction == 'none':
            if EINOPS_AVAILABLE:
                loss = reduce(loss, 'b ... -> b', 'mean')
            else:
                loss = loss.view(loss.shape[0], -1).mean(dim=1)

        return loss * self.loss_weight


@LOSS_REGISTRY.register()
class PseudoHuberLossWithLPIPS(Module):
    """Pseudo Huber Loss combined with LPIPS"""
    def __init__(self, data_dim: int = 3, lpips_kwargs: dict = dict(), loss_weight=1.0):
        super().__init__()
        self.pseudo_huber = PseudoHuberLoss(data_dim)
        self.lpips = LPIPSLoss(**lpips_kwargs)
        self.loss_weight = loss_weight

    def forward(self, pred_flow, target_flow, *, pred_data, times, data):
        huber_loss = self.pseudo_huber(pred_flow, target_flow, reduction='none')
        lpips_loss = self.lpips(data, pred_data, reduction='none')

        time_weighted_loss = huber_loss * (1 - times) + lpips_loss * (1. / times.clamp(min=1e-1))
        return time_weighted_loss.mean() * self.loss_weight


# Note: MSELoss is already registered in basic_loss.py, so we don't register it again here
# Use the existing MSELoss from basic_loss.py instead

@LOSS_REGISTRY.register()
class MeanVarianceNetLoss(Module):
    """Loss for mean-variance network"""
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        from torch.distributions import Normal
        dist = Normal(*pred)
        return -dist.log_prob(target).mean() * self.loss_weight

