import torch
import lpips
import numpy as np

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, crop_border: int = 0, input_order='HWC', test_y_channel=False, model='alex', **kwargs):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity).

    Args:
        img (ndarray): Ground-truth image. [0, 255], uint8
        img2 (ndarray): Predicted image. [0, 255], uint8
        crop_border (int): Crop border size for LPIPS. Default: 0.
        input_order (str): 'HWC' or 'CHW'. Default: 'HWC'.
        model (str): LPIPS model type: 'alex', 'vgg', or 'squeeze'
        test_y_channel (bool): If True, test on Y channel in YCrCb color space. Default: False.

    Returns:
        float: LPIPS score.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loss_fn = create_lpips_loss_fn(model, device)

    # Reorder and normalize
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float32) / 255. * 2 - 1
    img2 = img2.astype(np.float32) / 255. * 2 - 1

    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).to(device)
    img2 = torch.from_numpy(np.transpose(img2, (2, 0, 1))).unsqueeze(0).to(device)

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    return loss_fn(img, img2).item()


def create_lpips_loss_fn(model, device, supress_warning=True):
    if supress_warning:
        from os import devnull
        from contextlib import redirect_stdout, redirect_stderr
        with open(devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
            loss_fn = lpips.LPIPS(net=model, verbose=False).to(device)
    else:
        loss_fn = lpips.LPIPS(net=model, verbose=False).to(device)

    return loss_fn
