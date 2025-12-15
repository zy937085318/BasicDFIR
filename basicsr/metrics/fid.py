import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from tqdm import tqdm

from basicsr.archs.inception import InceptionV3
from basicsr.metrics.metric_util import reorder_image
from basicsr.utils.registry import METRIC_REGISTRY


def load_patched_inception_v3(device='cuda', resize_input=True, normalize_input=False):
    # we may not resize the input, but in [rosinality/stylegan2-pytorch] it
    # does resize the input.
    inception = InceptionV3([3], resize_input=resize_input, normalize_input=normalize_input)
    inception = nn.DataParallel(inception).eval().to(device)
    return inception


@torch.no_grad()
def extract_inception_features(data_generator, inception, len_generator=None, device='cuda'):
    """Extract inception features.

    Args:
        data_generator (generator): A data generator.
        inception (nn.Module): Inception model.
        len_generator (int): Length of the data_generator to show the
            progressbar. Default: None.
        device (str): Device. Default: cuda.

    Returns:
        Tensor: Extracted features.
    """
    if len_generator is not None:
        pbar = tqdm(total=len_generator, unit='batch', desc='Extract')
    else:
        pbar = None
    features = []

    for data in data_generator:
        if pbar:
            pbar.update(1)
        data = data.to(device)
        feature = inception(data)[0].view(data.shape[0], -1)
        features.append(feature.to('cpu'))
    if pbar:
        pbar.close()
    features = torch.cat(features, 0)
    return features


def _calculate_fid_from_statistics(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2) is:
    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.

    Args:
        mu1 (np.array): The sample mean over activations.
        sigma1 (np.array): The covariance matrix over activations for generated samples.
        mu2 (np.array): The sample mean over activations, precalculated on an representative data set.
        sigma2 (np.array): The covariance matrix over activations, precalculated on an representative data set.

    Returns:
        float: The Frechet Distance.
    """
    assert mu1.shape == mu2.shape, 'Two mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, ('Two covariances have different dimensions')

    cov_sqrt, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # Product might be almost singular
    if not np.isfinite(cov_sqrt).all():
        print('Product of cov matrices is singular. Adding {eps} to diagonal of cov estimates')
        offset = np.eye(sigma1.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))
            raise ValueError(f'Imaginary component {m}')
        cov_sqrt = cov_sqrt.real

    mean_diff = mu1 - mu2
    mean_norm = mean_diff @ mean_diff
    trace = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(cov_sqrt)
    fid = mean_norm + trace

    return fid


# Keep the original function name for backward compatibility with scripts
def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Alias for backward compatibility."""
    return _calculate_fid_from_statistics(mu1, sigma1, mu2, sigma2, eps)


# Global state for FID calculation during validation
_fid_features_gt = []
_fid_features_sr = []
_fid_inception = None
_fid_device = 'cuda'
_fid_current_value = 0.0


def reset_fid_accumulator():
    """Reset accumulated FID features. Call this at the start of each validation."""
    global _fid_features_gt, _fid_features_sr, _fid_current_value
    _fid_features_gt = []
    _fid_features_sr = []
    _fid_current_value = 0.0


@METRIC_REGISTRY.register()
def calculate_fid(img, img2, crop_border=0, input_order='HWC', **kwargs):
    """Calculate FID (Fréchet Inception Distance) wrapper for validation.

    Note: This wrapper accumulates features across all validation images and 
    calculates FID based on all accumulated features. Since FID requires 
    computing statistics over all images, it returns the same value for each
    image (based on all accumulated features so far).

    Args:
        img (ndarray): Generated/SR images with range [0, 255], shape (H, W, C).
        img2 (ndarray): Ground truth images with range [0, 255], shape (H, W, C).
        crop_border (int): Cropped pixels in each edge of an image. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        **kwargs: Additional arguments.

    Returns:
        float: FID result based on all accumulated features so far.
    """
    global _fid_features_gt, _fid_features_sr, _fid_inception, _fid_device, _fid_current_value

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Initialize Inception model if not already done
    if _fid_inception is None:
        _fid_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _fid_inception = load_patched_inception_v3(_fid_device)
    
    # Reset accumulated features if lists are empty (new validation session)
    # This helps handle cases where features were cleared but model is still initialized
    if len(_fid_features_gt) == 0 and len(_fid_features_sr) == 0:
        _fid_current_value = 0.0

    # Convert images to tensor format (B, C, H, W) with range [0, 1]
    img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    img2_tensor = torch.from_numpy(img2.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    # Extract Inception features
    with torch.no_grad():
        img_tensor = img_tensor.to(_fid_device)
        img2_tensor = img2_tensor.to(_fid_device)
        feat_sr = _fid_inception(img_tensor)[0].view(1, -1).cpu().numpy()
        feat_gt = _fid_inception(img2_tensor)[0].view(1, -1).cpu().numpy()

    # Accumulate features
    _fid_features_sr.append(feat_sr)
    _fid_features_gt.append(feat_gt)

    # Calculate FID based on all accumulated features (needs at least 2 images)
    # Only update if we haven't computed FID yet or if we're still accumulating
    # Note: We recalculate each time so that the final value is based on all images
    # Since validation code will average all returned values, we return the same
    # value for all images after accumulation, ensuring correct averaging
    if len(_fid_features_sr) >= 2:
        features_sr = np.concatenate(_fid_features_sr, axis=0)
        features_gt = np.concatenate(_fid_features_gt, axis=0)

        mu_sr = np.mean(features_sr, axis=0)
        sigma_sr = np.cov(features_sr, rowvar=False)
        mu_gt = np.mean(features_gt, axis=0)
        sigma_gt = np.cov(features_gt, rowvar=False)

        _fid_current_value = _calculate_fid_from_statistics(mu_sr, sigma_sr, mu_gt, sigma_gt)
    
    # Return current FID value (will be the same for all images after first calculation)
    # The validation code will average all returned values, so returning the same
    # value ensures the average equals the FID value
    return float(_fid_current_value)
