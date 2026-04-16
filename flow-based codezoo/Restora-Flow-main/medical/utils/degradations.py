# Parts of this file are adapted from https://github.com/annegnx/PnP-Flow

import torch


# Box inpainting helpers
def apply_rectangular_mask(x, mask_size):
    """
    Black rectangular mask of mask_size[0] x mask_size[1] pixels at the center of the image.

    :param x: Input tensor (shape: batch_size x channels x height x width)
    :param mask_size: Tuple or list containing (height, width) of the mask
    :return: Tensor with the mask applied
    """
    half_height_mask = mask_size[0] // 2
    half_width_mask = mask_size[1] // 2
    d_height = x.shape[2] // 2
    d_width = x.shape[3] // 2

    mask = torch.ones_like(x)
    mask[:, :, d_height - half_height_mask:d_height + half_height_mask,
         d_width - half_width_mask:d_width + half_width_mask] = 0

    return mask * x


# Superresolution helpers
def upsample(x, sf):
    """
    s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    """
    st = 0
    z = torch.zeros(
        (x.shape[0],
         x.shape[1],
         x.shape[2] *
         sf,
         x.shape[3] *
         sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf):
    """
    s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    """
    st = 0
    return x[..., st::sf, st::sf]


def create_downsampling_matrix(H, W, sf, device):
    assert H % sf == 0 and W % sf == 0, "Image dimensions must be divisible by sf"

    H_ds, W_ds = H // sf, W // sf  # Downsampled dimensions
    N = H * W  # Total number of pixels in the original image
    M = H_ds * W_ds  # Total number of pixels in the downsampled image

    # Initialize downsampling matrix of size (M, N)
    downsample_matrix = torch.zeros((M, N), device=device)

    # Fill the matrix with 1s at positions corresponding to downsampling
    for i in range(H_ds):
        for j in range(W_ds):
            # The index in the downsampled matrix
            downsampled_idx = i * W_ds + j

            # The corresponding index in the original flattened matrix
            original_idx = (i * sf * W) + (j * sf)

            # Set the value to 1 to perform downsampling
            downsample_matrix[downsampled_idx, original_idx] = 1

    return downsample_matrix


# Degradation classes
class Degradation:
    def H(self, x):
        raise NotImplementedError()

    def H_adj(self, x):
        raise NotImplementedError()


class Denoising(Degradation):
    def H(self, x):
        return x

    def H_adj(self, x):
        return x


class BoxInpainting(Degradation):
    def __init__(self, mask_size):
        super().__init__()
        self.mask_size = mask_size

    def H(self, x):
        return apply_rectangular_mask(x, self.mask_size)

    def H_adj(self, x):
        return apply_rectangular_mask(x, self.mask_size)


class Superresolution(Degradation):
    def __init__(self, sf, dim_image, device="cuda") -> None:
        super().__init__()
        self.sf = sf
        self.downsampling_matrix = create_downsampling_matrix(
            dim_image, dim_image, sf, device)

    def H(self, x):
        return downsample(x, self.sf)

    def H_adj(self, x):
        return upsample(x, self.sf)


class OcclusionsSimulation(Degradation):
    """
    Placeholder degradation class for occlusion simulation.

    Notes:
        - Occlusions are applied randomly per image during evaluation in `sampling_xray.py`.
        - Some images may have no occlusions.
        - Both `H` and `H_adj` are identity operations, since this class only marks the
          type of degradation rather than performing any actual transformation.
    """
    def H(self, x):
        return x

    def H_adj(self, x):
        return x
