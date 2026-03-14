# Parts of this file are adapted from https://github.com/annegnx/PnP-Flow

import torch
import numpy as np


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


def apply_random_mask(x, p):
    """
    Random mask on x
    """
    np.random.seed(42)
    mask = torch.from_numpy(np.random.binomial(n=1, p=1-p, size=(
        x.shape[0], x.shape[2], x.shape[3]))).to(x.device)

    return mask.unsqueeze(1) * x


def apply_predefined_mask(x, mask_type="central_block"):
    """
    Apply one of 8 predefined mask types to the input tensor.

    :param x: Input tensor of shape (B, C, H, W)
    :param mask_type: One of the predefined types:
                      ['central_block', 'random_patches', 'half_image', 'diagonal',
                       'grid', 'random_blobs', 'random_pixel', 'periphery']
    :return: Masked tensor
    """
    B, C, H, W = x.shape
    dh, dw = H // 2, W // 2
    mask = torch.ones_like(x)

    if mask_type == "central_block":
        # Mask 60x60 block at center
        h, w = 60, 60
        mask[:, :, dh - h // 2:dh + h // 2, dw - w // 2:dw + w // 2] = 0

    elif mask_type == "random_patches":
        # Use fixed patch positions (4 fixed 20x20 patches)
        patch_coords = [(10, 10), (50, 20), (70, 60), (30, 80)]

        for top, left in patch_coords:
            mask[:, :, top:top + 20, left:left + 20] = 0

    elif mask_type == "half_image":
        # Mask left half of the image
        mask[:, :, :, W // 2:] = 0

    elif mask_type == "diagonal":
        # Diagonal band of width 30
        Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        base_mask = np.ones((H, W), dtype=np.uint8)
        base_mask[np.abs(X - Y) < 15] = 0  # 30-pixel-wide diagonal
        base_mask = np.tile(base_mask, (B * C, 1, 1))
        mask = torch.from_numpy(base_mask).view(B, C, H, W).to(x.device)

    elif mask_type == "grid":
        # Checkerboard 16x16 grid
        for i in range(0, H, 16):
            for j in range(0, W, 16):
                if (i // 16 + j // 16) % 2 == 0:
                    mask[:, :, i:i + 16, j:j + 16] = 0

    elif mask_type == "random_lines":
        base_mask = np.ones((H, W), dtype=np.uint8) * 255  # Start white (255)

        lines = [
            (50, 50, 70, 70),
            (55, 65, 80, 60),
            (60, 40, 85, 55),
            (45, 75, 65, 85),
            (40, 64, 90, 64),  # horizontal center line
            (64, 40, 64, 90),  # vertical center line
            (50, 90, 70, 70),
            (55, 55, 75, 65),
            (60, 60, 80, 80),
            (65, 50, 85, 70),
            (70, 65, 90, 85),
            (55, 40, 75, 55),
            (40, 50, 60, 70),
            (50, 55, 70, 75),
            (60, 65, 80, 85),
            (45, 60, 65, 80),
            (50, 64, 90, 64),  # extra horizontal center line
            (64, 50, 64, 90),  # extra vertical center line
        ]

        for (x1, y1, x2, y2) in lines:
            cv2.line(base_mask, (x1, y1), (x2, y2), color=0, thickness=5)

        # Normalize mask to 1 (keep), 0 (mask)
        base_mask = base_mask.astype(np.float32) / 255.0  # lines = 0, rest = 1

        # Convert to torch tensor, expand dims for batch and channels
        mask = torch.from_numpy(base_mask).unsqueeze(0).unsqueeze(0)  # shape (1,1,H,W)
        mask = mask.expand(B, C, H, W).to(x.device)

    elif mask_type == "random_blobs":
        # Use 3 fixed blobs at fixed positions and radii
        blob_specs = [(40, 40, 15), (90, 20, 12), (64, 100, 10)]
        base_mask = np.ones((H, W), dtype=np.uint8)
        Y, X = np.ogrid[:H, :W]

        for x0, y0, r in blob_specs:
            dist = (X - x0) ** 2 + (Y - y0) ** 2
            base_mask[dist <= r ** 2] = 0

        base_mask = np.tile(base_mask, (B * C, 1, 1))
        mask = torch.from_numpy(base_mask).view(B, C, H, W).to(x.device)

    elif mask_type == "random_pixel":
        # Mask 30% of pixels at random
        prob_mask = torch.rand((B, C, H, W), device=x.device)
        mask = (prob_mask > 0.3).to(x.dtype)

    elif mask_type == "periphery":
        # Mask outer 16 pixels on all sides
        border = 50
        mask[:, :, :border, :] = 0
        mask[:, :, -border:, :] = 0
        mask[:, :, :, :border] = 0
        mask[:, :, :, -border:] = 0

    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")

    return mask * x


def upsample(x, sf):
    '''s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    '''
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
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
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
        self.downsampling_matrix = create_downsampling_matrix(dim_image, dim_image, sf, device)

    def H(self, x):
        return downsample(x, self.sf)

    def H_adj(self, x):
        return upsample(x, self.sf)


class RandomInpainting(Degradation):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def H(self, x):
        return apply_random_mask(x, self.p)

    def H_adj(self, x):
        return apply_random_mask(x, self.p)


class DiverseMaskInpainting(Degradation):
    def __init__(self, mask_type='central_block'):
        super().__init__()
        self.mask_type = mask_type

    def H(self, x):
        return apply_predefined_mask(x, self.mask_type)

    def H_adj(self, x):
        return apply_predefined_mask(x, self.mask_type)
