# -*- coding: utf-8 -*-
# @Time    : 2025/11/29 18:17
# @Author  : Yan Zhang
# @FileName: utils.py
# @Email   : yanzhang1991@cqupt.edu.cn
import torch


def add_random_noise(images, noise_range=0.1, noise_prob=0.2):
    noise = (torch.rand_like(images) * 2 - 1) * noise_range
    mask = torch.rand(images.size(0), 1, 1, 1, device=images.device) < noise_prob
    noisy_images = images + noise * mask

    noisy_images = torch.clamp(noisy_images, -1.0, 1.0)

    return noisy_images

def expand_tensor_like(input_tensor, expand_to):
    """`input_tensor` is a 1d vector of length equal to the batch size of `expand_to`,
    expand `input_tensor` to have the same shape as `expand_to` along all remaining dimensions.

    Args:
        input_tensor (Tensor): (batch_size,).
        expand_to (Tensor): (batch_size, ...).

    Returns:
        Tensor: (batch_size, ...).
    """
    assert input_tensor.ndim == 1, "Input tensor must be a 1d vector."
    assert (
        input_tensor.shape[0] == expand_to.shape[0]
    ), f"The first (batch_size) dimension must match. Got shape {input_tensor.shape} and {expand_to.shape}."

    dim_diff = expand_to.ndim - input_tensor.ndim

    t_expanded = input_tensor.clone()
    t_expanded = t_expanded.reshape(-1, *([1] * dim_diff))

    return t_expanded.expand_as(expand_to)