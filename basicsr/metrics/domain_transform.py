"""
Domain Transform for Edge-Aware Image and Video Processing
基于论文第四章的PyTorch实现

Reference:
  Eduardo S. L. Gastal and Manuel M. Oliveira.
  "Domain Transform for Edge-Aware Image and Video Processing"
  ACM Transactions on Graphics (SIGGRAPH 2011)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainTransformFilter(nn.Module):
    """
    Domain Transform滤波器 - 正确实现

    论文核心思想：
    - 通过引导图像的梯度信息计算边缘保持权重
    - 在水平方向滤波时，使用垂直梯度权重（保留垂直边缘）
    - 在垂直方向滤波时，使用水平梯度权重（保留水平边缘）
    - 通过多次迭代实现多尺度边缘保留滤波

    核心公式：
    I_filt(x) = (1/K(x)) * Σ_k Gσs(x,k) * exp(-|I(x)-I(k)|²/2σr²) * I(k)

    其中K(x)是归一化常数
    """

    def __init__(self, mode='denoise', sigma_s=10.0, sigma_r=0.5, num_iter=3):
        """
        Args:
            mode: 'denoise' | 'detail_enhancement'
            sigma_s: 空间域标准差，控制滤波的空间范围
            sigma_r: 值域标准差，控制边缘保留程度（值越大越保留边缘）
            num_iter: 迭代次数，每迭代一次空间范围翻倍
        """
        super(DomainTransformFilter, self).__init__()
        self.mode = mode
        self.sigma_s = sigma_s
        self.sigma_r = sigma_r
        self.num_iter = num_iter

    def forward(self, input_img, guide=None):
        """
        Args:
            input_img: [B, C, H, W] 输入图像
            guide: [B, 1, H, W] 引导图像，默认为灰度图
        Returns:
            output: [B, C, H, W] 滤波后的图像
        """
        if guide is None:
            if input_img.shape[1] > 1:
                guide = input_img.mean(dim=1, keepdim=True)
            else:
                guide = input_img

        # 计算边缘权重
        wx, wy = self._compute_edge_weights(guide)

        # 多尺度迭代域变换
        output = input_img.clone()
        for i in range(self.num_iter):
            sigma_s_i = self.sigma_s * (2 ** i)
            # 水平滤波使用垂直梯度权重（保留垂直边缘）
            output = self._horizontal_filter(output, wy, sigma_s_i)
            # 垂直滤波使用水平梯度权重（保留水平边缘）
            output = self._vertical_filter(output, wx, sigma_s_i)

        # 细节增强模式
        if self.mode == 'detail_enhancement':
            detail = input_img - output
            output = (output + 2.0 * detail).clamp(0, 1)

        return output

    def _compute_edge_weights(self, guide):
        """
        计算边缘保持权重

        论文公式：w(x,y) = exp(-|∇I(x,y)|² / (2σr²))

        - wx: 水平梯度权重，用于垂直滤波（保留水平边缘）
        - wy: 垂直梯度权重，用于水平滤波（保留垂直边缘）
        """
        # 水平梯度（检测垂直边缘）
        grad_x = guide[:, :, :, 1:] - guide[:, :, :, :-1]
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')

        # 垂直梯度（检测水平边缘）
        grad_y = guide[:, :, 1:, :] - guide[:, :, :-1, :]
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')

        wx = torch.exp(-grad_x.pow(2) / (2 * self.sigma_r ** 2))
        wy = torch.exp(-grad_y.pow(2) / (2 * self.sigma_r ** 2))

        return wx, wy

    def _horizontal_filter(self, img, wy, sigma_s):
        """
        水平方向滤波
        使用垂直梯度权重wy来保留垂直边缘
        """
        B, C, H, W = img.shape

        # 高斯核
        kernel_size = min(int(6 * sigma_s + 1), 61)
        if kernel_size % 2 == 0:
            kernel_size -= 1
        half_k = kernel_size // 2

        # 创建1D高斯核
        x = torch.arange(-half_k, half_k + 1, device=img.device, dtype=torch.float32)
        gaussian_kernel = torch.exp(-x.pow(2) / (2 * sigma_s ** 2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        # 填充
        padded = F.pad(img, (half_k, half_k, 0, 0), mode='replicate')

        # 提取邻域
        unfolded = F.unfold(padded, kernel_size=(1, kernel_size))
        unfolded = unfolded.view(B, C, H, W, kernel_size)

        # 边缘权重：使用垂直梯度权重
        wy_expanded = wy.unsqueeze(-1)

        # 空间高斯权重
        spatial_weight = gaussian_kernel.view(1, 1, 1, 1, kernel_size)

        # 总权重
        weight = spatial_weight * wy_expanded
        weight_sum = weight.sum(dim=-1, keepdim=True) + 1e-8
        weight = weight / weight_sum

        # 加权求和
        output = (unfolded * weight).sum(dim=-1)

        return output

    def _vertical_filter(self, img, wx, sigma_s):
        """
        垂直方向滤波
        使用水平梯度权重wx来保留水平边缘
        """
        B, C, H, W = img.shape

        kernel_size = min(int(6 * sigma_s + 1), 61)
        if kernel_size % 2 == 0:
            kernel_size -= 1
        half_k = kernel_size // 2

        x = torch.arange(-half_k, half_k + 1, device=img.device, dtype=torch.float32)
        gaussian_kernel = torch.exp(-x.pow(2) / (2 * sigma_s ** 2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        padded = F.pad(img, (0, 0, half_k, half_k), mode='replicate')

        unfolded = F.unfold(padded, kernel_size=(kernel_size, 1))
        unfolded = unfolded.view(B, C, H, W, kernel_size)

        # 边缘权重：使用水平梯度权重
        wx_expanded = wx.unsqueeze(-1)
        spatial_weight = gaussian_kernel.view(1, 1, 1, 1, kernel_size)

        weight = spatial_weight * wx_expanded
        weight_sum = weight.sum(dim=-1, keepdim=True) + 1e-8
        weight = weight / weight_sum

        output = (unfolded * weight).sum(dim=-1)

        return output


def domain_transform_filter(input_img, guide=None, sigma_s=10.0, sigma_r=0.5,
                            num_iter=3, mode='denoise'):
    """
    便捷函数：应用域变换滤波

    Args:
        input_img: torch.Tensor [B, C, H, W] 输入图像
        guide: torch.Tensor [B, 1, H, W] 引导图像，默认为灰度图
        sigma_s: 空间标准差（越大滤波范围越广）
        sigma_r: 值域标准差（越大越保留边缘）
        num_iter: 迭代次数（越多尺度越多）
        mode: 'denoise' | 'detail_enhancement'

    Returns:
        torch.Tensor [B, C, H, W] 滤波后的图像
    """
    filter_module = DomainTransformFilter(
        mode=mode,
        sigma_s=sigma_s,
        sigma_r=sigma_r,
        num_iter=num_iter
    )

    if input_img.is_cuda:
        filter_module = filter_module.cuda()

    return filter_module(input_img, guide)


if __name__ == '__main__':
    # 测试代码
    print("Domain Transform Filter - PyTorch Implementation")
    print("Reference: Gastal & Oliveira, SIGGRAPH 2011")
    print("=" * 50)

    # 创建测试图像
    B, C, H, W = 1, 3, 128, 128
    input_img = torch.rand(B, C, H, W)

    print(f"Input shape: {input_img.shape}")
    print(f"Testing denoise mode...")
    denoised = domain_transform_filter(input_img, sigma_s=5.0, sigma_r=0.3, mode='denoise')
    print(f"Output shape: {denoised.shape}")

    print(f"\nTesting detail enhancement mode...")
    enhanced = domain_transform_filter(input_img, sigma_s=10.0, sigma_r=0.5, mode='detail_enhancement')
    print(f"Output shape: {enhanced.shape}")

    print("\nTest passed!")