import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, List, Dict, Optional, Union
from enum import Enum


class PaddingMode(Enum):
    """图像不能整除时的填充模式"""
    ZEROS = "zeros"  # 用0填充（黑色）
    ONES = "ones"  # 用1填充（白色）
    REFLECT = "reflect"  # 镜像填充
    REPLICATE = "replicate"  # 复制边缘像素填充
    CIRCULAR = "circular"  # 循环填充


class MergeMode(Enum):
    """重叠区域的合并模式"""
    AVERAGE = "average"  # 对重叠像素取平均值
    MAX = "max"  # 取最大值
    MIN = "min"  # 取最小值
    FIRST = "first"  # 使用第一个图块的值（从左到右，从上到下）
    LAST = "last"  # 使用最后一个图块的值
    WEIGHTED = "weighted"  # 加权平均（基于距离）


class ImageTiler:
    """
    图像分块工具，用于处理固定大小的模型输入。
    
    参数:
        tile_size (int): 正方形图块的大小（如128表示128x128）
        overlap_h (int): 水平方向重叠像素数（默认: 0）
        overlap_v (int): 垂直方向重叠像素数（默认: 0）
        padding_mode (PaddingMode): 图像不能整除时的填充方式
        merge_mode (MergeMode): 重叠区域的合并方式
        device (str): 使用的设备（'cuda' 或 'cpu'）
    """
    
    def __init__(
        self,
        tile_size: int = 128,
        overlap_h: int = 0,
        overlap_v: int = 0,
        padding_mode: Union[str, PaddingMode] = PaddingMode.ZEROS,
        merge_mode: Union[str, MergeMode] = MergeMode.AVERAGE,
        device: str = "cuda"
    ):
        self.tile_size = tile_size
        self.overlap_h = overlap_h
        self.overlap_v = overlap_v
        
        # 如果传入的是字符串，转换为枚举类型
        if isinstance(padding_mode, str):
            self.padding_mode = PaddingMode(padding_mode)
        else:
            self.padding_mode = padding_mode
            
        if isinstance(merge_mode, str):
            self.merge_mode = MergeMode(merge_mode)
        else:
            self.merge_mode = merge_mode
            
        self.device = device
        
        # 检查参数是否有效
        if overlap_h >= tile_size:
            raise ValueError(f"水平重叠({overlap_h})必须小于图块大小({tile_size})")
        if overlap_v >= tile_size:
            raise ValueError(f"垂直重叠({overlap_v})必须小于图块大小({tile_size})")
        if overlap_h < 0 or overlap_v < 0:
            raise ValueError(f"重叠值不能为负数")
        if tile_size <= 0:
            raise ValueError(f"图块大小必须为正数")
        
    def split_image(
        self,
        image: torch.Tensor,
        return_metadata: bool = True
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], Dict]]:
        """
        将图像分割成重叠的图块。
        
        参数:
            image: 输入的图像张量，形状为 (C, H, W) 或 (B, C, H, W)
            return_metadata: 是否返回图块信息
            
        返回:
            如果 return_metadata=False: 返回图块列表
            如果 return_metadata=True: 返回(图块列表, 信息字典)
            
        信息字典包含:
            - original_size: 原始图像的尺寸 (H, W)
            - padded_size: 填充后的尺寸 (H, W)
            - num_tiles_h: 水平方向图块数量
            - num_tiles_v: 垂直方向图块数量
            - tile_positions: 每个图块的位置 (top, left)
            - padding_info: 填充信息 (上, 下, 左, 右)
        """
        # 检查输入维度
        if image.dim() not in [3, 4]:
            raise ValueError(f"输入图像必须是3维(C, H, W)或4维(B, C, H, W)，当前是{image.dim()}维")
        
        # 处理批次维度
        has_batch = image.dim() == 4  # 是否有批次维度
        if not has_batch:
            image = image.unsqueeze(0)  # 添加批次维度
            
        B, C, H, W = image.shape  # 批次、通道数、高度、宽度
        
        # 处理边缘情况：图像小于图块大小
        if H < self.tile_size or W < self.tile_size:
            # 对于特别小的图像，我们会填充到至少图块大小
            pass  # 下面的填充操作会处理
        
        # 计算步长（不重叠的部分）
        step_h = self.tile_size - self.overlap_h  # 水平步长
        step_v = self.tile_size - self.overlap_v  # 垂直步长
        
        # 计算需要多少个图块
        num_tiles_h = int(np.ceil((W - self.overlap_h) / step_h))  # 水平方向图块数
        num_tiles_v = int(np.ceil((H - self.overlap_v) / step_v))  # 垂直方向图块数
        
        # 计算填充后需要的尺寸
        required_w = (num_tiles_h - 1) * step_h + self.tile_size  # 需要的宽度
        required_h = (num_tiles_v - 1) * step_v + self.tile_size  # 需要的高度
        
        # 计算需要填充多少
        pad_w = max(0, required_w - W)  # 水平方向需要填充的总数
        pad_h = max(0, required_h - H)  # 垂直方向需要填充的总数
        
        # 平均分配到两边（左右对称填充）
        pad_left = pad_w // 2  # 左边填充
        pad_right = pad_w - pad_left  # 右边填充
        pad_top = pad_h // 2  # 上边填充
        pad_bottom = pad_h - pad_top  # 下边填充
        
        # 应用填充
        padding_mode_str = self.padding_mode.value
        if padding_mode_str == "zeros":
            pad_value = 0.0  # 用0填充
            padded_image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=pad_value)
        elif padding_mode_str == "ones":
            pad_value = 1.0  # 用1填充
            padded_image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=pad_value)
        else:
            # 对于其他填充模式（镜像、复制、循环）
            padded_image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode=padding_mode_str)
        
        _, _, padded_h, padded_w = padded_image.shape  # 填充后的尺寸
        
        # 提取图块
        tiles = []  # 存储所有图块
        tile_positions = []  # 存储每个图块的位置
        
        for i in range(num_tiles_v):  # 垂直方向遍历
            for j in range(num_tiles_h):  # 水平方向遍历
                top = i * step_v  # 图块的上边界
                left = j * step_h  # 图块的左边界
                
                # 提取图块
                tile = padded_image[:, :, top:top+self.tile_size, left:left+self.tile_size]
                
                # 如果输入没有批次维度，去掉它
                if not has_batch:
                    tile = tile.squeeze(0)
                    
                tiles.append(tile)
                tile_positions.append((top, left))
        
        if return_metadata:
            metadata = {
                'original_size': (H, W),  # 原始尺寸
                'padded_size': (padded_h, padded_w),  # 填充后尺寸
                'num_tiles_h': num_tiles_h,  # 水平图块数
                'num_tiles_v': num_tiles_v,  # 垂直图块数
                'total_tiles': len(tiles),  # 总图块数
                'tile_positions': tile_positions,  # 图块位置
                'padding_info': (pad_top, pad_bottom, pad_left, pad_right),  # 填充信息
                'tile_size': self.tile_size,  # 图块大小
                'overlap_h': self.overlap_h,  # 水平重叠
                'overlap_v': self.overlap_v,  # 垂直重叠
                'step_h': step_h,  # 水平步长
                'step_v': step_v,  # 垂直步长
                'padding_mode': self.padding_mode.value,  # 填充模式
                'merge_mode': self.merge_mode.value  # 合并模式
            }
            return tiles, metadata
        else:
            return tiles
    
    def merge_tiles(
        self,
        tiles: List[torch.Tensor],
        metadata: Dict,
        return_original_size: bool = True
    ) -> torch.Tensor:
        """
        将图块合并回单张图像。
        
        参数:
            tiles: 图块张量列表，形状为 (C, H, W) 或 (B, C, H, W)
            metadata: split_image 返回的信息字典
            return_original_size: 是否裁剪回原始大小（去掉填充）
            
        返回:
            合并后的图像张量，形状为 (C, H, W) 或 (B, C, H, W)
        """
        # 检查图块是否有批次维度
        has_batch = tiles[0].dim() == 4
        
        # 获取尺寸信息
        if has_batch:
            B, C, H_tile, W_tile = tiles[0].shape
        else:
            C, H_tile, W_tile = tiles[0].shape
            
        padded_h, padded_w = metadata['padded_size']  # 填充后的尺寸
        num_tiles_h = metadata['num_tiles_h']  # 水平图块数
        num_tiles_v = metadata['num_tiles_v']  # 垂直图块数
        step_h = metadata['step_h']  # 水平步长
        step_v = metadata['step_v']  # 垂直步长
        
        # 初始化输出图像
        if has_batch:
            merged = torch.zeros((B, C, padded_h, padded_w), device=tiles[0].device, dtype=tiles[0].dtype)  # 合并后的图像
            weight_map = torch.zeros((B, 1, padded_h, padded_w), device=tiles[0].device, dtype=tiles[0].dtype)  # 权重图
        else:
            merged = torch.zeros((C, padded_h, padded_w), device=tiles[0].device, dtype=tiles[0].dtype)
            weight_map = torch.zeros((1, padded_h, padded_w), device=tiles[0].device, dtype=tiles[0].dtype)
        
        # 为重叠区域创建权重图
        if self.merge_mode == MergeMode.WEIGHTED:
            # 创建基于距离的权重
            weight_tile = self._create_weight_tile(self.tile_size, self.overlap_h, self.overlap_v)
            if has_batch:
                weight_tile = weight_tile.unsqueeze(0).unsqueeze(0)  # 形状变为 (1, 1, H, W)
            else:
                weight_tile = weight_tile.unsqueeze(0)  # 形状变为 (1, H, W)
        
        # 合并所有图块
        tile_idx = 0  # 图块索引
        for i in range(num_tiles_v):  # 垂直方向遍历
            for j in range(num_tiles_h):  # 水平方向遍历
                top = i * step_v  # 图块在原图中的上边界
                left = j * step_h  # 图块在原图中的左边界
                
                tile = tiles[tile_idx]  # 当前图块
                
                if self.merge_mode == MergeMode.AVERAGE:
                    # 平均值模式：累加值和权重
                    if has_batch:
                        merged[:, :, top:top+self.tile_size, left:left+self.tile_size] += tile
                        weight_map[:, :, top:top+self.tile_size, left:left+self.tile_size] += 1
                    else:
                        merged[:, top:top+self.tile_size, left:left+self.tile_size] += tile
                        weight_map[:, top:top+self.tile_size, left:left+self.tile_size] += 1
                        
                elif self.merge_mode == MergeMode.WEIGHTED:
                    # 加权平均模式：按权重累加
                    if has_batch:
                        merged[:, :, top:top+self.tile_size, left:left+self.tile_size] += tile * weight_tile
                        weight_map[:, :, top:top+self.tile_size, left:left+self.tile_size] += weight_tile
                    else:
                        merged[:, top:top+self.tile_size, left:left+self.tile_size] += tile * weight_tile
                        weight_map[:, top:top+self.tile_size, left:left+self.tile_size] += weight_tile
                        
                elif self.merge_mode == MergeMode.MAX:
                    # 最大值模式：取所有图块中的最大值
                    if has_batch:
                        merged[:, :, top:top+self.tile_size, left:left+self.tile_size] = torch.maximum(
                            merged[:, :, top:top+self.tile_size, left:left+self.tile_size],
                            tile
                        )
                    else:
                        merged[:, top:top+self.tile_size, left:left+self.tile_size] = torch.maximum(
                            merged[:, top:top+self.tile_size, left:left+self.tile_size],
                            tile
                        )
                        
                elif self.merge_mode == MergeMode.MIN:
                    # 最小值模式：取所有图块中的最小值
                    if tile_idx == 0:  # 第一个图块，直接赋值
                        if has_batch:
                            merged[:, :, top:top+self.tile_size, left:left+self.tile_size] = tile
                        else:
                            merged[:, top:top+self.tile_size, left:left+self.tile_size] = tile
                    else:  # 后续图块，取较小值
                        if has_batch:
                            merged[:, :, top:top+self.tile_size, left:left+self.tile_size] = torch.minimum(
                                merged[:, :, top:top+self.tile_size, left:left+self.tile_size],
                                tile
                            )
                        else:
                            merged[:, top:top+self.tile_size, left:left+self.tile_size] = torch.minimum(
                                merged[:, top:top+self.tile_size, left:left+self.tile_size],
                                tile
                            )
                            
                elif self.merge_mode == MergeMode.FIRST:
                    # 使用第一个图块：只写入还没有被写入的区域
                    if tile_idx == 0 or (has_batch and weight_map[:, :, top:top+self.tile_size, left:left+self.tile_size].sum() == 0) or \
                       (not has_batch and weight_map[:, top:top+self.tile_size, left:left+self.tile_size].sum() == 0):
                        if has_batch:
                            merged[:, :, top:top+self.tile_size, left:left+self.tile_size] = tile
                            weight_map[:, :, top:top+self.tile_size, left:left+self.tile_size] = 1
                        else:
                            merged[:, top:top+self.tile_size, left:left+self.tile_size] = tile
                            weight_map[:, top:top+self.tile_size, left:left+self.tile_size] = 1
                            
                elif self.merge_mode == MergeMode.LAST:
                    # 使用最后一个图块：总是覆盖写入
                    if has_batch:
                        merged[:, :, top:top+self.tile_size, left:left+self.tile_size] = tile
                    else:
                        merged[:, top:top+self.tile_size, left:left+self.tile_size] = tile
                
                tile_idx += 1  # 移动到下一个图块
        
        # 对平均值和加权平均模式进行归一化
        if self.merge_mode in [MergeMode.AVERAGE, MergeMode.WEIGHTED]:
            weight_map = torch.clamp(weight_map, min=1e-8)  # 避免除以0
            if has_batch:
                merged = merged / weight_map  # 除以权重得到平均值
            else:
                merged = merged / weight_map.squeeze(0)
        
        # 如果需要，裁剪回原始大小（去掉填充的部分）
        if return_original_size:
            pad_top, pad_bottom, pad_left, pad_right = metadata['padding_info']
            original_h, original_w = metadata['original_size']
            
            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                if has_batch:
                    merged = merged[:, :, pad_top:padded_h-pad_bottom, pad_left:padded_w-pad_right]
                else:
                    merged = merged[:, pad_top:padded_h-pad_bottom, pad_left:padded_w-pad_right]
        
        return merged
    
    def _create_weight_tile(self, tile_size: int, overlap_h: int, overlap_v: int) -> torch.Tensor:
        """
        创建用于加权合并的权重图块。
        权重在中心区域较高，在边缘（特别是重叠区域）较低。
        """
        weights = torch.ones((tile_size, tile_size), dtype=torch.float32)  # 初始化全1权重
        
        # 计算中心点
        center_h, center_v = tile_size // 2, tile_size // 2
        
        # 创建网格坐标
        y, x = torch.meshgrid(
            torch.arange(tile_size, dtype=torch.float32),
            torch.arange(tile_size, dtype=torch.float32),
            indexing='ij'
        )
        
        # 计算到中心的距离（归一化到0-1）
        dist_h = torch.abs(x - center_h) / (tile_size / 2)  # 水平方向距离
        dist_v = torch.abs(y - center_v) / (tile_size / 2)  # 垂直方向距离
        
        # 权重随距离中心变远而减小
        # 在重叠区域，权重应该更低
        if overlap_h > 0:
            # 水平边缘权重较低
            edge_weight_h = torch.clamp(1.0 - dist_h * 2, min=0.1)
            weights = weights * edge_weight_h
        
        if overlap_v > 0:
            # 垂直边缘权重较低
            edge_weight_v = torch.clamp(1.0 - dist_v * 2, min=0.1)
            weights = weights * edge_weight_v
        
        return weights
    
    def process_image(
        self,
        image: torch.Tensor,
        process_fn: callable,
        return_original_size: bool = True
    ) -> torch.Tensor:
        """
        处理图像的完整流程：分割成图块 -> 处理每个图块 -> 合并回原图。
        
        参数:
            image: 输入图像张量 (C, H, W) 或 (B, C, H, W)
            process_fn: 处理每个图块的函数，输入一个图块，返回处理后的图块
            return_original_size: 合并后是否裁剪回原始大小
            
        返回:
            处理后的图像张量，形状与输入相同
        """
        # 1. 分割图像成图块
        tiles, metadata = self.split_image(image, return_metadata=True)
        
        # 2. 处理每个图块
        processed_tiles = []
        for tile in tiles:
            processed_tile = process_fn(tile)  # 调用处理函数
            processed_tiles.append(processed_tile)
        
        # 3. 合并处理后的图块
        merged = self.merge_tiles(processed_tiles, metadata, return_original_size=return_original_size)
        
        return merged


def create_tiler_from_config(config: Dict) -> ImageTiler:
    """
    从配置字典创建 ImageTiler 对象。
    
    参数:
        config: 配置字典，包含以下键：
            - tile_size (int): 图块大小
            - overlap_h (int, 可选): 水平重叠
            - overlap_v (int, 可选): 垂直重叠
            - padding_mode (str, 可选): 填充模式
            - merge_mode (str, 可选): 合并模式
            - device (str, 可选): 使用的设备
            
    返回:
        ImageTiler 对象
    """
    return ImageTiler(
        tile_size=config.get('tile_size', 128),
        overlap_h=config.get('overlap_h', 0),
        overlap_v=config.get('overlap_v', 0),
        padding_mode=config.get('padding_mode', 'zeros'),
        merge_mode=config.get('merge_mode', 'average'),
        device=config.get('device', 'cuda')
    )


# -------------------------------------------------------------------------
# 兼容 image/image_split.py 的函数式接口
# 以 image_split.py 为准对齐函数名与返回的 pad_info/patches 结构，方便迁移到其他项目
# -------------------------------------------------------------------------

def split_with_overlap(
    img: torch.Tensor,
    h: int,
    w: int,
    stride_h: Optional[int] = None,
    stride_w: Optional[int] = None,
    padding_mode: str = 'reflect',
    return_padding: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
    """
    带重叠的切分（滑动窗口），自动 padding 到能被 h,w 整除的最小尺寸

    参数:
        img: 4维张量 (B, C, H, W)
        h, w: 每个 patch 的高度和宽度
        stride_h, stride_w: 滑动步长，None 表示无重叠（等于 h,w）
        padding_mode: 'reflect', 'replicate', 'constant', 'circular'
        return_padding: 是否返回 padding 信息（用于后续合并）

    返回:
        patches: 6维张量 (B, n_h, n_w, C, h, w)
        pad_info: (如果 return_padding=True) 包含 padding 信息的字典
    """
    if img.dim() != 4:
        raise ValueError(f"img 必须是 4 维张量 (B, C, H, W)，当前为 {img.dim()} 维")

    B, C, H, W = img.shape
    stride_h = stride_h or h
    stride_w = stride_w or w

    # 计算垂直方向需要的 patches 数量和总高度
    if H <= h:
        n_h = 1
        needed_h = h
    else:
        n_h = math.ceil((H - h) / stride_h) + 1
        needed_h = (n_h - 1) * stride_h + h

    # 计算水平方向
    if W <= w:
        n_w = 1
        needed_w = w
    else:
        n_w = math.ceil((W - w) / stride_w) + 1
        needed_w = (n_w - 1) * stride_w + w

    # 计算 padding
    pad_h = needed_h - H
    pad_w = needed_w - W

    # 对称 padding（优先底部和右侧）
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # 应用 padding
    if pad_h > 0 or pad_w > 0:
        img_padded = F.pad(
            img,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode=padding_mode
        )
    else:
        img_padded = img

    # 使用 unfold 提取 patches: 结果形状 (B, C, n_h, n_w, h, w)
    patches = img_padded.unfold(2, h, stride_h).unfold(3, w, stride_w)
    # 调整为 (B, n_h, n_w, C, h, w)
    patches = patches.permute(0, 2, 3, 1, 4, 5)

    if return_padding:
        pad_info = {
            'original_size': (H, W),
            'padded_size': (needed_h, needed_w),
            'padding': (pad_top, pad_bottom, pad_left, pad_right),
            'n_patches': (n_h, n_w),
            'stride': (stride_h, stride_w),
            'patch_size': (h, w)
        }
        return patches, pad_info

    return patches


def create_gaussian_window(
    h: int,
    w: int,
    sigma_factor: float = 4.0,
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    """
    创建二维高斯权重窗口

    参数:
        h, w: 窗口高度和宽度
        sigma_factor: 标准差系数，sigma = size / sigma_factor，越小边缘越锐利
    """
    # 创建坐标网格，中心为原点
    y = torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2
    x = torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2
    y, x = torch.meshgrid(y, x, indexing='ij')

    # 计算高斯分布
    sigma_y = h / sigma_factor
    sigma_x = w / sigma_factor
    gaussian = torch.exp(-(y ** 2 / (2 * sigma_y ** 2) + x ** 2 / (2 * sigma_x ** 2)))

    # 归一化到 [0, 1]
    return gaussian / gaussian.max()


def merge_with_padding(
    patches: torch.Tensor,
    pad_info: Dict,
    mode: str = 'mean',
    gaussian_sigma: float = 4.0
) -> torch.Tensor:
    """
    将带 padding 切分的 patches 合并回原图尺寸

    参数:
        patches: (B, n_h, n_w, C, h, w) 或 (B, num_patches, C, h, w)
        pad_info: padding 信息字典（与 split_with_overlap 返回的一致）
        mode: 'mean'(平均), 'sum'(求和), 'gaussian'(高斯加权)
        gaussian_sigma: 高斯分布的 sigma 系数，默认 4.0
    """
    if patches.dim() not in (5, 6):
        raise ValueError(
            f"patches 需要是 5 维或 6 维张量，当前为 {patches.dim()} 维"
        )

    B = patches.shape[0]
    C = patches.shape[-3]
    H, W = pad_info['original_size']
    needed_h, needed_w = pad_info['padded_size']
    pad_top, pad_bottom, pad_left, pad_right = pad_info['padding']
    stride_h, stride_w = pad_info['stride']
    h, w = pad_info['patch_size']
    n_h, n_w = pad_info['n_patches']
    device = patches.device

    # 统一为 5 维: (B, num_patches, C, h, w)
    if patches.dim() == 6:
        patches = patches.reshape(B, n_h * n_w, C, h, w)

    num_patches = patches.shape[1]

    # 准备权重
    if mode == 'gaussian':
        # 创建高斯权重窗口 (h, w)
        weight_window = create_gaussian_window(h, w, gaussian_sigma, device)
        # 扩展到 (1, 1, 1, h, w) 用于广播
        weight_window = weight_window.view(1, 1, 1, h, w)
        # 应用到所有 patches: (B, num_patches, C, h, w) * (1, 1, 1, h, w)
        weighted_patches = patches * weight_window
    else:
        weighted_patches = patches

    # reshape 为 F.fold 格式: (B, C*h*w, num_patches)
    patches_flat = weighted_patches.permute(0, 2, 3, 4, 1).reshape(
        B, C * h * w, num_patches
    )

    # 使用 F.fold 合并
    merged_padded = F.fold(
        patches_flat,
        output_size=(needed_h, needed_w),
        kernel_size=(h, w),
        stride=(stride_h, stride_w)
    )

    # 计算权重和（用于归一化）
    if mode == 'mean':
        ones = torch.ones_like(patches_flat)
        norm = F.fold(
            ones,
            output_size=(needed_h, needed_w),
            kernel_size=(h, w),
            stride=(stride_h, stride_w)
        )
        merged_padded = merged_padded / norm.clamp(min=1)

    elif mode == 'gaussian':
        # 高斯模式：需要计算权重图的 fold 和
        weight_expanded = weight_window.view(
            1, 1, h, w, 1
        ).expand(B, C, -1, -1, num_patches)
        weight_flat = weight_expanded.reshape(B, C * h * w, num_patches)

        # fold 权重和
        weight_sum = F.fold(
            weight_flat,
            output_size=(needed_h, needed_w),
            kernel_size=(h, w),
            stride=(stride_h, stride_w)
        )

        # 归一化：加权求和 / 权重和
        merged_padded = merged_padded / weight_sum.clamp(min=1e-8)

    # 裁剪掉 padding
    merged = merged_padded[:, :, pad_top:pad_top + H, pad_left:pad_left + W]

    return merged