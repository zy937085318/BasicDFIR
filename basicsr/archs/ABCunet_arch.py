from turtle import forward

from torch.nn.init import _calculate_fan_in_and_fan_out
import torch.nn as nn
from .unet_arch import *
from basicsr.utils.registry import ARCH_REGISTRY
import torch
import torch.nn.functional as F
import torchvision
try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = DEVICE


@ARCH_REGISTRY.register()
class ABCUNet(nn.Module):
    def __init__(self,
                 input_channels,
                 img_size,
                 ch=32,
                 output_channels=9,
                 ch_mult=(1, 2, 4, 8),
                 num_res_blocks=2,
                 attn_resolutions=(16,),
                 dropout=0.,
                 resamp_with_conv=True,
                 act=Swish(),
                 normalize=group_norm,
                 meanflow=False,
                 check_point=None
                 ):
        super().__init__()
        self.input_channels = input_channels
        self.input_height = img_size
        self.ch = ch
        self.output_channels = 9 #output_channels = input_channels if output_channels is None else output_channels
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.resamp_with_conv = resamp_with_conv
        self.act = act
        self.normalize = normalize

        # init
        self.num_resolutions = num_resolutions = len(ch_mult)
        in_ht = self.input_height
        in_ch = input_channels
        temb_ch = ch * 4
        assert in_ht % 2 ** (num_resolutions -
                             1) == 0, "input_height doesn't satisfy the condition"
        self.A = nn.Parameter(torch.tensor(1.0))
        self.B = nn.Parameter(torch.tensor(1.0))
        self.C = nn.Parameter(torch.tensor(1.0))

        # Timestep embedding
        self.meanflow = meanflow
        if self.meanflow:
            self.temb_net = Dual_TimestepEmbedding(
                embedding_dim=ch,
                hidden_dim=temb_ch,
                output_dim=temb_ch,
                act=act,
            )
        else:
            self.temb_net = TimestepEmbedding(
                embedding_dim=ch,
                hidden_dim=temb_ch,
                output_dim=temb_ch,
                act=act,
            )

        # Downsampling
        self.first_conv = conv2d(in_ch, ch)
        unet_chs = [ch]
        in_ht = in_ht
        in_ch = ch
        down_modules = []
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            block_modules = {}
            out_ch = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block_modules['{}a_{}a_block'.format(i_level, i_block)] = \
                    ResidualBlock(
                        in_ch=in_ch,
                        temb_ch=temb_ch,
                        out_ch=out_ch,
                        dropout=dropout,
                        act=act,
                        normalize=normalize,
                )
                if in_ht in attn_resolutions:
                    block_modules['{}a_{}b_attn'.format(i_level, i_block)] = SelfAttention(
                        out_ch, normalize=normalize)
                unet_chs += [out_ch]
                in_ch = out_ch
            # Downsample
            if i_level != num_resolutions - 1:
                block_modules['{}b_downsample'.format(i_level)] = downsample(
                    out_ch, with_conv=resamp_with_conv)
                in_ht //= 2
                unet_chs += [out_ch]
            # convert list of modules to a module list, and append to a list
            down_modules += [nn.ModuleDict(block_modules)]
        # conver to a module list
        self.down_modules = nn.ModuleList(down_modules)

        # Middle
        mid_modules = []
        mid_modules += [ResidualBlock(in_ch,
                                      temb_ch=temb_ch,
                                      out_ch=in_ch,
                                      dropout=dropout,
                                      act=act,
                                      normalize=normalize)]
        mid_modules += [SelfAttention(in_ch, normalize=normalize)]
        mid_modules += [ResidualBlock(in_ch,
                                      temb_ch=temb_ch,
                                      out_ch=in_ch,
                                      dropout=dropout,
                                      act=act,
                                      normalize=normalize)]
        self.mid_modules = nn.ModuleList(mid_modules)

        # Upsampling
        up_modules = []
        for i_level in reversed(range(num_resolutions)):
            # Residual blocks for this resolution
            block_modules = {}
            out_ch = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                block_modules['{}a_{}a_block'.format(i_level, i_block)] = \
                    ResidualBlock(
                        in_ch=in_ch + unet_chs.pop(),
                        temb_ch=temb_ch,
                        out_ch=out_ch,
                        dropout=dropout,
                        act=act,
                        normalize=normalize)
                if in_ht in attn_resolutions:
                    block_modules['{}a_{}b_attn'.format(i_level, i_block)] = SelfAttention(
                        out_ch, normalize=normalize)
                in_ch = out_ch
            # Upsample
            if i_level != 0:
                block_modules['{}b_upsample'.format(i_level)] = upsample(
                    out_ch, with_conv=resamp_with_conv)
                in_ht *= 2
            # convert list of modules to a module list, and append to a list
            up_modules += [nn.ModuleDict(block_modules)]
        # conver to a module list
        self.up_modules = nn.ModuleList(up_modules)
        assert not unet_chs

        # End
        self.end_conv = nn.Sequential(
            normalize(in_ch),
            self.act,
            conv2d(in_ch, 9, init_scale=0.),
        )
        if check_point is not None:
            checkpoint = torch.load(check_point, map_location=torch.device('cpu'))
            self.load_state_dict(checkpoint, strict=False)

    # noinspection PyMethodMayBeStatic
    def _compute_cond_module(self, module, x, temp):
        for m in module:
            x = m(x, temp)
        return x

    # noinspection PyArgumentList,PyShadowingNames
    def forward_feature(self, x, c=None, temp=None):
        # Init
        B, C, H, W = x.size()
        # Timestep embedding
        if temp is None:
            temp = torch.ones(B).to(x.device)
        if self.meanflow:
            temp, remp = temp[:,0:1,...].squeeze(), temp[:,1:,...].squeeze()
            temb = self.temb_net(temp, remp)
        else:
            temb = self.temb_net(temp)
        assert list(temb.shape) == [B, self.ch * 4]

        # Downsampling
        hs = [self.first_conv(x)]
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            block_modules = self.down_modules[i_level]
            for i_block in range(self.num_res_blocks):
                resnet_block = block_modules['{}a_{}a_block'.format(
                    i_level, i_block)]
                h = resnet_block(hs[-1], temb)
                if h.size(2) in self.attn_resolutions:
                    attn_block = block_modules['{}a_{}b_attn'.format(
                        i_level, i_block)]
                    h = attn_block(h, temb)
                hs.append(h)
            # Downsample
            if i_level != self.num_resolutions - 1:
                downsample = block_modules['{}b_downsample'.format(i_level)]
                hs.append(downsample(hs[-1]))

        # Middle
        h = hs[-1]
        h = self._compute_cond_module(self.mid_modules, h, temb)

        # Upsampling
        for i_idx, i_level in enumerate(reversed(range(self.num_resolutions))):
            # Residual blocks for this resolution
            block_modules = self.up_modules[i_idx]
            for i_block in range(self.num_res_blocks + 1):
                resnet_block = block_modules['{}a_{}a_block'.format(
                    i_level, i_block)]
                h = resnet_block(torch.cat([h, hs.pop()], axis=1), temb)
                if h.size(2) in self.attn_resolutions:
                    attn_block = block_modules['{}a_{}b_attn'.format(
                        i_level, i_block)]
                    h = attn_block(h, temb)
            # Upsample
            if i_level != 0:
                upsample = block_modules['{}b_upsample'.format(i_level)]
                h = upsample(h)
        assert not hs

        # End
        h = self.end_conv(h)
        # assert list(h.size()) == [x.size(
        #     0), self.output_channels, x.size(2), x.size(3)]
        return h

    def forward(self, x, c=None, temp=None):
        """
            img: (B, 3, H, W) 输入RGB图像 (LR)
            x_t: (B, 3, H_target, W_target) 当前隐状态 (HR)
            t: (B,) 时间步
        """
        # ABC流系数
        abc_coeffs = self.forward_feature(x, c, temp)  # (B, 9, H, W)
        # 计算ABC流场
        abc_flow = self.compute_abc_flow_2d_from_image(x, self.A, self.B, self.C)  # (B, 3, H, W)
        
        # ABC流与系数的交互
        u_coeff = abc_coeffs[:, 0:3]  # (B, 3, H, W)
        v_coeff = abc_coeffs[:, 3:6]
        w_coeff = abc_coeffs[:, 6:9]
        
        # 加权ABC流
        u_result = u_coeff[:, 0:1] * abc_flow[:, 0:1] + u_coeff[:, 1:2] * abc_flow[:, 1:2] + u_coeff[:, 2:3] * abc_flow[:, 2:3]
        v_result = v_coeff[:, 0:1] * abc_flow[:, 0:1] + v_coeff[:, 1:2] * abc_flow[:, 1:2] + v_coeff[:, 2:3] * abc_flow[:, 2:3]
        w_result = w_coeff[:, 0:1] * abc_flow[:, 0:1] + w_coeff[:, 1:2] * abc_flow[:, 1:2] + w_coeff[:, 2:3] * abc_flow[:, 2:3]
        
        out = torch.cat([u_result, v_result, w_result], dim=1)  # (B, 3, H, W)
        return out
    
    def compute_abc_flow_2d_from_image(self, sr_xt, A, B, C):
        """
        从图像特征计算2D ABC流
        使用图像梯度来模拟 ABC 流的效应
        
        真正的3D ABC流需要3D网格，这里用2D近似：
        - 用 x 方向的梯度模拟 sin(z)
        - 用 y 方向的梯度模拟 sin(y)
        - 用通道模拟 z 方向
        """
        B_size, C_size, H, W = sr_xt.shape
        
        # 创建坐标网格
        x = torch.linspace(0, 2*torch.pi, W, device=sr_xt.device)
        y = torch.linspace(0, 2*torch.pi, H, device=sr_xt.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # ABC流公式的2D投影
        # 对于每个通道，使用不同的ABC组合
        u = A * torch.sin(Y) + C * torch.cos(Y)  # R分量
        v = B * torch.sin(X) + A * torch.cos(Y)   # G分量
        w = C * torch.sin(Y) + B * torch.cos(X)  # B分量
        
        # 广播到batch
        u = u.unsqueeze(0).expand(B_size, -1, -1)
        v = v.unsqueeze(0).expand(B_size, -1, -1)
        w = w.unsqueeze(0).expand(B_size, -1, -1)
        
        return torch.stack([u, v, w], dim=1)  # (B, 3, H, W)


