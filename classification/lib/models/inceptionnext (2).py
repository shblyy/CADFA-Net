"""
InceptionNeXt implementation, paper: https://arxiv.org/abs/2303.16900
Some code is borrowed from timm: https://github.com/huggingface/pytorch-image-models
"""

from functools import partial

import torch
import torch.nn as nn
import numpy as np

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import checkpoint_seq
from timm.models.layers import to_2tuple

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

from mmseg.models.builder import BACKBONES
# from mmengine.logging import get_logger as get_root_logger
from mmengine.runner import load_checkpoint

from mmengine.model import BaseModule
# from pytorch_wavelets import DWTForward
# from mmseg.apis import init_segmentor
from torchinfo import summary
from typing import Optional, Callable, Union, Tuple, Any
import torch
from torch import nn, Tensor
import numpy as np
import math
from torch import nn

__all__ = ['inceptionnext_tiny', 'inceptionnext_small', 'inceptionnext_base', 'inceptionnext_base_384']
#-------------------------------注意力机制----------------------------------------------
def makeDivisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
def callMethod(self, ElementName):
    return getattr(self, ElementName)
def setMethod(self, ElementName, ElementValue):
    return setattr(self, ElementName, ElementValue)
def shuffleTensor(Feature: Tensor, Mode: int=1) -> Tensor:
    # shuffle multiple tensors with the same indexs
    # all tensors must have the same shape
    if isinstance(Feature, Tensor):
        Feature = [Feature]

    Indexs = None
    Output = []
    for f in Feature:
        # not in-place operation, should update output
        B, C, H, W = f.shape
        if Mode == 1:
            # fully shuffle
            f = f.flatten(2)
            if Indexs is None:
                Indexs = torch.randperm(f.shape[-1], device=f.device)
            f = f[:, :, Indexs.to(f.device)]
            f = f.reshape(B, C, H, W)
        else:
            # shuflle along y and then x axis
            if Indexs is None:
                Indexs = [torch.randperm(H, device=f.device),
                          torch.randperm(W, device=f.device)]
            f = f[:, :, Indexs[0].to(f.device)]
            f = f[:, :, :, Indexs[1].to(f.device)]
        Output.append(f)
    return Output
class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size: int or tuple=1):
        super(AdaptiveAvgPool2d, self).__init__(output_size=output_size)

    def profileModule(self, Input: Tensor):
        Output = self.forward(Input)
        return Output, 0.0, 0.0

class AdaptiveMaxPool2d(nn.AdaptiveMaxPool2d):
    def __init__(self, output_size: int or tuple=1):
        super(AdaptiveMaxPool2d, self).__init__(output_size=output_size)

    def profileModule(self, Input: Tensor):
        Output = self.forward(Input)
        return Output, 0.0, 0.0
class BaseConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: Optional[int] = 1,
            padding: Optional[int] = None,
            groups: Optional[int] = 1,
            bias: Optional[bool] = None,
            BNorm: bool = False,
            # norm_layer: Optional[Callable[..., nn.Module]]=nn.BatchNorm2d,
            ActLayer: Optional[Callable[..., nn.Module]] = None,
            dilation: int = 1,
            Momentum: Optional[float] = 0.1,
            **kwargs: Any
    ) -> None:
        super(BaseConv2d, self).__init__()
        if padding is None:
            padding = int((kernel_size - 1) // 2 * dilation)

        if bias is None:
            bias = not BNorm

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias

        self.Conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, groups, bias, **kwargs)

        self.Bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=Momentum) if BNorm else nn.Identity()

        if ActLayer is not None:
            if isinstance(list(ActLayer().named_modules())[0][1], nn.Sigmoid):
                self.Act = ActLayer()
            else:
                self.Act = ActLayer(inplace=True)
        else:
            self.Act = ActLayer

        self.apply(initWeight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.Conv(x)
        x = self.Bn(x)
        if self.Act is not None:
            x = self.Act(x)
        return x

NormLayerTuple = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.SyncBatchNorm,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.GroupNorm,
    nn.BatchNorm3d,
)
def initWeight(Module):
    # init conv, norm , and linear layers
    ## empty module
    if Module is None:
        return
    ## conv layer
    elif isinstance(Module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(Module.weight, a=math.sqrt(5))
        if Module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(Module.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(Module.bias, -bound, bound)
    ## norm layer
    elif isinstance(Module, NormLayerTuple):
        if Module.weight is not None:
            nn.init.ones_(Module.weight)
        if Module.bias is not None:
            nn.init.zeros_(Module.bias)
    ## linear layer
    elif isinstance(Module, nn.Linear):
        nn.init.kaiming_uniform_(Module.weight, a=math.sqrt(5))
        if Module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(Module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(Module.bias, -bound, bound)
    elif isinstance(Module, (nn.Sequential, nn.ModuleList)):
        for m in Module:
            initWeight(m)
    elif list(Module.children()):
        for m in Module.children():
            initWeight(m)

class MCAttention(nn.Module):
    # Monte carlo attention
    def __init__(
            self,
            InChannels: int,
            HidChannels: int = None,
            SqueezeFactor: int = 4,
            PoolRes: list = [1, 2, 3],
            Act: Callable[..., nn.Module] = nn.ReLU,
            ScaleAct: Callable[..., nn.Module] = nn.Sigmoid,
            MoCOrder: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__()
        if HidChannels is None:
            HidChannels = max(makeDivisible(InChannels // SqueezeFactor, 8), 32)

        AllPoolRes = PoolRes + [1] if 1 not in PoolRes else PoolRes
        for k in AllPoolRes:
            Pooling = AdaptiveAvgPool2d(k)
            setMethod(self, 'Pool%d' % k, Pooling)

        self.SELayer = nn.Sequential(
            BaseConv2d(InChannels, HidChannels, 1, ActLayer=Act),
            BaseConv2d(HidChannels, InChannels, 1, ActLayer=ScaleAct),
        )

        self.PoolRes = PoolRes
        self.MoCOrder = MoCOrder

    def monteCarloSample(self, x: Tensor) -> Tensor:
        if self.training:
            PoolKeep = np.random.choice(self.PoolRes)
            x1 = shuffleTensor(x)[0] if self.MoCOrder else x
            AttnMap: Tensor = callMethod(self, 'Pool%d' % PoolKeep)(x1)
            if AttnMap.shape[-1] > 1:
                AttnMap = AttnMap.flatten(2)
                AttnMap = AttnMap[:, :, torch.randperm(AttnMap.shape[-1])[0]]
                AttnMap = AttnMap[:, :, None, None]  # squeeze twice
        else:
            AttnMap: Tensor = callMethod(self, 'Pool%d' % 1)(x)

        return AttnMap

    def forward(self, x: Tensor) -> Tensor:
        AttnMap = self.monteCarloSample(x)
        ChannelWeights = self.SELayer(AttnMap)  # 生成每个通道的权重
        # return x * self.SELayer(AttnMap)
        return ChannelWeights
#-----------------------------------注意力机制-------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
'''
即插即用模块：CPAM通道和位置注意力模块  SCI 2024

CPAM（通道和位置注意力机制）模块的主要作用是提升小目标（如细胞）分割任务中的精度。

通道注意力：CPAM首先执行通道注意力机制。它对每个通道进行全局平均池化，
然后通过1D卷积来捕捉通道之间的交互信息。这种方法避免了降维问题，
确保模型能够有效地聚焦在最相关的通道特征上。
位置注意力：在完成通道注意力后，CPAM对特征图的水平和垂直轴进行位置注意力处理，
通过池化操作获取空间结构信息。这一步有助于更准确地定位小目标的空间位置。

通过同时优化通道和位置信息，CPAM能够帮助模型更好地识别和分割小而密集的目标，提高分割精度和检测效果.
适用于：小目标分割，小目标检测等所有CV2维任务通用多尺度特征提取模块

'''
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Zoom_cat(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv_l_post_down = Conv(in_dim, 2*in_dim, 3, 1, 1)

    def forward(self, x):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = F.interpolate(s, m.shape[2:], mode='nearest')
        lms = torch.cat([l, m, s], dim=1)
        return lms
class SSFF(nn.Module):
    def __init__(self, inc, channel):
        super(SSFF, self).__init__()
        self.conv0 = Conv(inc[0], channel, 1)
        self.conv1 = Conv(inc[1], channel, 1)
        self.conv2 = Conv(inc[2], channel, 1)
        self.conv3d = nn.Conv3d(channel, channel, kernel_size=(1, 1, 1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3, 1, 1))

    def forward(self, x):
        p3, p4, p5 = x[0], x[1], x[2]
        p3 = self.conv0(p3)
        p4_2 = self.conv1(p4)
        p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
        p5_2 = self.conv2(p5)
        p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')
        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)
        combine = torch.cat([p3_3d, p4_3d, p5_3d], dim=2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x
class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
class local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(local_att, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out
class CPAM(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, ch=256):
        super().__init__()
        self.channel_att = channel_att(ch)
        self.local_att = local_att(ch)

    def forward(self, x):
        input1, input2 = x[0], x[1]
        input1 = self.channel_att(input1)
        x = input1 + input2
        x = self.local_att(x)
        return x
# CPAM通道位置注意力模块：专注于信息通道和空间位置相关的小物体，以提高检测和分割性能
# if __name__ == '__main__':
#     input1 = torch.randn(1, 256, 32, 32)
#     input2 = torch.randn(1, 256, 32, 32)
#     model = CPAM(ch=256)
#     output = model([input1, input2])
#     print('input1_size:', input1.size())
#     print('input2_size:', input2.size())
#     print('output_size:', output.size())


#--------------工作流程------------------#
# 输入的特征图x按照通道被划分为4部分，
# 其他三部分分别通过三种不同类型的深度卷积处理（标准二维卷积、水平和垂直带状卷积）。
# 将四部分重新拼接成一个特征图作为输出。
# class InceptionDWConv2d(nn.Module):
#     """ Inception depthweise convolution
#     """
#
#     def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
#         super().__init__()
#
#         gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
#         self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
#         self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
#                                   groups=gc)
#         self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
#                                   groups=gc)
#         self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
#
#     def forward(self, x):
#         x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
#         return torch.cat(
#             (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
#             dim=1,
#         )

class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )

#使用卷积操作实现通道上的特征变换，保持输入的空间维度（h,w）不变,改变通道数
class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """

    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

#多层感知机的分类头，将输入特征进行全局平均池化，通过全连接层进行特征变换，输出分类结果
class MlpHead(nn.Module):
    """ MLP classification head
    """

    def __init__(self, dim, num_classes=1000, mlp_ratio=3, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), drop=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.mean((2, 3))  # global average pooling
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

#实现了一个基本的神经网络模块
class MetaNeXtBlock(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            token_mixer=InceptionDWConv2d,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=4,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,
            #注意力模块
            attention_module=MCAttention,

    ):
        super().__init__()
        self.token_mixer = token_mixer(dim)#特征通道之间混合信息，默认使用InceptionDWConv2d
        # print("混合：",token_mixer)
        #注意力模块
        self.attention = attention_module(dim)
        self.norm = norm_layer(dim)#对混合后的特征进行归一化，
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)#使用convMlp在通道上特征变换
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None #有助于控制梯度传播，防止梯度爆炸或消失
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x  #保存输入的x，作为跳跃连接，用于和后续主分支输出相加
        # jl = x
        # 注意力模块
        AttnMap = self.attention(x)
        # print("AttnMap维度:",AttnMap.shape)
        x = self.token_mixer(x)  #对特征进行混合
        x = x * AttnMap
        # x = self.attention([jl,x])
        #-----------------------
        x = self.norm(x)   #混合后归一化
        x = self.mlp(x)  #通道维度特征变换
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut  #形成残差连接
        return x

#代表一个阶段stage，每个阶段由多个MetaNeXtBlock 组成。
class MetaNeXtStage(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            ds_stride=2,
            depth=2,
            drop_path_rates=None,
            ls_init_value=1.0,
            act_layer=nn.GELU,
            norm_layer=None,
            mlp_ratio=4,
    ):
        super().__init__()
        self.grad_checkpointing = False
        if ds_stride > 1:
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                nn.Conv2d(in_chs, out_chs, kernel_size=ds_stride, stride=ds_stride),
            )
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(MetaNeXtBlock(
                dim=out_chs,
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratio,
            ))
            in_chs = out_chs
        #按顺序通过每个MetaNeXtBlock块
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        # print("前shape:",x.shape)
        x = self.downsample(x)  #下采样
        # print("后shape:",x.shape)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x
#先下采样，再通过多个MetaNeXtBlock 逐步处理特征，执行 token mixing 和 MLP 操作，返回处理后的特征。

class MetaNeXt(nn.Module):
    r""" MetaNeXt
        A PyTorch impl of : `InceptionNeXt: When Inception Meets ConvNeXt`  - https://arxiv.org/pdf/2203.xxxxx.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: (3, 3, 9, 3)
        dims (tuple(int)): Feature dimension at each stage. Default: (96, 192, 384, 768)
        token_mixers: Token mixer function. Default: nn.Identity
        norm_layer: Normalziation layer. Default: nn.BatchNorm2d
        act_layer: Activation function for MLP. Default: nn.GELU
        mlp_ratios (int or tuple(int)): MLP ratios. Default: (4, 4, 4, 3)
        head_fn: classifier head
        drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            in_chans=3,
            num_classes=1000,
            depths=(3, 3, 9, 3),  #每个stage的block数量
            dims=(96, 192, 384, 768),  #每个stage的特征维度
            # token_mixers=nn.Identity,
            token_mixers=InceptionDWConv2d,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.GELU,
            mlp_ratios=(4, 4, 4, 3),
            head_fn=MlpHead,
            drop_rate=0.,
            drop_path_rate=0.,
            ls_init_value=1e-6,
            **kwargs,
    ):
        super().__init__()

        num_stage = len(depths)
        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage
        if not isinstance(mlp_ratios, (list, tuple)):
            mlp_ratios = [mlp_ratios] * num_stage

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        #将输入通道数3，转为输出通道数96，即输出特征的维度
        self.stem = nn.Sequential(
            # nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            norm_layer(dims[0])
        )

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        #特征提取阶段
        stages = []
        prev_chs = dims[0]
        # feature resolution stages, each consisting of multiple residual blocks
        for i in range(num_stage):
            out_chs = dims[i]  #每个stage的输出特征通道数
            stages.append(MetaNeXtStage(
                prev_chs,
                out_chs,
                ds_stride=2 if i > 0 else 1,
                depth=depths[i],
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratios[i],
            ))
            prev_chs = out_chs
        self.stages = nn.Sequential(*stages)
        self.num_features = prev_chs
        # self.apply(self._init_weights)
        # self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def forward(self, x):
        # print("输入x1:", x.shape)
        input_size = x.size(2)
        # scale = [4, 8, 16, 32]  #下采样倍数
        scale = [1, 2, 4, 8]
        features = [None, None, None, None] #初始化一个列表，用来存储不同尺度的特征图
        # print("stem前",x.shape)
        x = self.stem(x)# 通过stem层（初始卷积层）提取初步特征
        # print("stem后",x.shape)
        features[scale.index(input_size // x.size(2))] = x  #根据scale的值进行缩放
        #遍历模型中的所有 stage。每个 stage 是由多个 MetaNeXtBlock 组成的模块
        for idx, layer in enumerate(self.stages):
            x = layer(x)
            if input_size // x.size(2) in scale:
                features[scale.index(input_size // x.size(2))] = x
        return features
    #初始化权重的方式，如果有预训练模型，_init_weights不会起作用。
    #若不使用预训练模型，_init_weights在模型构建时对权重进行初始化。
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

#----------------------------新增-------------------------------

@BACKBONES.register_module()
class MetaNeXtBackbone(BaseModule):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 depths=(3, 3, 9, 3),
                 # dims=(96, 192, 384, 768),
                 dims=(96/2, 192/2, 384/2, 768/2),
                 token_mixers=InceptionDWConv2d,
                 mlp_ratios=(4, 4, 4, 3),
                 drop_rate=0.,
                 drop_path_rate=0.,
                 ls_init_value=1e-6,
                 # pretrained='https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_tiny.pth',
                 pretrained=None,
                 init_cfg=None):
        super(MetaNeXtBackbone, self).__init__(init_cfg=init_cfg)

        # 初始化自定义的模型，例如 InceptionNeXt
        self.model = MetaNeXt(in_chans=in_chans,
                              num_classes=num_classes,
                              depths=depths,
                              dims=dims,
                              token_mixers=token_mixers,
                              mlp_ratios=mlp_ratios,
                              drop_rate=drop_rate,
                              drop_path_rate=drop_path_rate,
                              ls_init_value=ls_init_value)

        # 预训练权重
        self.pretrained = pretrained
        self.init_weights()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    #初始化预训练模型的权重
    def init_weights(self):
        if isinstance(self.pretrained, str):
            map_location = torch.device('cuda:0')
            # 加载预训练模型的 state_dict
            state_dict = torch.hub.load_state_dict_from_url(self.pretrained, map_location=map_location, check_hash=True)
            # 删除 head 部分的权重键（如果有）
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
            # 加载修改后的 state_dict
            self.model.load_state_dict(state_dict, strict=False)
            print("Pretrained weights loaded successfully.")

        else:
            # 如果没有预训练权重，使用随机初始化
            self.apply(self._init_weights)

    def forward(self, x):
        # print("输入x:", x.shape)
        # Forward 方法，返回四个阶段的特征图
        features = self.model(x)
        # return tuple(feaWWtures)  # 返回 tuple 格式的特征图
        C1 = features[0]  # 第一个 stage 的特征图
        C2 = features[1]  # 第二个 stage 的特征图
        C3 = features[2]  # 第三个 stage 的特征图
        C4 = features[3]  # 第四个 stage 的特征图

        # 返回类似于 ResNet 的特征图
        return C1, C2, C3, C4


#----------------------------新增-------------------------------

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'head.fc',
        **kwargs
    }

#将预训练模型的权重加载到当前模型的参数中
def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}
    for k, v in weight_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f'loading weights... {idx}/{len(model_dict)} items')
    return model_dict


default_cfgs = dict(
    inceptionnext_tiny=_cfg(
        url='https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_tiny.pth',
    ),
    inceptionnext_small=_cfg(
        url='https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_small.pth',
    ),
    inceptionnext_base=_cfg(
        url='https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_base.pth',
    ),
    inceptionnext_base_384=_cfg(
        url='https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_base_384.pth',
        input_size=(3, 384, 384), crop_pct=1.0,
    ),
)


def inceptionnext_tiny(pretrained=False, **kwargs):
    model = MetaNeXt(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),
                     token_mixers=InceptionDWConv2d,
                     **kwargs
                     )
    #为模型设置一个默认的配置字典，这个配置字典里包含了如输入尺寸、预训练权重的 URL 等信息。
    model.default_cfg = default_cfgs['inceptionnext_tiny']
    if pretrained:
        #加载预训练模型
        state_dict = torch.hub.load_state_dict_from_url(url=model.default_cfg['url'], map_location="cpu",
                                                        check_hash=True)
        #使用加载的权重更新模型参数
        model.load_state_dict(state_dict)
    print(model)
    return model


def inceptionnext_small(pretrained=False, **kwargs):
    model = MetaNeXt(depths=(3, 3, 27, 3), dims=(96, 192, 384, 768),
                     token_mixers=InceptionDWConv2d,
                     **kwargs
                     )
    model.default_cfg = default_cfgs['inceptionnext_small']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=model.default_cfg['url'], map_location="cpu",
                                                        check_hash=True)
        model.load_state_dict(state_dict)
    return model


def inceptionnext_base(pretrained=False, **kwargs):
    model = MetaNeXt(depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024),
                     token_mixers=InceptionDWConv2d,
                     **kwargs
                     )
    model.default_cfg = default_cfgs['inceptionnext_base']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=model.default_cfg['url'], map_location="cpu",
                                                        check_hash=True)
        model.load_state_dict(state_dict)
    return model


def inceptionnext_base_384(pretrained=False, **kwargs):
    model = MetaNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024],
                     mlp_ratios=[4, 4, 4, 3],
                     token_mixers=InceptionDWConv2d,
                     **kwargs
                     )
    model.default_cfg = default_cfgs['inceptionnext_base_384']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url=model.default_cfg['url'], map_location="cpu",
                                                        check_hash=True)
        model.load_state_dict(state_dict)
    return model



# if __name__ == '__main__':
#     model = inceptionnext_tiny(pretrained=False)
#     inputs = torch.randn((1, 3, 640, 640))
#     for i in model(inputs):
#         print(i.size())