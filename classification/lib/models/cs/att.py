import math
import copy
import warnings
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class SAB(nn.Module):
    def __init__(self, kernel_size=3):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

        # self.init_weights('normal')

    # def init_weights(self, scheme=''):
    #     named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y=max_out
        # y = torch.cat([avg_out, max_out], dim=1)
        # y = self.conv(y)
        return x*self.sigmoid(y)

class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr=False, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()
        if is_light_sr:  # a larger compression ratio is used for light-SR
            compress_ratio = 6
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            # nn.Conv2d(num_feat // compress_ratio, num_feat // compress_ratio, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            # nn.Conv2d(num_feat, num_feat, kernel_size=1, stride=1, padding=0),
            ChannelAttention(num_feat, squeeze_factor),
            SAB(3)
        )

    def forward(self, x):
        return self.cab(x)


import torch
import time
from torchinfo import summary
from thop import profile

# 定义设备（如果有 GPU 则使用 GPU，否则使用 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义输入张量并将其转移到指定设备
input_tensor = torch.rand(2, 64, 128, 128).to(device)  # 1表示批大小为1，64表示通道数为64，128x128为空间尺寸

# 创建 CAB 模块实例并将其转移到指定设备
CAB_model = CAB(num_feat=64).to(device)

# 打印模型结构和参数量
print("Model Summary:")
summary(CAB_model, input_size=(2, 64, 128, 128))

# 计算 FLOPs 和参数量
flops, params = profile(CAB_model, inputs=(input_tensor,))
print(f"FLOPs: {flops / 1e9:.4f} GFLOPs")  # 转换为 GFLOPs
print(f"Parameters: {params / 1e6:.4f} M")  # 转换为 M

# 测量执行时间
start_time = time.time()
output = CAB_model(input_tensor)
end_time = time.time()

print(f"Execution Time: {end_time - start_time:.4f} seconds")
print("Output Shape:", output.shape)
