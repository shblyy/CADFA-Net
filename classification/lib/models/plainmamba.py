'''
Author: Chenhongyi Yang
'''

import math
from einops import repeat

import torch
import torch.nn as nn

from mmcv.cnn.bricks.transformer import build_dropout
# from mmcv.cnn.utils.weight_init import trunc_normal_

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.layernorm import RMSNorm

import torch

def trunc_normal_(tensor, mean=0., std=1.):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.mul_(std).add_(mean)

class DynamicDirectionAttention(nn.Module):
    def __init__(self, d_state, num_directions):
        super(DynamicDirectionAttention, self).__init__()
        # 定义一个简单的 MLP，用于计算方向的权重
        self.fc = nn.Linear(d_state, num_directions)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x 的形状为 [batch_size, L, d_state]
        # 通过 MLP 生成每个方向的权重，并使用 softmax 归一化
        direction_weights = self.fc(x)
        direction_weights = self.softmax(direction_weights)
        return direction_weights  # 返回权重，形状为 [batch_size, L, num_directions]



class PlainMamba2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_size=7,
        conv_bias=True,
        bias=False,
        init_layer_scale=None,
        default_hw_shape=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.default_hw_shape = default_hw_shape
        self.default_permute_order = None
        self.default_permute_order_inverse = None
        self.n_directions = 4
        if default_hw_shape is not None:
            orders, inverse_orders, directions = self.get_permute_order(default_hw_shape)
            (
                self.default_permute_order,
                self.default_permute_order_inverse,
                self.default_direction
            ) = orders, inverse_orders, directions

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        assert conv_size % 2 == 1
        padding = int(conv_size // 2)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=(conv_size, conv_size),
            stride=(1, 1),
            padding=(padding, padding),
            groups=self.d_inner
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        #方向感知参数，该参数存储每个方向上特定权重，包括上下左右方向
        self.direction_Bs = nn.Parameter(torch.zeros(self.n_directions+1, self.d_state))
        trunc_normal_(self.direction_Bs, std=0.02)

    #按照蛇形扫描的方式，返回每个像素的扫描顺序
    def get_permute_order(self, hw_shape):

        if self.default_permute_order is not None:
            if hw_shape[0] == self.default_hw_shape[0] and hw_shape[1] == self.default_hw_shape[1]:
                return self.default_permute_order, self.default_permute_order_inverse, self.default_direction
        H, W = hw_shape
        L = H * W
        print(hw_shape)
        # [start, right, left, up, down] [0, 1, 2, 3, 4]
#o1 是存储第一种扫描顺序的列表，d1 存储对应的扫描方向，o1_inverse 用于记录逆序的扫描索引。
        o1 = []
        d1 = []
        o1_inverse = [-1 for _ in range(L)]
        #i, j 是扫描时的初始位置（左上角），j_d = "right" 表示当前扫描方向为向右。
        i, j = 0, 0
        j_d = "right"
        #循环执行扫描操作：如果当前扫描方向是向右 (j_d == "right")，并且还没有到达行的右边界，就继续向右扫描。
        # 如果到达边界，就换行向下，并将方向改为向左扫描 (j_d = "left")。
        #如果当前扫描方向是向左，就向左移动，直到到达边界，再向下换行。
        #每个像素的索引（idx）根据其在二维平面上的位置 (i, j) 计算出来，并记录在 o1 中
        #每次扫描方向改变时，将对应的方向（如右、左、下）添加到 d1 中。
        while i < H:
            assert j_d in ["right", "left"]
            idx = i * W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if j_d == "right":
                if j < W-1:
                    j = j + 1
                    d1.append(1)
                else:
                    i = i + 1
                    d1.append(4)
                    j_d = "left"

            else:
                if j > 0:
                    j = j - 1
                    d1.append(2)
                else:
                    i = i + 1
                    d1.append(4)
                    j_d = "right"
        d1 = [0] + d1[:-1]

#开始初始化第二种扫描顺序 o2，即从右下角开始逆序扫描。如果高度是奇数，初始扫描方向是向左；否则，向右开始扫描。
        o2 = []
        d2 = []
        o2_inverse = [-1 for _ in range(L)]

        if H % 2 == 1:
            i, j = H-1, W-1
            j_d = "left"
        else:
            i, j = H-1, 0
            j_d = "right"

        while i > -1:
            assert j_d in ["right", "left"]
            idx = i * W + j
            o2_inverse[idx] = len(o2)
            o2.append(idx)
            if j_d == "right":
                if j < W - 1:
                    j = j + 1
                    d2.append(1)
                else:
                    i = i - 1
                    d2.append(3)
                    j_d = "left"
            else:
                if j > 0:
                    j = j - 1
                    d2.append(2)
                else:
                    i = i - 1
                    d2.append(3)
                    j_d = "right"
        d2 = [0] + d2[:-1]
#o3 存储从上到下扫描的顺序，i_d = "down" 表示垂直方向的扫描。从上到下，逐列扫描；每次扫描结束后，改变方向。
        o3 = []
        d3 = []
        o3_inverse = [-1 for _ in range(L)]
        i, j = 0, 0
        i_d = "down"
        while j < W:
            assert i_d in ["down", "up"]
            idx = i * W + j
            o3_inverse[idx] = len(o3)
            o3.append(idx)
            if i_d == "down":
                if i < H - 1:
                    i = i + 1
                    d3.append(4)
                else:
                    j = j + 1
                    d3.append(1)
                    i_d = "up"
            else:
                if i > 0:
                    i = i - 1
                    d3.append(3)
                else:
                    j = j + 1
                    d3.append(1)
                    i_d = "down"
        d3 = [0] + d3[:-1]
#o4 存储从右到左扫描的顺序，初始扫描方向根据宽度是否为奇数来确定。执行从右向左的扫描，类似之前的垂直扫描，处理到达边界时换行的逻辑。
        o4 = []
        d4 = []
        o4_inverse = [-1 for _ in range(L)]

        if W % 2 == 1:
            i, j = H - 1, W - 1
            i_d = "up"
        else:
            i, j = 0, W - 1
            i_d = "down"
        while j > -1:
            assert i_d in ["down", "up"]
            idx = i * W + j
            o4_inverse[idx] = len(o4)
            o4.append(idx)
            if i_d == "down":
                if i < H - 1:
                    i = i + 1
                    d4.append(4)
                else:
                    j = j - 1
                    d4.append(2)
                    i_d = "up"
            else:
                if i > 0:
                    i = i - 1
                    d4.append(3)
                else:
                    j = j - 1
                    d4.append(2)
                    i_d = "down"
        d4 = [0] + d4[:-1]

        o1 = tuple(o1)
        d1 = tuple(d1)
        o1_inverse = tuple(o1_inverse)

        o2 = tuple(o2)
        d2 = tuple(d2)
        o2_inverse = tuple(o2_inverse)

        o3 = tuple(o3)
        d3 = tuple(d3)
        o3_inverse = tuple(o3_inverse)

        o4 = tuple(o4)
        d4 = tuple(d4)
        o4_inverse = tuple(o4_inverse)
# 返回四种扫描顺序、逆序和方向
        return (o1, o2, o3, o4), (o1_inverse, o2_inverse, o3_inverse, o4_inverse), (d1, d2, d3, d4)

    def forward(self, x, hw_shape):
        print(x.shape)
        batch_size, L, _ = x.shape
        H, W = hw_shape
        E = self.d_inner

        conv_state, ssm_state = None, None

        xz = self.in_proj(x)  # [B, L, 2 * E]
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        x, z = xz.chunk(2, dim=-1)
        x_2d = x.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
        x_2d = self.act(self.conv2d(x_2d))
        x_conv = x_2d.permute(0, 2, 3, 1).reshape(batch_size, L, E)

        x_dbl = self.x_proj(x_conv)  # (B, L, dt_rank + d_state * 2)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)

        dt = dt.permute(0, 2, 1).contiguous()  # [B, d_innter, L]
        B = B.permute(0, 2, 1).contiguous()  # [B, d_state, L]
        C = C.permute(0, 2, 1).contiguous()  # [B, d_state, L]

        assert self.activation in ["silu", "swish"]

        orders, inverse_orders, directions = self.get_permute_order(hw_shape)
        # print("orders",orders[0])
        # print("inverse_orders",inverse_orders[0])
        # print(directions)
        direction_Bs = [self.direction_Bs[d, :] for d in directions]  # each [L, d_state]
        # print(direction_Bs)
        # print(direction_Bs)
        direction_Bs = [dB[None, :, :].expand(batch_size, -1, -1).permute(0, 2, 1).to(dtype=B.dtype) for dB in direction_Bs]
        print(len(direction_Bs))
        print("orders", orders[0].shape)
        print("direction_Bs",direction_Bs[0].shape)
        ys = [
            selective_scan_fn(
                x_conv[:, o, :].permute(0, 2, 1).contiguous(),
                dt,
                A,
                (B + dB).contiguous(),
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            ).permute(0, 2, 1)[:, inv_o, :]
            for o, inv_o, dB in zip(orders, inverse_orders, direction_Bs)
        ]
        # print("ys0:",ys[0])
        # print("ys1:",ys[1])
        # print("ys2:",ys[2])
        # print("ys3:",ys[3])
        y = sum(ys) * self.act(z)
        print(y.shape)
        out = self.out_proj(y)

        if self.init_layer_scale is not None:
            out = out * self.gamma
        return out


class PlainMambaLayer(nn.Module):
    def __init__(
        self,
        embed_dims,
        use_rms_norm,
        with_dwconv,
        drop_path_rate,
        mamba_cfg,
    ):
        super(PlainMambaLayer, self).__init__()
        mamba_cfg.update({'d_model': embed_dims})

        if use_rms_norm:
            self.norm = RMSNorm(embed_dims)
        else:
            self.norm = nn.LayerNorm(embed_dims)

        self.with_dwconv = with_dwconv
        if self.with_dwconv:
            self.dw = nn.Sequential(
                nn.Conv2d(
                    embed_dims,
                    embed_dims,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    bias=False,
                    groups=embed_dims
                ),
                nn.BatchNorm2d(embed_dims),
                nn.GELU(),
            )
        self.mamba = PlainMamba2D(**mamba_cfg)
        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))

    def forward(self, x, hw_shape):
        mixed_x = self.drop_path(self.mamba(self.norm(x), hw_shape))
        mixed_x = mixed_x + x
        if self.with_dwconv:
            b, l, c = mixed_x.shape
            h, w = hw_shape
            mixed_x = mixed_x.reshape(b, h, w, c).permute(0, 3, 1, 2)
            mixed_x = self.dw(mixed_x)
            mixed_x = mixed_x.reshape(b, c, h * w).permute(0, 2, 1)
        return mixed_x


# Assuming PlainMamba2D and PlainMambaLayer are defined in your environment
# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化一个 PlainMamba2D 模块
d_model = 64  # 特征维度（通道数）
plain_mamba = PlainMamba2D(
    d_model=d_model,
    d_state=16,
    expand=2,
    conv_size=7,
    default_hw_shape=(32, 32)  # 假设输入图像尺寸为 32x32
)

# 将模型移动到 GPU（如果可用）
plain_mamba = plain_mamba.to(device)

# 示例输入
batch_size = 8  # 批次大小
H, W = 32, 32  # 图像高度和宽度
L = H * W  # 展平后的图像长度

# 创建一个模拟输入张量，假设这是展平后的批量图像，特征维度为 d_model
x = torch.randn(batch_size, L, d_model).to(device)  # 将输入移动到 GPU

# 前向传播使用 PlainMamba2D 实例
output = plain_mamba(x, (H, W))

# 输出结果的形状
print(output.shape)  # 前向传播后的输出形状