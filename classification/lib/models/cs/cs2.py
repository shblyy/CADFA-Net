import torch
from torch import nn
from lib.models.cs.scsa import SCSA

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        self.SCSA=SCSA(channels)
    def forward(self, x):
        b, c, h, w = x.size()
        # group_x torch.Size([8, 8, 64, 64])
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        # print("group_x",group_x.shape)
        # x_h torch.Size([8, 8, 64, 1])

        x_h = self.pool_h(group_x)
        # print("x_h", x_h.shape)
        # x_w torch.Size([8, 8, 64, 1])x_w torch.Size([8, 8, 1,64])
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        # print("x_w", x_w.shape)
        # hw torch.Size([8, 8, 128, 1])
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        # print("hw", hw.shape)
        # x_h torch.Size([8, 8, 64, 1])
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        # print("x_h", x_h.shape)
        # x1 torch.Size([8, 8, 64, 64])
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())

        # x1 = self.SCSA(x)
        # print("x1", x1.shape)
        # x2 torch.Size([8, 8, 64, 64])
        x2 = self.conv3x3(group_x)
        # print("x2", x2.shape)
        # x11 torch.Size([8, 1, 8])
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # print("x11", x11.shape)
        # x12 torch.Size([8, 8, 4096])
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # print("x12", x12.shape)
        # x21 torch.Size([8, 1, 8])
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # print("x21", x21.shape)
        # x22 torch.Size([8, 8, 4096])
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # print("x22", x22.shape)
        # weights torch.Size([8, 1, 64, 64])
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        # print("weights", weights.shape)
        # weights1 torch.Size([8, 8, 64, 64])
        # return torch.Size([1, 64, 64, 64])
        # print("weights1", (group_x * weights.sigmoid()).shape)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


# 输入 N C HW,  输出 N C H W
if __name__ == '__main__':
    from torch import nn
    from scsa import SCSA
    from thop import profile
    import time

    # 创建模型实例
    model = EMA(96*2*2*2).cuda()
    input = torch.rand(1, 96*2*2*2, 56//2//2//2, 56//2//2//2).cuda()

    # 计算 FLOPs 和参数量
    flops, params = profile(model, inputs=(input,))
    print(f"FLOPs: {flops / 1e9:.4f} GFLOPs")
    print(f"Params: {params / 1e6:.4f} M")

    # 测量执行时间
    start_time = time.time()
    output = model(input)
    end_time = time.time()

    print(f"Execution Time: {(end_time - start_time) * 1000:.4f} ms")
