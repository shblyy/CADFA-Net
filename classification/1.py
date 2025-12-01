import torch
import torch.nn as nn


class MultiScaleConvModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConvModule, self).__init__()

        # 第一部分 (上半部分)
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_after_concat = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 第二部分 (下半部分)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.residual_bn_relu = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.dilated_conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dilated_conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dilated_conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 残差连接后的 1x1 卷积
        self.residual_final_conv = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 第一部分（多尺度卷积）
        branch1 = self.branch_5x5(x)
        branch2 = self.branch_3x3(x)
        branch3 = self.branch_1x1(x)

        # 将多分支特征拼接
        out1 = torch.cat([branch1, branch2, branch3], dim=1)
        out1 = self.conv_after_concat(out1)

        # 第二部分（扩张卷积 + 残差连接）
        residual = self.residual_conv(x)
        residual = self.residual_bn_relu(residual)

        dilated1 = self.dilated_conv1(residual)
        dilated2 = self.dilated_conv2(residual)
        dilated3 = self.dilated_conv3(residual)

        out2 = torch.cat([dilated1, dilated2, dilated3], dim=1)
        out2 = self.residual_final_conv(out2)

        # 最终将残差连接的结果与多尺度卷积的结果相加
        output = out1 + out2

        return output


# 测试模型
if __name__ == '__main__':
    model = MultiScaleConvModule(in_channels=64, out_channels=64)
    input_tensor = torch.randn(1, 64, 32, 32)  # 假设输入形状为 (batch_size, channels, height, width)
    output = model(input_tensor)
    print(output.shape)
