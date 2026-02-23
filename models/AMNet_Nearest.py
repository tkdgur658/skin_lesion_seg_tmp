# https://github.com/8yike/AM-Net/blob/main/AMNet.py
# Modified: all bilinear interpolation replaced with nearest for deterministic reproducibility
import torch
import torch.nn as nn
import torch.nn.functional as F

# 加入的残差块代码内容(MFIM中的)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 使用空间通道特征融合模块 SCFFM
class SCModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SCModule, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_att = self.channel_attention(x)
        x_channel = x * channel_att
        spatial_att = self.spatial_attention(x)
        x_spatial = x * spatial_att
        x = x_channel + x_spatial
        return x

class SCFFMModule(nn.Module):
    def __init__(self, in_channels_low, in_channels_high, out_channels, reduction=16):
        super(SCFFMModule, self).__init__()
        self.conv_low = nn.Conv2d(in_channels_low, out_channels, kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_high = nn.Conv2d(in_channels_high, out_channels, kernel_size=1)
        self.sc = SCModule(out_channels, reduction=reduction)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x_low, x_high):
        x_low = self.conv_low(x_low)
        x_low = self.global_avg_pool(x_low)

        x_high = self.conv_high(x_high)
        x_high = self.sc(x_high)
        channel_weight = self.adaptive_avg_pool(x_high)
        x_low = x_low * channel_weight
        return x_low + x_high


# 细节边界增强注意力模块（DBEAM）
class DBEAM(nn.Module):
    def __init__(self, channels=64, r=4):
        super(DBEAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        x1 = x * wei
        xs = self.spatial_attention(x)
        x2 = x * xs
        y = x1 + x2
        return y



class DepthwiseSeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthwiseSeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )



class AMNet_Nearest(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 2, base_c: int = 32):
        super(AMNet_Nearest, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes


        self.conv_input = DepthwiseSeparableConv(input_channels, base_c, kernel_size=3, padding=1)

        self.residual1 = BasicBlock(base_c, base_c)
        self.left_conv1 = DepthwiseSeparableConv(base_c, base_c * 2, kernel_size=3, padding=1)
        self.right_conv1 = nn.Sequential(
            DepthwiseSeparableConv(base_c, base_c * 2, kernel_size=7, padding=3),
            DepthwiseSeparableConv(base_c * 2, base_c * 2, kernel_size=5, padding=2),
        )
        self.conv1 = nn.Conv2d(base_c * 5, base_c * 2, kernel_size=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.residual2 = BasicBlock(base_c * 2, base_c * 2)
        self.left_conv2 = DepthwiseSeparableConv(base_c * 2, base_c * 4, kernel_size=3, padding=1)
        self.right_conv2 = nn.Sequential(
            DepthwiseSeparableConv(base_c * 2, base_c * 4, kernel_size=7, padding=3),
            DepthwiseSeparableConv(base_c * 4, base_c * 4, kernel_size=5, padding=2),
        )
        self.conv2 = nn.Conv2d(base_c * 10, base_c * 4, kernel_size=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.residual3 = BasicBlock(base_c * 4, base_c * 4)
        self.left_conv3 = DepthwiseSeparableConv(base_c * 4, base_c * 8, kernel_size=3, padding=1)
        self.right_conv3 = nn.Sequential(
            DepthwiseSeparableConv(base_c * 4, base_c * 8, kernel_size=7, padding=3),
            DepthwiseSeparableConv(base_c * 8, base_c * 8, kernel_size=5, padding=2),
        )
        self.conv3 = nn.Conv2d(base_c * 20, base_c * 8, kernel_size=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.residual4 = BasicBlock(base_c * 8, base_c * 8)  # 情况2
        self.left_conv4 = DepthwiseSeparableConv(base_c * 8, base_c * 16, kernel_size=3, padding=1)
        self.right_conv4 = nn.Sequential(
            DepthwiseSeparableConv(base_c * 8, base_c * 16, kernel_size=7, padding=3),
            DepthwiseSeparableConv(base_c * 16, base_c * 16, kernel_size=5, padding=2),
        )
        self.conv4 = nn.Conv2d(base_c * 40, base_c * 16, kernel_size=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        # 底层加入DBEAM模块
        self.dbeam = DBEAM(channels=base_c * 16, r=4)


        self.left_conv5 = DepthwiseSeparableConv(base_c * 16, base_c * 8, kernel_size=3, padding=1)
        self.right_conv5 = nn.Sequential(
            DepthwiseSeparableConv(base_c * 16, base_c * 8, kernel_size=7, padding=3),
            DepthwiseSeparableConv(base_c * 8, base_c * 8, kernel_size=5, padding=2),
        )
        self.conv5 = nn.Conv2d(base_c * 16, base_c * 8, kernel_size=1)
        self.Upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.scffm_module1 = SCFFMModule(in_channels_low=base_c * 8, in_channels_high=base_c * 8, out_channels=base_c * 8,
                                     reduction=16)


        self.left_conv6 = DepthwiseSeparableConv(base_c * 16, base_c * 8, kernel_size=3, padding=1)
        self.right_conv6 = nn.Sequential(
            DepthwiseSeparableConv(base_c * 16, base_c * 8, kernel_size=7, padding=3),
            DepthwiseSeparableConv(base_c * 8, base_c * 8, kernel_size=5, padding=2),
        )
        self.conv6 = nn.Conv2d(base_c * 16, base_c * 4, kernel_size=1)
        self.Upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.scffm_module2 = SCFFMModule(in_channels_low=base_c * 4, in_channels_high=base_c * 4, out_channels=base_c * 4,
                                     reduction=16)


        self.left_conv7 = DepthwiseSeparableConv(base_c * 8, base_c * 4, kernel_size=3, padding=1)
        self.right_conv7 = nn.Sequential(
            DepthwiseSeparableConv(base_c * 8, base_c * 4, kernel_size=7, padding=3),
            DepthwiseSeparableConv(base_c * 4, base_c * 4, kernel_size=5, padding=2),
        )
        self.conv7 = nn.Conv2d(base_c * 8, base_c * 2, kernel_size=1)
        self.Upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.scffm_module3 = SCFFMModule(in_channels_low=base_c * 2, in_channels_high=base_c * 2, out_channels=base_c * 2,
                                     reduction=16)


        self.left_conv8 = DepthwiseSeparableConv(base_c * 4, base_c * 2, kernel_size=3, padding=1)
        self.right_conv8 = nn.Sequential(
            DepthwiseSeparableConv(base_c * 4, base_c * 2, kernel_size=7, padding=3),
            DepthwiseSeparableConv(base_c * 2, base_c * 2, kernel_size=5, padding=2),
        )
        self.conv8 = nn.Conv2d(base_c * 4, base_c, kernel_size=1)
        self.Upsample4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.scffm_module4 = SCFFMModule(in_channels_low=base_c, in_channels_high=base_c, out_channels=base_c, reduction=16)   # 32


        self.left_conv9 = DepthwiseSeparableConv(base_c * 2, base_c, kernel_size=3, padding=1)
        self.right_conv9 = nn.Sequential(
            DepthwiseSeparableConv(base_c * 2, base_c, kernel_size=7, padding=3),
            DepthwiseSeparableConv(base_c, base_c, kernel_size=5, padding=2),
        )

        self.conv9 = nn.Conv2d(base_c * 2, base_c, kernel_size=1)
        self.conv_output = nn.Conv2d(base_c, num_classes, kernel_size=1)


    def forward(self, x):
        x = self.conv_input(x)

        left_out1 = self.left_conv1(x)
        right_out1 = self.right_conv1(x)
        residual1 = self.residual1(x)
        out1 = torch.cat([left_out1, right_out1, residual1], dim=1)
        out1 = self.conv1(out1)
        out1 = self.maxpool1(out1)


        left_out2 = self.left_conv2(out1)
        right_out2 = self.right_conv2(out1)
        residual2 = self.residual2(out1)
        out2 = torch.cat([left_out2, right_out2, residual2], dim=1)
        out2 = self.conv2(out2)
        out2 = self.maxpool2(out2)


        left_out3 = self.left_conv3(out2)
        right_out3 = self.right_conv3(out2)
        residual3 = self.residual3(out2)
        out3 = torch.cat([left_out3, right_out3, residual3], dim=1)
        out3 = self.conv3(out3)
        out3 = self.maxpool3(out3)


        left_out4 = self.left_conv4(out3)
        right_out4 = self.right_conv4(out3)
        residual4 = self.residual4(out3)
        out4 = torch.cat([left_out4, right_out4, residual4], dim=1)
        out4 = self.conv4(out4)
        out4 = self.maxpool4(out4)

        # 加入DBEAM模块
        out4_dbeam = self.dbeam(out4)


        left_out5 = self.left_conv5(out4_dbeam)
        right_out5 = self.right_conv5(out4_dbeam)
        out5 = torch.cat([left_out5, right_out5], dim=1)
        out5 = self.conv5(out5)
        out5 = self.Upsample1(out5)

        if out3.size()[2:] != out5.size()[2:]:
            out5 = F.interpolate(out5, size=out3.size()[2:], mode='nearest')

        # 添加SCFFM模块
        out_scffm1 = self.scffm_module1(out3, out5)
        out6 = torch.cat([out3, out_scffm1], dim=1)
        left_out6 = self.left_conv6(out6)
        right_out6 = self.right_conv6(out6)
        out6 = torch.cat([left_out6, right_out6], dim=1)
        out6 = self.conv6(out6)
        out6 = self.Upsample2(out6)

        if out2.size()[2:] != out6.size()[2:]:
            out6 = F.interpolate(out6, size=out2.size()[2:], mode='nearest')

        # 添加SCFFM模块
        out_scffm2 = self.scffm_module2(out2, out6)
        out7 = torch.cat([out2, out_scffm2], dim=1)

        left_out7 = self.left_conv7(out7)
        right_out7 = self.right_conv7(out7)
        out7 = torch.cat([left_out7, right_out7], dim=1)
        out7 = self.conv7(out7)
        out7 = self.Upsample3(out7)

        if out1.size()[2:] != out7.size()[2:]:
            out7 = F.interpolate(out7, size=out1.size()[2:], mode='nearest')

        # 添加SCFFM模块
        out_scffm3 = self.scffm_module3(out1, out7)
        out8 = torch.cat([out1, out_scffm3], dim=1)

        left_out8 = self.left_conv8(out8)
        right_out8 = self.right_conv8(out8)
        out8 = torch.cat([left_out8, right_out8], dim=1)
        out8 = self.conv8(out8)
        out8 = self.Upsample4(out8)

        if x.size()[2:] != out8.size()[2:]:
            out8 = F.interpolate(out8, size=x.size()[2:], mode='nearest')

        # 添加SCFFM模块
        out_scffm4 = self.scffm_module4(x, out8)
        out9 = torch.cat([x, out_scffm4], dim=1)

        left_out9 = self.left_conv9(out9)
        right_out9 = self.right_conv9(out9)
        out9 = torch.cat([left_out9, right_out9], dim=1)
        out9 = self.conv9(out9)

        out = self.conv_output(out9)
        return {"out": out}
