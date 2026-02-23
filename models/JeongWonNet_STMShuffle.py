import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_


class RepConv(nn.Module):
    """Re-parameterizable Convolution Block"""
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, groups=1, use_identity=True, use_activation=True):
        super(RepConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.use_identity = use_identity and (stride == 1) and (in_channels == out_channels)
        
        self.conv_kxk = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride, padding, groups=groups, bias=False)
        self.bn_kxk = nn.BatchNorm2d(out_channels)
        
        if kernel_size > 1:
            self.conv_1x1 = nn.Conv2d(in_channels, out_channels, 1,
                                      stride, 0, groups=groups, bias=False)
            self.bn_1x1 = nn.BatchNorm2d(out_channels)
        else:
            self.conv_1x1 = None
        
        if self.use_identity:
            self.bn_identity = nn.BatchNorm2d(out_channels)
        
        if use_activation:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.Identity()
           
    def forward(self, x):
        if hasattr(self, 'fused_conv'):
            return self.activation(self.fused_conv(x))
        
        out = self.bn_kxk(self.conv_kxk(x))
        
        if self.conv_1x1 is not None:
            out += self.bn_1x1(self.conv_1x1(x))
        
        if self.use_identity:
            out += self.bn_identity(x)
        
        return self.activation(out)
   
    def switch_to_deploy(self):
        if hasattr(self, 'fused_conv'):
            return
        
        kernel, bias = self._fuse_bn_tensor(self.conv_kxk, self.bn_kxk)
        
        if self.conv_1x1 is not None:
            kernel_1x1, bias_1x1 = self._fuse_bn_tensor(self.conv_1x1, self.bn_1x1)
            kernel += self._pad_1x1_to_kxk(kernel_1x1)
            bias += bias_1x1
        
        if self.use_identity:
            kernel_identity, bias_identity = self._fuse_bn_tensor(None, self.bn_identity)
            kernel += kernel_identity
            bias += bias_identity
        
        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, self.padding, groups=self.groups, bias=True
        )
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias
        
        self.__delattr__('conv_kxk')
        self.__delattr__('bn_kxk')
        if self.conv_1x1 is not None:
            self.__delattr__('conv_1x1')
            self.__delattr__('bn_1x1')
        if hasattr(self, 'bn_identity'):
            self.__delattr__('bn_identity')
   
    def _fuse_bn_tensor(self, conv, bn):
        if conv is None:
            input_dim = self.in_channels // self.groups
            kernel_value = torch.zeros((self.in_channels, input_dim,
                                        self.kernel_size, self.kernel_size),
                                       dtype=bn.weight.dtype, device=bn.weight.device)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim,
                             self.kernel_size // 2, self.kernel_size // 2] = 1
            kernel = kernel_value
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
        else:
            kernel = conv.weight
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
        
        std = torch.sqrt(running_var + eps)
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
   
    def _pad_1x1_to_kxk(self, kernel_1x1):
        if self.kernel_size == 1:
            return kernel_1x1
        else:
            pad = self.kernel_size // 2
            return F.pad(kernel_1x1, [pad, pad, pad, pad])


class AffinePRCM(nn.Module):
    """Affine Modulation PRCM with Low-Rank Basis Reconstruction"""
    def __init__(self, channels, num_basis=8, dropout_rate=0.5):
        super().__init__()
        self.num_basis = num_basis
        self.channels = channels
        
        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        self.scale_proj = nn.Linear(num_basis, channels, bias=False)
        self.shift_proj = nn.Linear(num_basis, channels, bias=False)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        ctx = x.mean(dim=[2, 3])
        coeff = ctx @ self.basis.t()
        coeff = self.coeff_dropout(coeff)
        
        alpha = self.scale_proj(coeff).sigmoid().unsqueeze(-1).unsqueeze(-1)
        beta = self.shift_proj(coeff).unsqueeze(-1).unsqueeze(-1)
        
        return x * alpha + beta


class SplitTransformMergeBlock(nn.Module):
    """Split-Transform-Merge Block (ShuffleNet V2 inspired)"""
    def __init__(self, in_channels, out_channels, kernel_size=7, num_basis=8, dropout_rate=0.5):
        super().__init__()
        
        assert in_channels % 2 == 0, f"in_channels must be even, got {in_channels}"
        assert out_channels % 2 == 0, f"out_channels must be even, got {out_channels}"
        
        self.split_channels = in_channels // 2
        self.out_split_channels = out_channels // 2
        
        if self.split_channels != self.out_split_channels:
            self.passive_adjust = nn.Conv2d(
                self.split_channels, 
                self.out_split_channels, 
                kernel_size=1, 
                bias=False
            )
        else:
            self.passive_adjust = nn.Identity()
        
        if self.split_channels != self.out_split_channels:
            self.pw_conv = nn.Conv2d(
                self.split_channels, 
                self.out_split_channels, 
                kernel_size=1, 
                bias=False
            )
        else:
            self.pw_conv = nn.Identity()
        
        self.dw_repconv = RepConv(
            self.out_split_channels, 
            self.out_split_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2,
            groups=self.out_split_channels
        )
        
        self.affine_prcm = AffinePRCM(
            self.out_split_channels,
            num_basis=num_basis,
            dropout_rate=dropout_rate
        )
    def channel_shuffle(self, x, groups):
        # ShuffleNet의 핵심: 그룹 간 정보 교환
        B, C, H, W = x.data.size()
        channels_per_group = C // groups
        x = x.view(B, groups, channels_per_group, H, W)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(B, -1, H, W)
        return x
    def forward(self, x):
        x_passive, x_active = torch.chunk(x, 2, dim=1)
        
        x_passive = self.passive_adjust(x_passive)
        
        x_active = self.pw_conv(x_active)
        x_active = self.dw_repconv(x_active)
        x_active = self.affine_prcm(x_active)
        
        out = torch.cat([x_passive, x_active], dim=1)
        
        out = self.channel_shuffle(out, 2) 
        
        return out


class JeongWonNet_STMShuffle(nn.Module):
    """Split-Transform-Merge UNet with Affine Modulation PRCM"""
    def __init__(self, 
                 num_classes=1, 
                 input_channels=3, 
                 c_list=[24, 48, 64, 96, 128, 192],
                 num_basis=8,
                 dropout_rate=0.5,
                 gt_ds=True):
        super().__init__()
        self.gt_ds = gt_ds
        
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_list[0]),
            nn.ReLU(inplace=True)
        )
        
        self.encoder1 = SplitTransformMergeBlock(
            c_list[0], c_list[0], kernel_size=7, num_basis=num_basis, dropout_rate=dropout_rate
        )
        self.encoder2 = SplitTransformMergeBlock(
            c_list[0], c_list[1], kernel_size=7, num_basis=num_basis, dropout_rate=dropout_rate
        )
        self.encoder3 = SplitTransformMergeBlock(
            c_list[1], c_list[2], kernel_size=7, num_basis=num_basis, dropout_rate=dropout_rate
        )
        self.encoder4 = SplitTransformMergeBlock(
            c_list[2], c_list[3], kernel_size=7, num_basis=num_basis, dropout_rate=dropout_rate
        )
        self.encoder5 = SplitTransformMergeBlock(
            c_list[3], c_list[4], kernel_size=7, num_basis=num_basis, dropout_rate=dropout_rate
        )
        self.encoder6 = SplitTransformMergeBlock(
            c_list[4], c_list[5], kernel_size=7, num_basis=num_basis, dropout_rate=dropout_rate
        )
        
        if gt_ds:
            self.gt_conv1 = nn.Conv2d(c_list[4], num_classes, 1)
            self.gt_conv2 = nn.Conv2d(c_list[3], num_classes, 1)
            self.gt_conv3 = nn.Conv2d(c_list[2], num_classes, 1)
            self.gt_conv4 = nn.Conv2d(c_list[1], num_classes, 1)
            self.gt_conv5 = nn.Conv2d(c_list[0], num_classes, 1)
        
        self.decoder1 = SplitTransformMergeBlock(
            c_list[5], c_list[4], kernel_size=7, num_basis=num_basis, dropout_rate=dropout_rate
        )
        self.decoder2 = SplitTransformMergeBlock(
            c_list[4], c_list[3], kernel_size=7, num_basis=num_basis, dropout_rate=dropout_rate
        )
        self.decoder3 = SplitTransformMergeBlock(
            c_list[3], c_list[2], kernel_size=7, num_basis=num_basis, dropout_rate=dropout_rate
        )
        self.decoder4 = SplitTransformMergeBlock(
            c_list[2], c_list[1], kernel_size=7, num_basis=num_basis, dropout_rate=dropout_rate
        )
        self.decoder5 = SplitTransformMergeBlock(
            c_list[1], c_list[0], kernel_size=7, num_basis=num_basis, dropout_rate=dropout_rate
        )
        
        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        is_eval = not self.training
        
        x = self.stem(x)
        
        e1 = F.max_pool2d(self.encoder1(x), 2)
        e2 = F.max_pool2d(self.encoder2(e1), 2)
        e3 = F.max_pool2d(self.encoder3(e2), 2)
        e4 = F.max_pool2d(self.encoder4(e3), 2)
        e5 = F.max_pool2d(self.encoder5(e4), 2)
        e6 = self.encoder6(e5)
        
        d5 = self.decoder1(e6) + e5
        d4 = F.interpolate(self.decoder2(d5), scale_factor=2, mode='bilinear', align_corners=True) + e4
        d3 = F.interpolate(self.decoder3(d4), scale_factor=2, mode='bilinear', align_corners=True) + e3
        d2 = F.interpolate(self.decoder4(d3), scale_factor=2, mode='bilinear', align_corners=True) + e2
        d1 = F.interpolate(self.decoder5(d2), scale_factor=2, mode='bilinear', align_corners=True) + e1
        
        out = F.interpolate(self.final(d1), scale_factor=2, mode='bilinear', align_corners=True)
        
        if self.gt_ds and not is_eval:
            h, w = x.shape[2], x.shape[3]
            return (
                F.interpolate(self.gt_conv1(d5), (h, w), mode='bilinear', align_corners=True),
                F.interpolate(self.gt_conv2(d4), (h, w), mode='bilinear', align_corners=True),
                F.interpolate(self.gt_conv3(d3), (h, w), mode='bilinear', align_corners=True),
                F.interpolate(self.gt_conv4(d2), (h, w), mode='bilinear', align_corners=True),
                F.interpolate(self.gt_conv5(d1), (h, w), mode='bilinear', align_corners=True)
            ), out
        else:
            return out
