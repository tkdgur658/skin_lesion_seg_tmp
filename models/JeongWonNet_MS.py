# The code will release soon!!

import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_
# Re-parameterizable Conv Block
class RepConv(nn.Module):
    """
    Multi-Scale Re-parameterizable Conv
    훈련: 3x3 + 5x5 + 1x1 + Identity
    추론: 단일 5x5 Conv로 융합
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5],
                 stride=1, groups=1, use_identity=True, use_activation=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_kernel = max(kernel_sizes)
        self.stride = stride
        self.padding = self.max_kernel // 2
        self.groups = groups
        self.kernel_sizes = sorted(kernel_sizes, reverse=True)
        
        self.use_identity = use_identity and (stride == 1) and (in_channels == out_channels)
        
        # 각 스케일별 Conv + BN
        self.branches = nn.ModuleList()
        for ks in self.kernel_sizes:
            pad = ks // 2
            conv = nn.Conv2d(in_channels, out_channels, ks, 
                           stride, pad, groups=groups, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.branches.append(nn.ModuleDict({'conv': conv, 'bn': bn}))
        
        # Identity branch
        if self.use_identity:
            self.bn_identity = nn.BatchNorm2d(out_channels)
        
        self.activation = nn.ReLU(inplace=True) if use_activation else nn.Identity()
    
    def forward(self, x):
        if hasattr(self, 'fused_conv'):
            return self.activation(self.fused_conv(x))
        
        # 모든 스케일 합산
        out = 0
        for branch in self.branches:
            out = out + branch['bn'](branch['conv'](x))
        
        if self.use_identity:
            out = out + self.bn_identity(x)
        
        return self.activation(out)
    
    def switch_to_deploy(self):
        if hasattr(self, 'fused_conv'):
            return
        
        # 모든 branch를 최대 커널 사이즈로 융합
        kernel = 0
        bias = 0
        
        for branch in self.branches:
            k, b = self._fuse_bn_tensor(branch['conv'], branch['bn'])
            kernel = kernel + self._pad_to_max_kernel(k, branch['conv'].kernel_size[0])
            bias = bias + b
        
        if self.use_identity:
            k_id, b_id = self._fuse_bn_tensor(None, self.bn_identity)
            kernel = kernel + k_id
            bias = bias + b_id
        
        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.max_kernel,
            self.stride, self.padding, groups=self.groups, bias=True
        )
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias
        
        # 메모리 정리
        self.__delattr__('branches')
        if hasattr(self, 'bn_identity'):
            self.__delattr__('bn_identity')
    
    def _fuse_bn_tensor(self, conv, bn):
        if conv is None:
            # Identity
            input_dim = self.in_channels // self.groups
            kernel_value = torch.zeros((self.in_channels, input_dim,
                                       self.max_kernel, self.max_kernel),
                                      dtype=bn.weight.dtype, device=bn.weight.device)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 
                           self.max_kernel // 2, self.max_kernel // 2] = 1
            kernel = kernel_value
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
    
    def _pad_to_max_kernel(self, kernel, current_size):
        if current_size == self.max_kernel:
            return kernel
        pad = (self.max_kernel - current_size) // 2
        return F.pad(kernel, [pad, pad, pad, pad])

class PRCM(nn.Module):
    """
    Parametric Resonance Channel Mixer (PRCM)
    - 각 채널별 글로벌 컨텍스트를 학습 가능한 low-rank basis로 분석
    - basis coefficient를 통해 채널 가중치를 생성해 중요 표현 강조
    """
    def __init__(self, channels, num_basis=2):
        super().__init__()
        self.num_basis = num_basis
        # 학습 가능한 basis 벡터: [num_basis, channels]
        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        # low-rank coefficient → channel 가중치 맵핑
        self.fuser = nn.Linear(num_basis, channels, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        # 채널별 전역 평균 컨텍스트: [B, C]
        ctx = x.mean(dim=[2, 3])
        # basis 투영: [B, num_basis]
        coeff = ctx @ self.basis.t()
        # 채널별 가중치 생성 및 sigmoid: [B, C, 1, 1]
        w = self.fuser(coeff).sigmoid().unsqueeze(-1).unsqueeze(-1)
        return x * w


class JeongWonNet_MS(nn.Module):
    """
    Simplified Hybrid Depthwise Separable UNet with PRCM:
    - UCMNet 스타일의 단순한 skip connection (decoder + encoder)
    - Bridge connection 제거로 명확한 구조
    - 1×1 Pointwise Conv (채널 조정)
    - 3×3 Depthwise Conv (groups=out_ch)
    - GroupNorm + GELU
    - PRCM으로 채널 간 중요 패턴 강조
    """
    def __init__(self, num_classes=1, input_channels=3, c_list=[24, 48, 64, 96, 128, 192], gt_ds=True):
        super().__init__()
        self.gt_ds = gt_ds

        def make_dw_block(in_ch, out_ch):
            layers = []
            # 채널 조정이 필요할 때만 1×1 pointwise conv
            if in_ch != out_ch:
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))
            # Depthwise convolution
            layers.append(RepConv(out_ch, out_ch, kernel_sizes=[1, 3, 5, 7],  groups=out_ch))
            layers.append(PRCM(out_ch, num_basis=8))
            return nn.Sequential(*layers)

        # Encoder blocks
        self.encoder1 = make_dw_block(input_channels, c_list[0])
        self.encoder2 = make_dw_block(c_list[0], c_list[1])
        self.encoder3 = make_dw_block(c_list[1], c_list[2])
        self.encoder4 = make_dw_block(c_list[2], c_list[3])
        self.encoder5 = make_dw_block(c_list[3], c_list[4])
        self.encoder6 = make_dw_block(c_list[4], c_list[5])

        # Deep Supervision heads
        if gt_ds:
            self.gt_conv1 = nn.Conv2d(c_list[4], num_classes, 1)
            self.gt_conv2 = nn.Conv2d(c_list[3], num_classes, 1)
            self.gt_conv3 = nn.Conv2d(c_list[2], num_classes, 1)
            self.gt_conv4 = nn.Conv2d(c_list[1], num_classes, 1)
            self.gt_conv5 = nn.Conv2d(c_list[0], num_classes, 1)

        # Decoder blocks
        self.decoder1 = make_dw_block(c_list[5], c_list[4])
        self.decoder2 = make_dw_block(c_list[4], c_list[3])
        self.decoder3 = make_dw_block(c_list[3], c_list[2])
        self.decoder4 = make_dw_block(c_list[2], c_list[1])
        self.decoder5 = make_dw_block(c_list[1], c_list[0])

        # Final 1x1 conv to num_classes
        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        # Initialize weights
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

    def forward(self, x):
        is_eval = not self.training
        
        # === Encoder forward ===
        e1 = F.max_pool2d(self.encoder1(x), 2)      # [B, c0, H/2, W/2]
        e2 = F.max_pool2d(self.encoder2(e1), 2)     # [B, c1, H/4, W/4]
        e3 = F.max_pool2d(self.encoder3(e2), 2)     # [B, c2, H/8, W/8]
        e4 = F.max_pool2d(self.encoder4(e3), 2)     # [B, c3, H/16, W/16]
        e5 = F.max_pool2d(self.encoder5(e4), 2)     # [B, c4, H/32, W/32]
        e6 = self.encoder6(e5)                      # [B, c5, H/32, W/32]

        # === Decoder forward with simple skip connections ===
        # d5: 가장 깊은 레벨, e5와 같은 해상도
        d5 = self.decoder1(e6)                      # [B, c4, H/32, W/32]
        d5 = d5 + e5                                # Simple skip connection

        # d4: upsampling 후 e4와 더함
        d4 = F.interpolate(self.decoder2(d5), scale_factor=2, mode='bilinear', align_corners=True)  # [B, c3, H/16, W/16]
        d4 = d4 + e4                                # Simple skip connection

        # d3: upsampling 후 e3와 더함
        d3 = F.interpolate(self.decoder3(d4), scale_factor=2, mode='bilinear', align_corners=True)  # [B, c2, H/8, W/8]
        d3 = d3 + e3                                # Simple skip connection

        # d2: upsampling 후 e2와 더함
        d2 = F.interpolate(self.decoder4(d3), scale_factor=2, mode='bilinear', align_corners=True)  # [B, c1, H/4, W/4]
        d2 = d2 + e2                                # Simple skip connection

        # d1: upsampling 후 e1과 더함
        d1 = F.interpolate(self.decoder5(d2), scale_factor=2, mode='bilinear', align_corners=True)  # [B, c0, H/2, W/2]
        d1 = d1 + e1                                # Simple skip connection

        # Final segmentation map
        out = F.interpolate(self.final(d1), scale_factor=2, mode='bilinear', align_corners=True)    # [B, num_classes, H, W]

        # Return deep supervision outputs if training
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
