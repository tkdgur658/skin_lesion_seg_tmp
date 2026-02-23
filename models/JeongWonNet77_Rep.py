# The code will release soon!!

import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_
# Re-parameterizable Conv Block
class RepConv(nn.Module):
    """
    Re-parameterizable Convolution Block
    훈련: Conv + BN + Identity(or 1x1) branch
    추론: 단일 3x3 Conv로 융합
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, groups=1, use_identity=True, use_activation=True):
        super(RepConv, self).__init__()
       
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
       
        # use_identity는 stride=1이고 in/out 채널이 같을 때만 활성화
        self.use_identity = use_identity and (stride == 1) and (in_channels == out_channels)
       
        # 주 Branch: kernel_size Conv + BN
        self.conv_kxk = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride, padding, groups=groups, bias=False)
        self.bn_kxk = nn.BatchNorm2d(out_channels)
       
        # 1x1 Branch (더 많은 표현력)
        if kernel_size > 1:
            self.conv_1x1 = nn.Conv2d(in_channels, out_channels, 1,
                                      stride, 0, groups=groups, bias=False)
            self.bn_1x1 = nn.BatchNorm2d(out_channels)
        else:
            self.conv_1x1 = None
       
        # Identity Branch (residual connection)
        if self.use_identity:
            self.bn_identity = nn.BatchNorm2d(out_channels)
       
        # 활성화 함수 선택적 적용
        if use_activation:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.Identity()
           
    def forward(self, x):
        if hasattr(self, 'fused_conv'):
            # 추론 모드: 융합된 단일 Conv만 사용
            return self.activation(self.fused_conv(x))
       
        # 훈련 모드: 모든 branch 합산
        out = self.bn_kxk(self.conv_kxk(x))
       
        if self.conv_1x1 is not None:
            out += self.bn_1x1(self.conv_1x1(x))
       
        if self.use_identity:
            out += self.bn_identity(x)
       
        return self.activation(out)
   
    def switch_to_deploy(self):
        """추론 모드로 전환: 모든 branch를 단일 Conv로 융합"""
        if hasattr(self, 'fused_conv'):
            return
       
        # 각 branch의 weight와 bias를 추출하여 합산
        kernel, bias = self._fuse_bn_tensor(self.conv_kxk, self.bn_kxk)
       
        if self.conv_1x1 is not None:
            kernel_1x1, bias_1x1 = self._fuse_bn_tensor(self.conv_1x1, self.bn_1x1)
            # 1x1을 kxk로 패딩
            kernel += self._pad_1x1_to_kxk(kernel_1x1)
            bias += bias_1x1
       
        if self.use_identity:
            kernel_identity, bias_identity = self._fuse_bn_tensor(None, self.bn_identity)
            kernel += kernel_identity
            bias += bias_identity
       
        # 융합된 Conv 생성
        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, self.padding, groups=self.groups, bias=True
        )
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias
       
        # 훈련용 레이어 제거 (메모리 절약)
        self.__delattr__('conv_kxk')
        self.__delattr__('bn_kxk')
        if self.conv_1x1 is not None:
            self.__delattr__('conv_1x1')
            self.__delattr__('bn_1x1')
        if hasattr(self, 'bn_identity'):
            self.__delattr__('bn_identity')
   
    def _fuse_bn_tensor(self, conv, bn):
        """Conv + BN을 융합하여 weight, bias 반환"""
        if conv is None:
            # Identity branch
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
        """1x1 kernel을 kxk로 패딩"""
        if self.kernel_size == 1:
            return kernel_1x1
        else:
            pad = self.kernel_size // 2
            return F.pad(kernel_1x1, [pad, pad, pad, pad])

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


class JeongWonNet77_Rep(nn.Module):
    """
    Simplified Hybrid Depthwise Separable UNet with PRCM:
    - UCMNet 스타일의 단순한 skip connection (decoder + encoder)
    - Bridge connection 제거로 명확한 구조
    - 1×1 Pointwise Conv (채널 조정)
    - 3×3 Depthwise Conv (groups=out_ch)
    - GroupNorm + GELU
    - PRCM으로 채널 간 중요 패턴 강조
    """
    def __init__(self, num_classes=1, input_channels=3, c_list=[6,12,18,24,32,48], gt_ds=True):
        super().__init__()
        self.gt_ds = gt_ds

        def make_dw_block(in_ch, out_ch):
            layers = []
            # 채널 조정이 필요할 때만 1×1 pointwise conv
            if in_ch != out_ch:
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))
            # Depthwise convolution
            layers.append(RepConv(out_ch, out_ch, kernel_size=7, padding=3, groups=out_ch))
            layers.append(PRCM(out_ch, num_basis=2))
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
