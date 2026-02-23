# The code will release soon!!

import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_
class EfficientDownsample(nn.Module):
    """
    EfficientFormer 방식의 학습 가능한 downsampling
    - Max pooling 대신 stride=2 depthwise conv 사용
    - Hardware-efficient 3×3 conv (Winograd 가속화 가능)
    - 정보 손실 최소화 + 학습 가능한 특징 추출
    """
    def __init__(self, in_channels, out_channels, use_prcm=True, num_basis=8):
        super().__init__()
        
        # 1. Depthwise conv with stride 2 (downsampling)
        self.dwconv = RepConv(
            in_channels, in_channels, 
            kernel_size=3,  # 3×3이 mobile GPU에서 가장 빠름
            stride=2,       # downsampling
            padding=1,
            groups=in_channels,  # depthwise
            use_identity=False,  # stride=2라서 identity 불가
            use_activation=True
        )
        
        # 2. Pointwise conv (channel 조정)
        if in_channels != out_channels:
            self.pwconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.pwconv = None
            self.bn = None
        
        # 3. Optional PRCM for channel recalibration
        self.prcm = PRCM(out_channels, num_basis) if use_prcm else None
        
    def forward(self, x):
        # Depthwise downsampling
        out = self.dwconv(x)
        
        # Channel adjustment
        if self.pwconv is not None:
            out = self.bn(self.pwconv(out))
        
        # Channel recalibration
        if self.prcm is not None:
            out = self.prcm(out)
            
        return out
        
class RepConv(nn.Module):
    """
    Re-parameterizable Convolution Block
    훈련: Conv + BN + Identity(or 1x1) branchF.interpolate
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
    """AWQ의 activation-aware 개념 적용"""
    def __init__(self, channels, num_basis=8):
        super().__init__()
        self.num_basis = num_basis
        
        # AWQ처럼 activation 분포 기반 중요도 학습
        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        self.fuser = nn.Linear(num_basis, channels, bias=False)
        
        # AWQ의 per-channel scaling 추가
        self.scale = nn.Parameter(torch.ones(channels))
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Activation magnitude 측정 (AWQ 방식)
        act_mag = x.abs().mean(dim=[2, 3])  # [B, C]
        
        # Basis projection with activation-awareness
        ctx = x.mean(dim=[2, 3])
        coeff = ctx @ self.basis.t()
        
        # AWQ의 salient channel protection
        w = self.fuser(coeff).sigmoid()
        
        # Activation-aware scaling (AWQ 핵심)
        w = w * self.scale.unsqueeze(0)
        
        return x * w.unsqueeze(-1).unsqueeze(-1)

class JeongWonNet77_Rep256Basis8S24_EffDownAWQ(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, c_list=[24, 48, 64, 96, 128, 192], gt_ds=True):
        super().__init__()
        self.gt_ds = gt_ds

        def make_dw_block(in_ch, out_ch):
            layers = []
            if in_ch != out_ch:
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))
            layers.append(RepConv(out_ch, out_ch, kernel_size=7, padding=3, groups=out_ch))
            layers.append(PRCM(out_ch, num_basis=8))
            return nn.Sequential(*layers)

        # Encoder blocks
        self.encoder1 = make_dw_block(input_channels, c_list[0])
        self.down1 = EfficientDownsample(c_list[0], c_list[0], use_prcm=False)
        
        self.encoder2 = make_dw_block(c_list[0], c_list[1])
        self.down2 = EfficientDownsample(c_list[1], c_list[1], use_prcm=False)
        
        self.encoder3 = make_dw_block(c_list[1], c_list[2])
        self.down3 = EfficientDownsample(c_list[2], c_list[2], use_prcm=False)
        
        self.encoder4 = make_dw_block(c_list[2], c_list[3])
        self.down4 = EfficientDownsample(c_list[3], c_list[3], use_prcm=False)
        
        self.encoder5 = make_dw_block(c_list[3], c_list[4])
        self.down5 = EfficientDownsample(c_list[4], c_list[4], use_prcm=False)
        
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

    def forward(self, x):
        is_eval = not self.training
        
        # ===== Encoder =====
        e1 = self.encoder1(x)                    # [B, c0, H, W]
        e1_down = self.down1(e1)                 # [B, c0, H/2, W/2] ← skip용
        
        e2 = self.encoder2(e1_down)              # [B, c1, H/2, W/2]
        e2_down = self.down2(e2)                 # [B, c1, H/4, W/4] ← skip용
        
        e3 = self.encoder3(e2_down)              # [B, c2, H/4, W/4]
        e3_down = self.down3(e3)                 # [B, c2, H/8, W/8] ← skip용
        
        e4 = self.encoder4(e3_down)              # [B, c3, H/8, W/8]
        e4_down = self.down4(e4)                 # [B, c3, H/16, W/16] ← skip용
        
        e5 = self.encoder5(e4_down)              # [B, c4, H/16, W/16]
        e5_down = self.down5(e5)                 # [B, c4, H/32, W/32] ← skip용
        
        e6 = self.encoder6(e5_down)              # [B, c5, H/32, W/32]

        # ===== Decoder with corrected skip connections =====
        d5 = self.decoder1(e6)                   # [B, c4, H/32, W/32]
        d5 = d5 + e5_down                        # ✅ 둘 다 H/32, W/32

        d4 = F.interpolate(self.decoder2(d5), scale_factor=2, mode='bilinear', align_corners=True)  
        # [B, c3, H/16, W/16]
        d4 = d4 + e4_down                        # ✅ 둘 다 H/16, W/16

        d3 = F.interpolate(self.decoder3(d4), scale_factor=2, mode='bilinear', align_corners=True)  
        # [B, c2, H/8, W/8]
        d3 = d3 + e3_down                        # ✅ 둘 다 H/8, W/8

        d2 = F.interpolate(self.decoder4(d3), scale_factor=2, mode='bilinear', align_corners=True)  
        # [B, c1, H/4, W/4]
        d2 = d2 + e2_down                        # ✅ 둘 다 H/4, W/4

        d1 = F.interpolate(self.decoder5(d2), scale_factor=2, mode='bilinear', align_corners=True)  
        # [B, c0, H/2, W/2]
        d1 = d1 + e1_down                        # ✅ 둘 다 H/2, W/2

        # Final output
        out = F.interpolate(self.final(d1), scale_factor=2, mode='bilinear', align_corners=True)    
        # [B, num_classes, H, W]

        # Deep supervision
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
