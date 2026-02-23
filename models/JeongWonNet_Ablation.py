"""
JeongWonNet Ablation Study Variants
Comparing JeongWonNet77_Rep256Basis8S24Drop vs JeongWonNet_STMShuffle

Key differences:
1. Split-Transform-Merge (STM) vs Full Processing
2. PRCM (multiplicative) vs AffinePRCM (affine: scale + shift)
3. Channel Shuffle
4. Stem layer

Variants:
- Ablation_STM: Rep256 + STM (no shuffle, no affine)
- Ablation_Shuffle: Rep256 + Shuffle (no STM, no affine)
- Ablation_Affine: Rep256 + AffinePRCM (no STM, no shuffle)
- Ablation_STMAffine: Rep256 + STM + AffinePRCM (no shuffle)
- Ablation_Full: Rep256 + STM + Shuffle + AffinePRCM (= STMShuffle)
"""

import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_


class RepConv(nn.Module):
    """Re-parameterizable Convolution Block"""
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, groups=1, use_identity=True, use_activation=True):
        super().__init__()
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

        self.activation = nn.ReLU(inplace=True) if use_activation else nn.Identity()

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
        else:
            kernel = conv.weight

        std = torch.sqrt(bn.running_var + bn.eps)
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * bn.weight / std

    def _pad_1x1_to_kxk(self, kernel_1x1):
        if self.kernel_size == 1:
            return kernel_1x1
        pad = self.kernel_size // 2
        return F.pad(kernel_1x1, [pad, pad, pad, pad])


class PRCM(nn.Module):
    """Original PRCM: Multiplicative scaling only"""
    def __init__(self, channels, num_basis=8, dropout_rate=0.5):
        super().__init__()
        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        self.fuser = nn.Linear(num_basis, channels, bias=False)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        ctx = x.mean(dim=[2, 3])
        coeff = self.coeff_dropout(ctx @ self.basis.t())
        w = self.fuser(coeff).sigmoid().unsqueeze(-1).unsqueeze(-1)
        return x * w


class AffinePRCM(nn.Module):
    """Affine PRCM: Scale + Shift"""
    def __init__(self, channels, num_basis=8, dropout_rate=0.5):
        super().__init__()
        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        self.scale_proj = nn.Linear(num_basis, channels, bias=False)
        self.shift_proj = nn.Linear(num_basis, channels, bias=False)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        ctx = x.mean(dim=[2, 3])
        coeff = self.coeff_dropout(ctx @ self.basis.t())
        alpha = self.scale_proj(coeff).sigmoid().unsqueeze(-1).unsqueeze(-1)
        beta = self.shift_proj(coeff).unsqueeze(-1).unsqueeze(-1)
        return x * alpha + beta


# =============================================================================
# Block Variants
# =============================================================================

class FullBlock(nn.Module):
    """Full processing block (no split) - like Rep256Basis8S24Drop"""
    def __init__(self, in_ch, out_ch, prcm_type='original', num_basis=8, dropout_rate=0.5):
        super().__init__()
        layers = []
        if in_ch != out_ch:
            layers.append(nn.Conv2d(in_ch, out_ch, 1, bias=False))
        layers.append(RepConv(out_ch, out_ch, kernel_size=7, padding=3, groups=out_ch))
        self.conv = nn.Sequential(*layers)

        if prcm_type == 'affine':
            self.prcm = AffinePRCM(out_ch, num_basis, dropout_rate)
        else:
            self.prcm = PRCM(out_ch, num_basis, dropout_rate)

    def forward(self, x):
        return self.prcm(self.conv(x))


class STMBlock(nn.Module):
    """Split-Transform-Merge block"""
    def __init__(self, in_ch, out_ch, prcm_type='original', use_shuffle=False,
                 num_basis=8, dropout_rate=0.5):
        super().__init__()
        self.use_shuffle = use_shuffle

        # Channel adjust: 홀수 입력(예: 3채널)은 groups=1, 짝수는 groups=2
        # BN + ReLU 추가하여 passive 브랜치도 비선형 활성화를 거치도록 함
        if in_ch != out_ch:
            groups = 2 if (in_ch % 2 == 0) else 1
            self.channel_adjust = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, groups=groups, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.channel_adjust = nn.Identity()

        half_ch = out_ch // 2
        self.dw_conv = RepConv(half_ch, half_ch, kernel_size=7, padding=3, groups=half_ch)

        if prcm_type == 'affine':
            self.prcm = AffinePRCM(half_ch, num_basis, dropout_rate)
        else:
            self.prcm = PRCM(half_ch, num_basis, dropout_rate)

    def channel_shuffle(self, x, groups=2):
        B, C, H, W = x.shape
        x = x.view(B, groups, C // groups, H, W)
        x = x.transpose(1, 2).contiguous()
        return x.view(B, C, H, W)

    def forward(self, x):
        x = self.channel_adjust(x)
        x_passive, x_active = torch.chunk(x, 2, dim=1)

        x_active = self.dw_conv(x_active)
        x_active = self.prcm(x_active)

        out = torch.cat([x_passive, x_active], dim=1)

        if self.use_shuffle:
            out = self.channel_shuffle(out)

        return out


# =============================================================================
# Ablation Model Factory
# =============================================================================

class JeongWonNet_AblationBase(nn.Module):
    """Base class for ablation variants"""
    def __init__(self, num_classes=1, input_channels=3, c_list=[24, 48, 64, 96, 128, 192],
                 use_stm=False, use_shuffle=False, prcm_type='original',
                 num_basis=8, dropout_rate=0.5, gt_ds=True):
        super().__init__()
        self.gt_ds = gt_ds

        def make_block(in_ch, out_ch):
            if use_stm:
                return STMBlock(in_ch, out_ch, prcm_type, use_shuffle, num_basis, dropout_rate)
            else:
                return FullBlock(in_ch, out_ch, prcm_type, num_basis, dropout_rate)

        # Encoder
        self.encoder1 = make_block(input_channels, c_list[0])
        self.encoder2 = make_block(c_list[0], c_list[1])
        self.encoder3 = make_block(c_list[1], c_list[2])
        self.encoder4 = make_block(c_list[2], c_list[3])
        self.encoder5 = make_block(c_list[3], c_list[4])
        self.encoder6 = make_block(c_list[4], c_list[5])

        # Deep Supervision
        if gt_ds:
            self.gt_conv1 = nn.Conv2d(c_list[4], num_classes, 1)
            self.gt_conv2 = nn.Conv2d(c_list[3], num_classes, 1)
            self.gt_conv3 = nn.Conv2d(c_list[2], num_classes, 1)
            self.gt_conv4 = nn.Conv2d(c_list[1], num_classes, 1)
            self.gt_conv5 = nn.Conv2d(c_list[0], num_classes, 1)

        # Decoder
        self.decoder1 = make_block(c_list[5], c_list[4])
        self.decoder2 = make_block(c_list[4], c_list[3])
        self.decoder3 = make_block(c_list[3], c_list[2])
        self.decoder4 = make_block(c_list[2], c_list[1])
        self.decoder5 = make_block(c_list[1], c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, 1)
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
        return out


# =============================================================================
# Concrete Ablation Variants (Main Models)
# =============================================================================

class JeongWonNet_Ablation_Baseline(JeongWonNet_AblationBase):
    """Baseline: Full + PRCM (= Rep256Basis8S24Drop equivalent)"""
    def __init__(self, num_classes=1, input_channels=3, c_list=[24, 48, 64, 96, 128, 192], gt_ds=True):
        super().__init__(num_classes, input_channels, c_list,
                        use_stm=False, use_shuffle=False, prcm_type='original', gt_ds=gt_ds)


class JeongWonNet_Ablation_STM(JeongWonNet_AblationBase):
    """Ablation: + STM (Split-Transform-Merge)"""
    def __init__(self, num_classes=1, input_channels=3, c_list=[24, 48, 64, 96, 128, 192], gt_ds=True):
        super().__init__(num_classes, input_channels, c_list,
                        use_stm=True, use_shuffle=False, prcm_type='original', gt_ds=gt_ds)


class JeongWonNet_Ablation_Affine(JeongWonNet_AblationBase):
    """Ablation: + AffinePRCM"""
    def __init__(self, num_classes=1, input_channels=3, c_list=[24, 48, 64, 96, 128, 192], gt_ds=True):
        super().__init__(num_classes, input_channels, c_list,
                        use_stm=False, use_shuffle=False, prcm_type='affine', gt_ds=gt_ds)


class JeongWonNet_Ablation_Shuffle(JeongWonNet_AblationBase):
    """Ablation: + STM + Shuffle"""
    def __init__(self, num_classes=1, input_channels=3, c_list=[24, 48, 64, 96, 128, 192], gt_ds=True):
        super().__init__(num_classes, input_channels, c_list,
                        use_stm=True, use_shuffle=True, prcm_type='original', gt_ds=gt_ds)


class JeongWonNet_Ablation_STMAffine(JeongWonNet_AblationBase):
    """Ablation: + STM + AffinePRCM"""
    def __init__(self, num_classes=1, input_channels=3, c_list=[24, 48, 64, 96, 128, 192], gt_ds=True):
        super().__init__(num_classes, input_channels, c_list,
                        use_stm=True, use_shuffle=False, prcm_type='affine', gt_ds=gt_ds)


class JeongWonNet_Ablation_Full(JeongWonNet_AblationBase):
    """Full: STM + Shuffle + AffinePRCM (= STMShuffle equivalent)"""
    def __init__(self, num_classes=1, input_channels=3, c_list=[24, 48, 64, 96, 128, 192], gt_ds=True):
        super().__init__(num_classes, input_channels, c_list,
                        use_stm=True, use_shuffle=True, prcm_type='affine', gt_ds=gt_ds)


if __name__ == "__main__":
    print("=" * 60)
    print("Ablation Study Variants")
    print("=" * 60)

    variants = [
        ("Baseline (Full+PRCM)", JeongWonNet_Ablation_Baseline),
        ("+STM", JeongWonNet_Ablation_STM),
        ("+Affine", JeongWonNet_Ablation_Affine),
        ("+STM+Shuffle", JeongWonNet_Ablation_Shuffle),
        ("+STM+Affine", JeongWonNet_Ablation_STMAffine),
        ("Full (STM+Shuffle+Affine)", JeongWonNet_Ablation_Full),
    ]

    x = torch.randn(2, 3, 256, 256)

    for name, ModelClass in variants:
        model = ModelClass()
        model.eval()
        with torch.no_grad():
            out = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:30s} | Params: {params:,} | Output: {out.shape}")
