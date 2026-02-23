"""
JeongWonNet_CtxBridge_StdExp_TConv with AMNet-style Deep Blocks (Deterministic)

깊은 레이어(enc5-6, dec1-2)의 DWBlock_Exp를 AMNet 인코더/디코더 블록으로 대체

얕은 레이어 (encoder1-4, decoder3-5):
    pw(1x1) → dw(7x7) → std(3x3) → prcm  (기존 DWBlock_Std 유지)

깊은 인코더 (encoder5-6) — AMNet encoder block:
    ├── residual: BasicBlock(in, in)
    ├── left:  DWSepConv(in, out, k=3)
    └── right: DWSepConv(in, out, k=7) → DWSepConv(out, out, k=5)
    → cat → 1x1 merge → prcm

깊은 디코더 (decoder1-2) — AMNet decoder block:
    ├── left:  DWSepConv(in, out, k=3)
    └── right: DWSepConv(in, out, k=7) → DWSepConv(out, out, k=5)
    → cat → 1x1 merge → prcm_bridge

업샘플링: ConvTranspose2d + BN (deterministic)
"""
import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_


# ===================== Shared Blocks =====================

class RepConv(nn.Module):
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
            out = out + self.bn_1x1(self.conv_1x1(x))

        if self.use_identity:
            out = out + self.bn_identity(x)

        return self.activation(out)

    def switch_to_deploy(self):
        if hasattr(self, 'fused_conv'):
            return

        kernel, bias = self._fuse_bn_tensor(self.conv_kxk, self.bn_kxk)

        if self.conv_1x1 is not None:
            kernel_1x1, bias_1x1 = self._fuse_bn_tensor(self.conv_1x1, self.bn_1x1)
            kernel = kernel + self._pad_1x1_to_kxk(kernel_1x1)
            bias = bias + bias_1x1

        if self.use_identity:
            kernel_identity, bias_identity = self._fuse_bn_tensor(None, self.bn_identity)
            kernel = kernel + kernel_identity
            bias = bias + bias_identity

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


# ===================== PRCM =====================

class PRCM(nn.Module):
    def __init__(self, channels, num_basis=8, dropout_rate=0.5):
        super().__init__()
        self.num_basis = num_basis
        self.channels = channels

        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        self.fuser = nn.Linear(num_basis, channels, bias=False)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x, return_ctx=False):
        B, C, H, W = x.shape

        ctx = x.mean(dim=[2, 3])
        coeff = ctx @ self.basis.t()
        coeff = self.coeff_dropout(coeff)

        w = self.fuser(coeff).sigmoid().unsqueeze(-1).unsqueeze(-1)

        if return_ctx:
            return x * w, ctx
        return x * w


class PRCM_Bridge(nn.Module):
    def __init__(self, channels, enc_channels, num_basis=8, dropout_rate=0.5, bridge_dropout=0.2):
        super().__init__()
        self.num_basis = num_basis
        self.channels = channels

        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        self.fuser = nn.Linear(num_basis, channels, bias=False)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        if enc_channels != channels:
            self.ctx_proj = nn.Linear(enc_channels, channels, bias=False)
        else:
            self.ctx_proj = nn.Identity()

        self.bridge_dropout = nn.Dropout(bridge_dropout) if bridge_dropout > 0 else nn.Identity()

    def forward(self, x, ctx_enc=None):
        B, C, H, W = x.shape

        ctx_self = x.mean(dim=[2, 3])

        if ctx_enc is not None:
            ctx_enc_proj = self.ctx_proj(ctx_enc)
            ctx_enc_proj = self.bridge_dropout(ctx_enc_proj)
            ctx_fused = ctx_self + ctx_enc_proj
        else:
            ctx_fused = ctx_self

        coeff = ctx_fused @ self.basis.t()
        coeff = self.coeff_dropout(coeff)

        w = self.fuser(coeff).sigmoid().unsqueeze(-1).unsqueeze(-1)
        return x * w


# ===================== AMNet Components =====================

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


class BasicBlock(nn.Module):
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


# ===================== AMNet-style Deep Blocks =====================

class AMNetEncBlock(nn.Module):
    """AMNet-style 3-branch encoder block + PRCM (for deep layers)

    ├── residual: BasicBlock(in_ch, in_ch)
    ├── left:  DWSepConv(in_ch, out_ch, k=3)
    └── right: DWSepConv(in_ch, out_ch, k=7) → DWSepConv(out_ch, out_ch, k=5)
    → cat(left, right, residual) → 1x1 merge → prcm
    """
    def __init__(self, in_channels, out_channels, num_basis=8, dropout_rate=0.5):
        super().__init__()

        self.residual = BasicBlock(in_channels, in_channels)

        self.left_conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1)
        self.right_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=7, padding=3),
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=5, padding=2),
        )

        cat_channels = in_channels + out_channels + out_channels
        self.merge = nn.Conv2d(cat_channels, out_channels, kernel_size=1)

        self.prcm = PRCM(out_channels, num_basis=num_basis, dropout_rate=dropout_rate)

    def forward(self, x, return_ctx=False):
        res = self.residual(x)
        left = self.left_conv(x)
        right = self.right_conv(x)

        out = torch.cat([left, right, res], dim=1)
        out = self.merge(out)

        if return_ctx:
            return self.prcm(out, return_ctx=True)
        return self.prcm(out)


class AMNetDecBlock_Bridge(nn.Module):
    """AMNet-style 2-branch decoder block + PRCM_Bridge (for deep layers)

    ├── left:  DWSepConv(in_ch, out_ch, k=3)
    └── right: DWSepConv(in_ch, out_ch, k=7) → DWSepConv(out_ch, out_ch, k=5)
    → cat(left, right) → 1x1 merge → prcm_bridge
    """
    def __init__(self, in_channels, out_channels, enc_channels, num_basis=8, dropout_rate=0.5, bridge_dropout=0.2):
        super().__init__()

        self.left_conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1)
        self.right_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=7, padding=3),
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=5, padding=2),
        )

        self.merge = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

        self.prcm = PRCM_Bridge(out_channels, enc_channels, num_basis=num_basis, dropout_rate=dropout_rate, bridge_dropout=bridge_dropout)

    def forward(self, x, ctx_enc=None):
        left = self.left_conv(x)
        right = self.right_conv(x)

        out = torch.cat([left, right], dim=1)
        out = self.merge(out)

        return self.prcm(out, ctx_enc)


# ===================== Shallow DWBlocks (unchanged) =====================

class DWBlock_Std(nn.Module):
    """DWBlock with 3x3 Standard Conv (for shallow layers)"""
    def __init__(self, in_channels, out_channels, kernel_size=7, num_basis=8, dropout_rate=0.5):
        super().__init__()

        if in_channels != out_channels:
            self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.pw_conv = None

        self.dw_conv = RepConv(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=out_channels
        )

        self.std_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.prcm = PRCM(out_channels, num_basis=num_basis, dropout_rate=dropout_rate)

    def forward(self, x, return_ctx=False):
        if self.pw_conv is not None:
            x = self.pw_conv(x)

        x = self.dw_conv(x)
        x = self.std_conv(x)

        if return_ctx:
            return self.prcm(x, return_ctx=True)
        return self.prcm(x)

    def switch_to_deploy(self):
        self.dw_conv.switch_to_deploy()


class DWBlock_Std_Bridge(nn.Module):
    """Decoder DWBlock with 3x3 Standard Conv (for shallow layers)"""
    def __init__(self, in_channels, out_channels, enc_channels, kernel_size=7, num_basis=8, dropout_rate=0.5, bridge_dropout=0.2):
        super().__init__()

        if in_channels != out_channels:
            self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.pw_conv = None

        self.dw_conv = RepConv(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=out_channels
        )

        self.std_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.prcm = PRCM_Bridge(out_channels, enc_channels, num_basis=num_basis, dropout_rate=dropout_rate, bridge_dropout=bridge_dropout)

    def forward(self, x, ctx_enc=None):
        if self.pw_conv is not None:
            x = self.pw_conv(x)

        x = self.dw_conv(x)
        x = self.std_conv(x)
        return self.prcm(x, ctx_enc)

    def switch_to_deploy(self):
        self.dw_conv.switch_to_deploy()


# ===================== Init =====================

def _init_weights(m):
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        fan_in //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_in))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# ===================== Model =====================

class JeongWonNet_CtxBridge_StdExp_TConv_AMDeep(nn.Module):
    """
    Context Bridge with Std Conv (shallow) + AMNet blocks (deep)
    Upsampling: ConvTranspose2d + BN (deterministic)

    Shallow (enc1-4, dec3-5): DWBlock_Std — pw → dw(7x7) → std(3x3) → prcm
    Deep encoder (enc5-6):    AMNetEncBlock — 3-branch(residual + left_3x3 + right_7x7→5x5) → merge → prcm
    Deep decoder (dec1-2):    AMNetDecBlock — 2-branch(left_3x3 + right_7x7→5x5) → merge → prcm_bridge
    """
    def __init__(self,
                 num_classes=1,
                 input_channels=3,
                 c_list=[24, 48, 64, 96, 128, 192],
                 kernel_size=7,
                 num_basis=8,
                 dropout_rate=0.5,
                 bridge_dropout=0.2,
                 gt_ds=True):
        super().__init__()
        self.gt_ds = gt_ds

        # Encoder - Shallow layers with Std Conv
        self.encoder1 = DWBlock_Std(input_channels, c_list[0], kernel_size, num_basis, dropout_rate)
        self.encoder2 = DWBlock_Std(c_list[0], c_list[1], kernel_size, num_basis, dropout_rate)
        self.encoder3 = DWBlock_Std(c_list[1], c_list[2], kernel_size, num_basis, dropout_rate)
        self.encoder4 = DWBlock_Std(c_list[2], c_list[3], kernel_size, num_basis, dropout_rate)
        # Deep layers with AMNet encoder block
        self.encoder5 = AMNetEncBlock(c_list[3], c_list[4], num_basis, dropout_rate)
        self.encoder6 = AMNetEncBlock(c_list[4], c_list[5], num_basis, dropout_rate)

        # Deep Supervision
        if gt_ds:
            self.gt_conv1 = nn.Conv2d(c_list[4], num_classes, 1)
            self.gt_conv2 = nn.Conv2d(c_list[3], num_classes, 1)
            self.gt_conv3 = nn.Conv2d(c_list[2], num_classes, 1)
            self.gt_conv4 = nn.Conv2d(c_list[1], num_classes, 1)
            self.gt_conv5 = nn.Conv2d(c_list[0], num_classes, 1)

        # Decoder - Deep layers with AMNet decoder block
        self.decoder1 = AMNetDecBlock_Bridge(c_list[5], c_list[4], c_list[4], num_basis, dropout_rate, bridge_dropout)
        self.decoder2 = AMNetDecBlock_Bridge(c_list[4], c_list[3], c_list[3], num_basis, dropout_rate, bridge_dropout)
        # Shallow layers with Std Conv
        self.decoder3 = DWBlock_Std_Bridge(c_list[3], c_list[2], c_list[2], kernel_size, num_basis, dropout_rate, bridge_dropout)
        self.decoder4 = DWBlock_Std_Bridge(c_list[2], c_list[1], c_list[1], kernel_size, num_basis, dropout_rate, bridge_dropout)
        self.decoder5 = DWBlock_Std_Bridge(c_list[1], c_list[0], c_list[0], kernel_size, num_basis, dropout_rate, bridge_dropout)

        # Transposed Convolution for upsampling
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(c_list[3], c_list[3], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(c_list[3])
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(c_list[2], c_list[2], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(c_list[2])
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(c_list[1], c_list[1], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(c_list[1])
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(c_list[0], c_list[0], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(c_list[0])
        )

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(_init_weights)

    def forward(self, x):
        is_eval = not self.training

        # Encoder
        e1_out, ctx1 = self.encoder1(x, return_ctx=True)
        e1 = F.max_pool2d(e1_out, 2)

        e2_out, ctx2 = self.encoder2(e1, return_ctx=True)
        e2 = F.max_pool2d(e2_out, 2)

        e3_out, ctx3 = self.encoder3(e2, return_ctx=True)
        e3 = F.max_pool2d(e3_out, 2)

        e4_out, ctx4 = self.encoder4(e3, return_ctx=True)
        e4 = F.max_pool2d(e4_out, 2)

        e5_out, ctx5 = self.encoder5(e4, return_ctx=True)
        e5 = F.max_pool2d(e5_out, 2)

        e6 = self.encoder6(e5)

        # Decoder
        d5 = self.decoder1(e6, ctx_enc=ctx5) + e5
        d4 = self.up2(self.decoder2(d5, ctx_enc=ctx4)) + e4
        d3 = self.up3(self.decoder3(d4, ctx_enc=ctx3)) + e3
        d2 = self.up4(self.decoder4(d3, ctx_enc=ctx2)) + e2
        d1 = self.up5(self.decoder5(d2, ctx_enc=ctx1)) + e1

        out = F.interpolate(self.final(d1), scale_factor=2, mode='nearest')

        if self.gt_ds and not is_eval:
            h, w = x.shape[2], x.shape[3]
            return (
                F.interpolate(self.gt_conv1(d5), (h, w), mode='nearest'),
                F.interpolate(self.gt_conv2(d4), (h, w), mode='nearest'),
                F.interpolate(self.gt_conv3(d3), (h, w), mode='nearest'),
                F.interpolate(self.gt_conv4(d2), (h, w), mode='nearest'),
                F.interpolate(self.gt_conv5(d1), (h, w), mode='nearest')
            ), out
        else:
            return out

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, (DWBlock_Std, DWBlock_Std_Bridge)):
                m.switch_to_deploy()


if __name__ == "__main__":
    model = JeongWonNet_CtxBridge_StdExp_TConv_AMDeep()
    x = torch.randn(2, 3, 256, 256)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compare with base model
    from JeongWonNet_CtxBridge_StdExp_TConv import JeongWonNet_CtxBridge_StdExp_TConv
    base = JeongWonNet_CtxBridge_StdExp_TConv()
    base_params = sum(p.numel() for p in base.parameters())
    am_params = sum(p.numel() for p in model.parameters())
    print(f"Base params:    {base_params:,}")
    print(f"AMDeep params:  {am_params:,}")
    print(f"Overhead:       {am_params - base_params:,} ({am_params / base_params * 100 - 100:+.1f}%)")
