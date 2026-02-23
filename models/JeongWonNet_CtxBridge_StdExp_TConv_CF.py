"""
JeongWonNet_CtxBridge_StdExp_TConv_CF (Concat-First Decoder, Deterministic)

디코더 skip connection 방식을 "process → add"에서 "concat → process"로 변경
AMNet 디코더와 동일한 fusion 순서: encoder feat과 decoder feat를 먼저 합친 뒤 블록이 처리

기존:  d4 = up(decoder(d5)) + e4              ← 처리 후 합산
변경:  d4 = decoder(cat(up(d5), e4))          ← 합산 후 처리
       → 디코더 블록이 encoder 정보를 보면서 처리 가능

얕은 레이어 (encoder1-4, decoder3-5): std(3x3)
깊은 레이어 (encoder5-6, decoder1-2): expand(1x1, x4)
업샘플링: ConvTranspose2d + BN (deterministic)
"""
import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_


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


# ===================== Encoder Blocks =====================

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


class DWBlock_Exp(nn.Module):
    """DWBlock with 1x1 Expansion-Projection (for deep layers)"""
    def __init__(self, in_channels, out_channels, kernel_size=7, num_basis=8, dropout_rate=0.5, expansion=4):
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

        hidden_dim = out_channels * expansion
        self.exp_proj = nn.Sequential(
            nn.Conv2d(out_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.prcm = PRCM(out_channels, num_basis=num_basis, dropout_rate=dropout_rate)

    def forward(self, x, return_ctx=False):
        if self.pw_conv is not None:
            x = self.pw_conv(x)

        x = self.dw_conv(x)
        x = self.exp_proj(x)

        if return_ctx:
            return self.prcm(x, return_ctx=True)
        return self.prcm(x)

    def switch_to_deploy(self):
        self.dw_conv.switch_to_deploy()


# ===================== Decoder Blocks (Concat-First) =====================
# in_channels = decoder_ch + encoder_ch (concatenated input)

class DWBlock_Std_Bridge(nn.Module):
    """Decoder DWBlock with 3x3 Standard Conv (for shallow layers)"""
    def __init__(self, in_channels, out_channels, enc_channels, kernel_size=7, num_basis=8, dropout_rate=0.5, bridge_dropout=0.2):
        super().__init__()

        # pw_conv handles concatenated input channels → out_channels
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


class DWBlock_Exp_Bridge(nn.Module):
    """Decoder DWBlock with 1x1 Expansion-Projection (for deep layers)"""
    def __init__(self, in_channels, out_channels, enc_channels, kernel_size=7, num_basis=8, dropout_rate=0.5, bridge_dropout=0.2, expansion=4):
        super().__init__()

        # pw_conv handles concatenated input channels → out_channels
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

        hidden_dim = out_channels * expansion
        self.exp_proj = nn.Sequential(
            nn.Conv2d(out_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.prcm = PRCM_Bridge(out_channels, enc_channels, num_basis=num_basis, dropout_rate=dropout_rate, bridge_dropout=bridge_dropout)

    def forward(self, x, ctx_enc=None):
        if self.pw_conv is not None:
            x = self.pw_conv(x)

        x = self.dw_conv(x)
        x = self.exp_proj(x)
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

class JeongWonNet_CtxBridge_StdExp_TConv_CF(nn.Module):
    """
    Context Bridge with Std Conv (shallow) + Expansion (deep)
    Concat-First Decoder: cat(up(decoder_out), encoder_feat) → decoder block
    Upsampling: ConvTranspose2d + BN (deterministic)

    기존 "process → add":  d4 = up(decoder(d5)) + e4
    변경 "concat → process": d4 = decoder(cat(up(d5), e4))
        → 디코더 블록이 encoder+decoder feature를 동시에 보고 처리

    Shallow (enc1-4, dec3-5): std(3x3)
    Deep (enc5-6, dec1-2): expand(1x1, x4)
    """
    def __init__(self,
                 num_classes=1,
                 input_channels=3,
                 c_list=[24, 48, 64, 96, 128, 192],
                 kernel_size=7,
                 num_basis=8,
                 dropout_rate=0.5,
                 bridge_dropout=0.2,
                 expansion=4,
                 gt_ds=True):
        super().__init__()
        self.gt_ds = gt_ds

        # Encoder - Shallow layers with Std Conv
        self.encoder1 = DWBlock_Std(input_channels, c_list[0], kernel_size, num_basis, dropout_rate)
        self.encoder2 = DWBlock_Std(c_list[0], c_list[1], kernel_size, num_basis, dropout_rate)
        self.encoder3 = DWBlock_Std(c_list[1], c_list[2], kernel_size, num_basis, dropout_rate)
        self.encoder4 = DWBlock_Std(c_list[2], c_list[3], kernel_size, num_basis, dropout_rate)
        # Deep layers with Expansion
        self.encoder5 = DWBlock_Exp(c_list[3], c_list[4], kernel_size, num_basis, dropout_rate, expansion)
        self.encoder6 = DWBlock_Exp(c_list[4], c_list[5], kernel_size, num_basis, dropout_rate, expansion)

        # Deep Supervision
        if gt_ds:
            self.gt_conv1 = nn.Conv2d(c_list[4], num_classes, 1)
            self.gt_conv2 = nn.Conv2d(c_list[3], num_classes, 1)
            self.gt_conv3 = nn.Conv2d(c_list[2], num_classes, 1)
            self.gt_conv4 = nn.Conv2d(c_list[1], num_classes, 1)
            self.gt_conv5 = nn.Conv2d(c_list[0], num_classes, 1)

        # Decoder - Concat-First: in_channels = decoder_ch + encoder_skip_ch
        # Deep layers with Expansion
        self.decoder1 = DWBlock_Exp_Bridge(
            c_list[5] + c_list[4], c_list[4], c_list[4],  # cat(e6, e5) → 192+128=320 → 128
            kernel_size, num_basis, dropout_rate, bridge_dropout, expansion)
        self.decoder2 = DWBlock_Exp_Bridge(
            c_list[4] + c_list[3], c_list[3], c_list[3],  # cat(up(d5), e4) → 128+96=224 → 96
            kernel_size, num_basis, dropout_rate, bridge_dropout, expansion)
        # Shallow layers with Std Conv
        self.decoder3 = DWBlock_Std_Bridge(
            c_list[3] + c_list[2], c_list[2], c_list[2],  # cat(up(d4), e3) → 96+64=160 → 64
            kernel_size, num_basis, dropout_rate, bridge_dropout)
        self.decoder4 = DWBlock_Std_Bridge(
            c_list[2] + c_list[1], c_list[1], c_list[1],  # cat(up(d3), e2) → 64+48=112 → 48
            kernel_size, num_basis, dropout_rate, bridge_dropout)
        self.decoder5 = DWBlock_Std_Bridge(
            c_list[1] + c_list[0], c_list[0], c_list[0],  # cat(up(d2), e1) → 48+24=72 → 24
            kernel_size, num_basis, dropout_rate, bridge_dropout)

        # Transposed Convolution for upsampling (applied BEFORE concat)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(c_list[4], c_list[4], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(c_list[4])
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(c_list[3], c_list[3], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(c_list[3])
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(c_list[2], c_list[2], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(c_list[2])
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(c_list[1], c_list[1], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(c_list[1])
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

        # Decoder — Concat-First: cat(decoder/up, encoder) → process
        d5 = self.decoder1(torch.cat([e6, e5], dim=1), ctx_enc=ctx5)
        d4 = self.decoder2(torch.cat([self.up2(d5), e4], dim=1), ctx_enc=ctx4)
        d3 = self.decoder3(torch.cat([self.up3(d4), e3], dim=1), ctx_enc=ctx3)
        d2 = self.decoder4(torch.cat([self.up4(d3), e2], dim=1), ctx_enc=ctx2)
        d1 = self.decoder5(torch.cat([self.up5(d2), e1], dim=1), ctx_enc=ctx1)

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
            if isinstance(m, (DWBlock_Std, DWBlock_Exp, DWBlock_Std_Bridge, DWBlock_Exp_Bridge)):
                if hasattr(m, 'switch_to_deploy'):
                    m.switch_to_deploy()


if __name__ == "__main__":
    model = JeongWonNet_CtxBridge_StdExp_TConv_CF()
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
    cf_params = sum(p.numel() for p in model.parameters())
    print(f"Base params: {base_params:,}")
    print(f"CF params:   {cf_params:,}")
    print(f"Overhead:    {cf_params - base_params:,} ({cf_params / base_params * 100 - 100:+.1f}%)")
