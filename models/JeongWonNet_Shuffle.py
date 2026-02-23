"""
JeongWonNet_Shuffle

기본 블록: ShuffleDWBlock
- 왼쪽: 1x1 conv(stride=2) → BN
- 오른쪽: 1x1 pw → RepConv dw(stride=2) → [PRCM or expand→project→PRCM]
- concat → channel shuffle

exp_stages: 뒤에서 몇 개 스테이지에 expansion-projection을 쓸지 결정
  예) exp_stages=3, 총 6스테이지:
      encoder1,2,3 → PRCM만
      encoder4,5,6 → expand→project→PRCM
  decoder는 encoder와 대칭
"""

import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_


def channel_shuffle(x, groups=2):
    B, C, H, W = x.shape
    assert C % groups == 0
    x = x.view(B, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    return x.view(B, C, H, W)


class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, groups=1, use_identity=True, use_activation=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.use_identity = use_identity and (stride == 1) and (in_channels == out_channels)

        self.conv_kxk = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                   padding, groups=groups, bias=False)
        self.bn_kxk = nn.BatchNorm2d(out_channels)

        if kernel_size > 1:
            self.conv_1x1 = nn.Conv2d(in_channels, out_channels, 1, stride, 0,
                                       groups=groups, bias=False)
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
            out = out + self.bn_1x1(self.conv_1x1(x))
        if self.use_identity:
            out = out + self.bn_identity(x)
        return self.activation(out)

    def switch_to_deploy(self):
        if hasattr(self, 'fused_conv'):
            return
        kernel, bias = self._fuse_bn_tensor(self.conv_kxk, self.bn_kxk)
        if self.conv_1x1 is not None:
            k1, b1 = self._fuse_bn_tensor(self.conv_1x1, self.bn_1x1)
            kernel = kernel + self._pad_1x1_to_kxk(k1)
            bias = bias + b1
        if self.use_identity:
            ki, bi = self._fuse_bn_tensor(None, self.bn_identity)
            kernel = kernel + ki
            bias = bias + bi
        self.fused_conv = nn.Conv2d(self.in_channels, self.out_channels,
                                     self.kernel_size, self.stride, self.padding,
                                     groups=self.groups, bias=True)
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias
        del self.conv_kxk, self.bn_kxk
        if self.conv_1x1 is not None:
            del self.conv_1x1, self.bn_1x1
        if hasattr(self, 'bn_identity'):
            del self.bn_identity

    def _fuse_bn_tensor(self, conv, bn):
        if conv is None:
            input_dim = self.in_channels // self.groups
            kernel_value = torch.zeros(
                (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                dtype=bn.weight.dtype, device=bn.weight.device)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2] = 1
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
    def __init__(self, channels, num_basis=8, dropout_rate=0.5):
        super().__init__()
        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        self.fuser = nn.Linear(num_basis, channels, bias=False)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x, return_ctx=False):
        ctx = x.mean(dim=[2, 3])
        coeff = self.coeff_dropout(ctx @ self.basis.t())
        w = self.fuser(coeff).sigmoid().unsqueeze(-1).unsqueeze(-1)
        if return_ctx:
            return x * w, ctx
        return x * w


class PRCM_Bridge(nn.Module):
    def __init__(self, channels, enc_channels, num_basis=8,
                 dropout_rate=0.5, bridge_dropout=0.2):
        super().__init__()
        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        self.fuser = nn.Linear(num_basis, channels, bias=False)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.ctx_proj = nn.Linear(enc_channels, channels, bias=False) \
            if enc_channels != channels else nn.Identity()
        self.bridge_dropout = nn.Dropout(bridge_dropout) if bridge_dropout > 0 else nn.Identity()

    def forward(self, x, ctx_enc=None):
        ctx_self = x.mean(dim=[2, 3])
        if ctx_enc is not None:
            ctx_fused = ctx_self + self.bridge_dropout(self.ctx_proj(ctx_enc))
        else:
            ctx_fused = ctx_self
        coeff = self.coeff_dropout(ctx_fused @ self.basis.t())
        w = self.fuser(coeff).sigmoid().unsqueeze(-1).unsqueeze(-1)
        return x * w


# ─────────────────────────────────────────────
# ShuffleDWBlock (Encoder, stride=2 내장)
# ─────────────────────────────────────────────
class ShuffleDWBlock(nn.Module):
    """
    왼쪽: 1x1 conv(stride=2) → BN
    오른쪽: 1x1 pw → RepConv dw(stride=2) → [PRCM | expand→project→PRCM]

    use_exp=False: dw → PRCM
    use_exp=True : dw → expand(x4) → project → PRCM
    """
    def __init__(self, in_channels, out_channels, kernel_size=7,
                 num_basis=8, dropout_rate=0.5, use_exp=False, expansion=4):
        super().__init__()
        assert out_channels % 2 == 0
        half = out_channels // 2

        self.left = nn.Sequential(
            nn.Conv2d(in_channels, half, 1, stride=2, bias=False),
            nn.BatchNorm2d(half)
        )
        self.right_pw = nn.Sequential(
            nn.Conv2d(in_channels, half, 1, bias=False),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True)
        )
        self.right_dw = RepConv(half, half,
                                 kernel_size=kernel_size,
                                 stride=2,
                                 padding=kernel_size // 2,
                                 groups=half,
                                 use_identity=False)
        self.use_exp = use_exp
        if use_exp:
            hidden_dim = half * expansion
            self.exp_proj = nn.Sequential(
                nn.Conv2d(half, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, half, 1, bias=False),
                nn.BatchNorm2d(half),
                nn.ReLU(inplace=True)
            )

        self.prcm = PRCM(half, num_basis=num_basis, dropout_rate=dropout_rate)

    def forward(self, x, return_ctx=False):
        x_l = self.left(x)
        x_r = self.right_pw(x)
        x_r = self.right_dw(x_r)
        if self.use_exp:
            x_r = self.exp_proj(x_r)
        if return_ctx:
            x_r, ctx = self.prcm(x_r, return_ctx=True)
        else:
            x_r = self.prcm(x_r)
        out = channel_shuffle(torch.cat([x_l, x_r], dim=1))
        if return_ctx:
            return out, ctx
        return out


# ─────────────────────────────────────────────
# ShuffleDWBlock_Bridge (Decoder)
# ─────────────────────────────────────────────
class ShuffleDWBlock_Bridge(nn.Module):
    """
    Decoder용 - PRCM_Bridge 사용
    use_exp=False: dw → PRCM_Bridge
    use_exp=True : dw → expand→project → PRCM_Bridge
    """
    def __init__(self, in_channels, out_channels, enc_channels, kernel_size=7,
                 num_basis=8, dropout_rate=0.5, bridge_dropout=0.2,
                 use_exp=False, expansion=4):
        super().__init__()
        assert out_channels % 2 == 0
        half = out_channels // 2

        self.left = nn.Sequential(
            nn.Conv2d(in_channels, half, 1, bias=False),
            nn.BatchNorm2d(half)
        )
        self.right_pw = nn.Sequential(
            nn.Conv2d(in_channels, half, 1, bias=False),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True)
        )
        self.right_dw = RepConv(half, half,
                                 kernel_size=kernel_size,
                                 stride=1,
                                 padding=kernel_size // 2,
                                 groups=half)
        self.use_exp = use_exp
        if use_exp:
            hidden_dim = half * expansion
            self.exp_proj = nn.Sequential(
                nn.Conv2d(half, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, half, 1, bias=False),
                nn.BatchNorm2d(half),
                nn.ReLU(inplace=True)
            )

        self.prcm = PRCM_Bridge(half, enc_channels,
                                  num_basis=num_basis,
                                  dropout_rate=dropout_rate,
                                  bridge_dropout=bridge_dropout)

    def forward(self, x, ctx_enc=None):
        x_l = self.left(x)
        x_r = self.right_pw(x)
        x_r = self.right_dw(x_r)
        if self.use_exp:
            x_r = self.exp_proj(x_r)
        x_r = self.prcm(x_r, ctx_enc)
        return channel_shuffle(torch.cat([x_l, x_r], dim=1))


# ─────────────────────────────────────────────
# 가중치 초기화
# ─────────────────────────────────────────────
def _init_weights(m):
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


# ─────────────────────────────────────────────
# JeongWonNet_Shuffle
# ─────────────────────────────────────────────
class JeongWonNet_Shuffle(nn.Module):
    """
    exp_stages: 뒤에서 몇 개 스테이지에 expand→project를 쓸지 결정 (0~6)
      exp_stages=0 → 전 스테이지 PRCM만
      exp_stages=3 → encoder4,5,6 / decoder1,2,3 에 expand→project 적용
      exp_stages=6 → 전 스테이지 expand→project

    예시 (exp_stages=3):
      encoder1,2,3 → PRCM
      encoder4,5,6 → expand→project→PRCM
      decoder1,2,3 → expand→project→PRCM_Bridge  (encoder와 대칭)
      decoder4,5   → PRCM_Bridge
    """
    def __init__(self, num_classes=1, input_channels=3,
                 c_list=[24, 48, 64, 96, 128, 192],
                 kernel_size=7, num_basis=8,
                 dropout_rate=0.5, bridge_dropout=0.2,
                 exp_stages=3, expansion=4, gt_ds=True):
        super().__init__()
        self.gt_ds = gt_ds
        n = len(c_list)  # 6

        # 뒤에서 exp_stages개 스테이지에 use_exp=True
        enc_use_exp = [i >= (n - exp_stages) for i in range(n)]
        # decoder는 encoder와 대칭 (decoder1 ↔ encoder6, ...)
        dec_use_exp = list(reversed(enc_use_exp))

        # ── Encoder ──────────────────────────────
        ch = [input_channels] + c_list
        self.encoders = nn.ModuleList([
            ShuffleDWBlock(ch[i], ch[i+1], kernel_size, num_basis, dropout_rate,
                           use_exp=enc_use_exp[i], expansion=expansion)
            for i in range(n)
        ])

        # ── Deep Supervision ─────────────────────
        if gt_ds:
            self.gt_convs = nn.ModuleList([
                nn.Conv2d(c_list[n-2-i], num_classes, 1) for i in range(n-1)
            ])

        # ── Decoder ──────────────────────────────
        # ctx 크기 = encoder 출력의 half = c_list[i] // 2
        self.decoders = nn.ModuleList([
            ShuffleDWBlock_Bridge(
                c_list[n-1-i], c_list[n-2-i], c_list[n-2-i] // 2,
                kernel_size, num_basis, dropout_rate, bridge_dropout,
                use_exp=dec_use_exp[i], expansion=expansion
            )
            for i in range(n-1)
        ])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)
        self.apply(_init_weights)

    def forward(self, x):
        is_eval = not self.training

        # ── Encoder ──────────────────────────────
        features, ctxs = [], []
        h = x
        for enc in self.encoders[:-1]:
            h, ctx = enc(h, return_ctx=True)
            features.append(h)
            ctxs.append(ctx)
        h = self.encoders[-1](h)  # bottleneck

        # ── Decoder ──────────────────────────────
        ds = []
        for i, dec in enumerate(self.decoders):
            skip = features[-(i+1)]
            ctx  = ctxs[-(i+1)]
            h = F.interpolate(dec(h, ctx_enc=ctx),
                              scale_factor=2, mode='bilinear', align_corners=True) + skip
            ds.append(h)

        out = F.interpolate(self.final(h),
                            scale_factor=2, mode='bilinear', align_corners=True)

        # ── Deep Supervision ─────────────────────
        if self.gt_ds and not is_eval:
            hw = (x.shape[2], x.shape[3])
            gt_outs = tuple(
                F.interpolate(self.gt_convs[i](ds[i]), hw, mode='bilinear', align_corners=True)
                for i in range(len(ds))
            )
            return gt_outs, out
        return out
