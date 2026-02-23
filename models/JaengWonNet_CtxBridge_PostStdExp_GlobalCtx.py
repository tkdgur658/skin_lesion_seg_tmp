import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_


class RepConv(nn.Module):
    # 그대로 재사용 (위에서 쓰던 버전과 동일)
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
        if hasattr(self, "fused_conv"):
            return self.activation(self.fused_conv(x))
        out = self.bn_kxk(self.conv_kxk(x))
        if self.conv_1x1 is not None:
            out = out + self.bn_1x1(self.conv_1x1(x))
        if self.use_identity:
            out = out + self.bn_identity(x)
        return self.activation(out)

    # switch_to_deploy 그대로...


class PRCM(nn.Module):
    def __init__(self, channels, num_basis=8, dropout_rate=0.5):
        super().__init__()
        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        self.fuser = nn.Linear(num_basis, channels, bias=False)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x, return_ctx=False):
        ctx = x.mean(dim=[2, 3])  # (B, C)
        coeff = self.coeff_dropout(ctx @ self.basis.t())
        w = self.fuser(coeff).sigmoid().unsqueeze(-1).unsqueeze(-1)
        if return_ctx:
            return x * w, ctx
        return x * w


class PRCM_Bridge_Global(nn.Module):
    """
    Decoder에서 encoder multi-scale GAP context(global_ctx)를 사용하는 bridge.
    x: (B, C, H, W), global_ctx: (B, enc_ctx_dim)
    """
    def __init__(self, channels, enc_ctx_dim, num_basis=8,
                 dropout_rate=0.5, bridge_dropout=0.2):
        super().__init__()
        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        self.fuser = nn.Linear(num_basis, channels, bias=False)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.ctx_proj = nn.Linear(enc_ctx_dim, channels, bias=False)
        self.bridge_dropout = nn.Dropout(bridge_dropout) if bridge_dropout > 0 else nn.Identity()

    def forward(self, x, global_ctx=None):
        ctx_self = x.mean(dim=[2, 3])
        if global_ctx is not None:
            g = self.bridge_dropout(self.ctx_proj(global_ctx))
            ctx_fused = ctx_self + g
        else:
            ctx_fused = ctx_self
        coeff = self.coeff_dropout(ctx_fused @ self.basis.t())
        w = self.fuser(coeff).sigmoid().unsqueeze(-1).unsqueeze(-1)
        return x * w

class DWBlock_PostStdExp(nn.Module):
    """
    pw(1x1) → dw(7x7) → std(3x3) → [expand(1x1,x4)→project(1x1)] → PRCM
    shallow/deep 공통, 채널 수/exp 비율만 다르게.
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=7, num_basis=8,
                 dropout_rate=0.5, expansion=4,
                 use_exp=True):
        super().__init__()

        if in_channels != out_channels:
            self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.pw_conv = None

        self.dw_conv = RepConv(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=out_channels
        )

        self.std_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.use_exp = use_exp
        if use_exp:
            hidden_dim = out_channels * expansion
            self.exp_proj = nn.Sequential(
                nn.Conv2d(out_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.exp_proj = nn.Identity()

        self.prcm = PRCM(out_channels, num_basis=num_basis, dropout_rate=dropout_rate)

    def forward(self, x, return_ctx=False):
        if self.pw_conv is not None:
            x = self.pw_conv(x)
        x = self.dw_conv(x)
        x = self.std_conv(x)
        x = self.exp_proj(x)
        if return_ctx:
            return self.prcm(x, return_ctx=True)
        return self.prcm(x)

    def switch_to_deploy(self):
        self.dw_conv.switch_to_deploy()
class DWBlock_PostStdExp_Bridge(nn.Module):
    """Decoder용 PostStdExp + GlobalCtx"""
    def __init__(self, in_channels, out_channels, enc_ctx_dim,
                 kernel_size=7, num_basis=8,
                 dropout_rate=0.5, bridge_dropout=0.2,
                 expansion=4, use_exp=True):
        super().__init__()

        if in_channels != out_channels:
            self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.pw_conv = None

        self.dw_conv = RepConv(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=out_channels
        )

        self.std_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.use_exp = use_exp
        if use_exp:
            hidden_dim = out_channels * expansion
            self.exp_proj = nn.Sequential(
                nn.Conv2d(out_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.exp_proj = nn.Identity()

        self.prcm = PRCM_Bridge_Global(
            out_channels, enc_ctx_dim,
            num_basis=num_basis,
            dropout_rate=dropout_rate,
            bridge_dropout=bridge_dropout
        )

    def forward(self, x, global_ctx=None):
        if self.pw_conv is not None:
            x = self.pw_conv(x)
        x = self.dw_conv(x)
        x = self.std_conv(x)
        x = self.exp_proj(x)
        return self.prcm(x, global_ctx)

    def switch_to_deploy(self):
        self.dw_conv.switch_to_deploy()
        
class JaengWonNet_CtxBridge_PostStdExp_GlobalCtx(nn.Module):
    """
    PostStdExp + GlobalCtx 버전.
    예: shallow(enc1-4, dec3-5) use_exp=True, deep(enc5-6, dec1-2) use_exp=False.
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

        # Encoder: shallow(1-4) with exp, deep(5-6) without exp
        self.encoder1 = DWBlock_PostStdExp(input_channels, c_list[0],
                                           kernel_size, num_basis,
                                           dropout_rate, expansion,
                                           use_exp=True)
        self.encoder2 = DWBlock_PostStdExp(c_list[0], c_list[1],
                                           kernel_size, num_basis,
                                           dropout_rate, expansion,
                                           use_exp=True)
        self.encoder3 = DWBlock_PostStdExp(c_list[1], c_list[2],
                                           kernel_size, num_basis,
                                           dropout_rate, expansion,
                                           use_exp=True)
        self.encoder4 = DWBlock_PostStdExp(c_list[2], c_list[3],
                                           kernel_size, num_basis,
                                           dropout_rate, expansion,
                                           use_exp=True)
        self.encoder5 = DWBlock_PostStdExp(c_list[3], c_list[4],
                                           kernel_size, num_basis,
                                           dropout_rate, expansion,
                                           use_exp=False)
        self.encoder6 = DWBlock_PostStdExp(c_list[4], c_list[5],
                                           kernel_size, num_basis,
                                           dropout_rate, expansion,
                                           use_exp=False)

        # encoder1~5 ctx dim = 각 stage out_channels
        self.total_ctx_dim = sum(c_list[:5])

        if gt_ds:
            self.gt_conv1 = nn.Conv2d(c_list[4], num_classes, 1)
            self.gt_conv2 = nn.Conv2d(c_list[3], num_classes, 1)
            self.gt_conv3 = nn.Conv2d(c_list[2], num_classes, 1)
            self.gt_conv4 = nn.Conv2d(c_list[1], num_classes, 1)
            self.gt_conv5 = nn.Conv2d(c_list[0], num_classes, 1)

        # Decoder: deep(dec1-2) without exp, shallow(dec3-5) with exp
        self.decoder1 = DWBlock_PostStdExp_Bridge(
            c_list[5], c_list[4], self.total_ctx_dim,
            kernel_size, num_basis,
            dropout_rate, bridge_dropout,
            expansion, use_exp=False
        )
        self.decoder2 = DWBlock_PostStdExp_Bridge(
            c_list[4], c_list[3], self.total_ctx_dim,
            kernel_size, num_basis,
            dropout_rate, bridge_dropout,
            expansion, use_exp=False
        )
        self.decoder3 = DWBlock_PostStdExp_Bridge(
            c_list[3], c_list[2], self.total_ctx_dim,
            kernel_size, num_basis,
            dropout_rate, bridge_dropout,
            expansion, use_exp=True
        )
        self.decoder4 = DWBlock_PostStdExp_Bridge(
            c_list[2], c_list[1], self.total_ctx_dim,
            kernel_size, num_basis,
            dropout_rate, bridge_dropout,
            expansion, use_exp=True
        )
        self.decoder5 = DWBlock_PostStdExp_Bridge(
            c_list[1], c_list[0], self.total_ctx_dim,
            kernel_size, num_basis,
            dropout_rate, bridge_dropout,
            expansion, use_exp=True
        )

        self.final = nn.Conv2d(c_list[0], num_classes, 1)
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

        e6, _ = self.encoder6(e5, return_ctx=True)  # 마지막 ctx는 global_ctx에 안 씀

        # multi-scale encoder GAP → global context
        global_ctx = torch.cat([ctx1, ctx2, ctx3, ctx4, ctx5], dim=1)

        # Decoder
        d5 = self.decoder1(e6, global_ctx=global_ctx) + e5
        d4 = F.interpolate(self.decoder2(d5, global_ctx=global_ctx),
                           scale_factor=2, mode="bilinear", align_corners=True) + e4
        d3 = F.interpolate(self.decoder3(d4, global_ctx=global_ctx),
                           scale_factor=2, mode="bilinear", align_corners=True) + e3
        d2 = F.interpolate(self.decoder4(d3, global_ctx=global_ctx),
                           scale_factor=2, mode="bilinear", align_corners=True) + e2
        d1 = F.interpolate(self.decoder5(d2, global_ctx=global_ctx),
                           scale_factor=2, mode="bilinear", align_corners=True) + e1

        out = F.interpolate(self.final(d1),
                            scale_factor=2, mode="bilinear", align_corners=True)

        if self.gt_ds and not is_eval:
            h, w = x.shape[2], x.shape[3]
            return (
                F.interpolate(self.gt_conv1(d5), (h, w), mode="bilinear", align_corners=True),
                F.interpolate(self.gt_conv2(d4), (h, w), mode="bilinear", align_corners=True),
                F.interpolate(self.gt_conv3(d3), (h, w), mode="bilinear", align_corners=True),
                F.interpolate(self.gt_conv4(d2), (h, w), mode="bilinear", align_corners=True),
                F.interpolate(self.gt_conv5(d1), (h, w), mode="bilinear", align_corners=True)
            ), out
        return out
