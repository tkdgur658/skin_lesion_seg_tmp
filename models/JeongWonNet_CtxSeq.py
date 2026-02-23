"""
JeongWonNet with Sequential Context Flow

Context Flow 방식:
    이전 스테이지의 context를 다음 스테이지에 순차적으로 전달

    Encoder: ctx1 → enc2, ctx2 → enc3, ...
    Decoder: ctx_d1 → dec2, ctx_d2 → dec3, ...

특징:
    - 순차적 정보 흐름
    - 이전 스테이지의 global 정보를 다음에서 활용
    - Recurrent-like 정보 전파
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


class PRCM_Seq(nn.Module):
    """
    PRCM with Sequential Context Flow

    ctx_fused = ctx_self + proj(ctx_prev)
    Returns: (output, ctx_self) for next stage
    """
    def __init__(self, channels, prev_channels=None, num_basis=8, dropout_rate=0.5):
        super().__init__()
        self.num_basis = num_basis
        self.channels = channels

        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        self.fuser = nn.Linear(num_basis, channels, bias=False)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Previous context projection
        if prev_channels is not None and prev_channels != channels:
            self.ctx_proj = nn.Linear(prev_channels, channels, bias=False)
        else:
            self.ctx_proj = nn.Identity() if prev_channels is not None else None

    def forward(self, x, ctx_prev=None):
        B, C, H, W = x.shape

        ctx_self = x.mean(dim=[2, 3])

        # Fuse with previous context
        if ctx_prev is not None and self.ctx_proj is not None:
            ctx_prev_proj = self.ctx_proj(ctx_prev)
            ctx_fused = ctx_self + ctx_prev_proj
        else:
            ctx_fused = ctx_self

        coeff = ctx_fused @ self.basis.t()
        coeff = self.coeff_dropout(coeff)

        w = self.fuser(coeff).sigmoid().unsqueeze(-1).unsqueeze(-1)
        return x * w, ctx_self


class DWBlock_Seq(nn.Module):
    """DWBlock with Sequential Context"""
    def __init__(self, in_channels, out_channels, prev_channels=None,
                 kernel_size=7, num_basis=8, dropout_rate=0.5):
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

        self.prcm = PRCM_Seq(out_channels, prev_channels, num_basis=num_basis, dropout_rate=dropout_rate)

    def forward(self, x, ctx_prev=None):
        if self.pw_conv is not None:
            x = self.pw_conv(x)

        x = self.dw_conv(x)
        return self.prcm(x, ctx_prev)

    def switch_to_deploy(self):
        self.dw_conv.switch_to_deploy()


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


class JeongWonNet_CtxSeq(nn.Module):
    """
    Sequential Context Flow

    Encoder: None → enc1 → ctx1 → enc2 → ctx2 → ...
    Decoder: None → dec1 → ctx_d1 → dec2 → ctx_d2 → ...
    """
    def __init__(self,
                 num_classes=1,
                 input_channels=3,
                 c_list=[24, 48, 64, 96, 128, 192],
                 kernel_size=7,
                 num_basis=8,
                 dropout_rate=0.5,
                 gt_ds=True):
        super().__init__()
        self.gt_ds = gt_ds

        # Encoder (sequential context flow)
        self.encoder1 = DWBlock_Seq(input_channels, c_list[0], None, kernel_size, num_basis, dropout_rate)
        self.encoder2 = DWBlock_Seq(c_list[0], c_list[1], c_list[0], kernel_size, num_basis, dropout_rate)
        self.encoder3 = DWBlock_Seq(c_list[1], c_list[2], c_list[1], kernel_size, num_basis, dropout_rate)
        self.encoder4 = DWBlock_Seq(c_list[2], c_list[3], c_list[2], kernel_size, num_basis, dropout_rate)
        self.encoder5 = DWBlock_Seq(c_list[3], c_list[4], c_list[3], kernel_size, num_basis, dropout_rate)
        self.encoder6 = DWBlock_Seq(c_list[4], c_list[5], c_list[4], kernel_size, num_basis, dropout_rate)

        # Deep Supervision
        if gt_ds:
            self.gt_conv1 = nn.Conv2d(c_list[4], num_classes, 1)
            self.gt_conv2 = nn.Conv2d(c_list[3], num_classes, 1)
            self.gt_conv3 = nn.Conv2d(c_list[2], num_classes, 1)
            self.gt_conv4 = nn.Conv2d(c_list[1], num_classes, 1)
            self.gt_conv5 = nn.Conv2d(c_list[0], num_classes, 1)

        # Decoder (sequential context flow)
        self.decoder1 = DWBlock_Seq(c_list[5], c_list[4], None, kernel_size, num_basis, dropout_rate)
        self.decoder2 = DWBlock_Seq(c_list[4], c_list[3], c_list[4], kernel_size, num_basis, dropout_rate)
        self.decoder3 = DWBlock_Seq(c_list[3], c_list[2], c_list[3], kernel_size, num_basis, dropout_rate)
        self.decoder4 = DWBlock_Seq(c_list[2], c_list[1], c_list[2], kernel_size, num_basis, dropout_rate)
        self.decoder5 = DWBlock_Seq(c_list[1], c_list[0], c_list[1], kernel_size, num_basis, dropout_rate)

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)
        self.apply(_init_weights)

    def forward(self, x):
        is_eval = not self.training

        # Encoder with sequential context
        e1_out, ctx1 = self.encoder1(x, ctx_prev=None)
        e1 = F.max_pool2d(e1_out, 2)

        e2_out, ctx2 = self.encoder2(e1, ctx_prev=ctx1)
        e2 = F.max_pool2d(e2_out, 2)

        e3_out, ctx3 = self.encoder3(e2, ctx_prev=ctx2)
        e3 = F.max_pool2d(e3_out, 2)

        e4_out, ctx4 = self.encoder4(e3, ctx_prev=ctx3)
        e4 = F.max_pool2d(e4_out, 2)

        e5_out, ctx5 = self.encoder5(e4, ctx_prev=ctx4)
        e5 = F.max_pool2d(e5_out, 2)

        e6_out, ctx6 = self.encoder6(e5, ctx_prev=ctx5)

        # Decoder with sequential context
        d5_out, ctx_d5 = self.decoder1(e6_out, ctx_prev=None)
        d5 = d5_out + e5

        d4_out, ctx_d4 = self.decoder2(d5, ctx_prev=ctx_d5)
        d4 = F.interpolate(d4_out, scale_factor=2, mode='bilinear', align_corners=True) + e4

        d3_out, ctx_d3 = self.decoder3(d4, ctx_prev=ctx_d4)
        d3 = F.interpolate(d3_out, scale_factor=2, mode='bilinear', align_corners=True) + e3

        d2_out, ctx_d2 = self.decoder4(d3, ctx_prev=ctx_d3)
        d2 = F.interpolate(d2_out, scale_factor=2, mode='bilinear', align_corners=True) + e2

        d1_out, ctx_d1 = self.decoder5(d2, ctx_prev=ctx_d2)
        d1 = F.interpolate(d1_out, scale_factor=2, mode='bilinear', align_corners=True) + e1

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

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, DWBlock_Seq):
                m.switch_to_deploy()


if __name__ == "__main__":
    print("=" * 60)
    print("JeongWonNet_CtxSeq Test")
    print("Sequential Context Flow")
    print("=" * 60)

    model = JeongWonNet_CtxSeq(
        num_classes=1,
        input_channels=3,
        c_list=[24, 48, 64, 96, 128, 192],
        kernel_size=7,
        num_basis=8,
        dropout_rate=0.5,
        gt_ds=True
    )

    x = torch.randn(2, 3, 256, 256)

    model.train()
    ds_outputs, final_out = model(x)

    print(f"\nTraining Mode:")
    print(f"  Final Output: {final_out.shape}")
    print(f"  Deep Supervision: {len(ds_outputs)} levels")

    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"\nEvaluation Mode: {out.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params:,}")

    print("\n[Sequential Context Flow 구조]")
    print("  Encoder: None → enc1 → ctx1 → enc2 → ctx2 → ...")
    print("  Decoder: None → dec1 → ctx_d1 → dec2 → ctx_d2 → ...")
    print("  ctx_fused = ctx_self + proj(ctx_prev)")
