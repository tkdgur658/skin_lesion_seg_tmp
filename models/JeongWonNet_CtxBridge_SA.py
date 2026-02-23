"""
JeongWonNet_CtxBridge with Simple Spatial Attention

PRCM (채널 attention) 뒤에 간단한 Spatial Attention 추가
- PRCM: 채널별 중요도 조절
- SA: 공간별 중요도 조절 (7x7 conv → sigmoid)
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


class PRCM_SA(nn.Module):
    """PRCM + Simple Spatial Attention"""
    def __init__(self, channels, num_basis=8, dropout_rate=0.5, sa_kernel=7):
        super().__init__()
        self.num_basis = num_basis
        self.channels = channels

        # PRCM (Channel Attention)
        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        self.fuser = nn.Linear(num_basis, channels, bias=False)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Simple Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=sa_kernel, padding=sa_kernel // 2),
            nn.Sigmoid()
        )

    def forward(self, x, return_ctx=False):
        B, C, H, W = x.shape

        # PRCM (Channel)
        ctx = x.mean(dim=[2, 3])
        coeff = ctx @ self.basis.t()
        coeff = self.coeff_dropout(coeff)
        w_ch = self.fuser(coeff).sigmoid().unsqueeze(-1).unsqueeze(-1)
        x = x * w_ch

        # Spatial Attention
        w_sp = self.spatial_attention(x)
        x = x * w_sp

        if return_ctx:
            return x, ctx
        return x


class PRCM_SA_Bridge(nn.Module):
    """PRCM + Spatial Attention with Encoder Context Bridge"""
    def __init__(self, channels, enc_channels, num_basis=8, dropout_rate=0.5, bridge_dropout=0.2, sa_kernel=7):
        super().__init__()
        self.num_basis = num_basis
        self.channels = channels

        # PRCM
        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        self.fuser = nn.Linear(num_basis, channels, bias=False)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Bridge
        if enc_channels != channels:
            self.ctx_proj = nn.Linear(enc_channels, channels, bias=False)
        else:
            self.ctx_proj = nn.Identity()
        self.bridge_dropout = nn.Dropout(bridge_dropout) if bridge_dropout > 0 else nn.Identity()

        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=sa_kernel, padding=sa_kernel // 2),
            nn.Sigmoid()
        )

    def forward(self, x, ctx_enc=None):
        B, C, H, W = x.shape

        # PRCM with Bridge
        ctx_self = x.mean(dim=[2, 3])

        if ctx_enc is not None:
            ctx_enc_proj = self.ctx_proj(ctx_enc)
            ctx_enc_proj = self.bridge_dropout(ctx_enc_proj)
            ctx_fused = ctx_self + ctx_enc_proj
        else:
            ctx_fused = ctx_self

        coeff = ctx_fused @ self.basis.t()
        coeff = self.coeff_dropout(coeff)
        w_ch = self.fuser(coeff).sigmoid().unsqueeze(-1).unsqueeze(-1)
        x = x * w_ch

        # Spatial Attention
        w_sp = self.spatial_attention(x)
        x = x * w_sp

        return x


class DWBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, num_basis=8, dropout_rate=0.5, sa_kernel=7):
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

        self.prcm_sa = PRCM_SA(out_channels, num_basis=num_basis, dropout_rate=dropout_rate, sa_kernel=sa_kernel)

    def forward(self, x, return_ctx=False):
        if self.pw_conv is not None:
            x = self.pw_conv(x)

        x = self.dw_conv(x)

        if return_ctx:
            return self.prcm_sa(x, return_ctx=True)
        return self.prcm_sa(x)

    def switch_to_deploy(self):
        self.dw_conv.switch_to_deploy()


class DWBlock_Bridge(nn.Module):
    def __init__(self, in_channels, out_channels, enc_channels, kernel_size=7, num_basis=8, dropout_rate=0.5, bridge_dropout=0.2, sa_kernel=7):
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

        self.prcm_sa = PRCM_SA_Bridge(out_channels, enc_channels, num_basis=num_basis, dropout_rate=dropout_rate, bridge_dropout=bridge_dropout, sa_kernel=sa_kernel)

    def forward(self, x, ctx_enc=None):
        if self.pw_conv is not None:
            x = self.pw_conv(x)

        x = self.dw_conv(x)
        return self.prcm_sa(x, ctx_enc)

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


class JeongWonNet_CtxBridge_SA(nn.Module):
    """
    Context Bridge with PRCM + Simple Spatial Attention

    PRCM: 채널별 중요도 (channel attention)
    SA: 공간별 중요도 (spatial attention, 7x7 conv)
    """
    def __init__(self,
                 num_classes=1,
                 input_channels=3,
                 c_list=[24, 48, 64, 96, 128, 192],
                 kernel_size=7,
                 num_basis=8,
                 dropout_rate=0.5,
                 bridge_dropout=0.2,
                 sa_kernel=7,
                 gt_ds=True):
        super().__init__()
        self.gt_ds = gt_ds

        # Encoder
        self.encoder1 = DWBlock(input_channels, c_list[0], kernel_size, num_basis, dropout_rate, sa_kernel)
        self.encoder2 = DWBlock(c_list[0], c_list[1], kernel_size, num_basis, dropout_rate, sa_kernel)
        self.encoder3 = DWBlock(c_list[1], c_list[2], kernel_size, num_basis, dropout_rate, sa_kernel)
        self.encoder4 = DWBlock(c_list[2], c_list[3], kernel_size, num_basis, dropout_rate, sa_kernel)
        self.encoder5 = DWBlock(c_list[3], c_list[4], kernel_size, num_basis, dropout_rate, sa_kernel)
        self.encoder6 = DWBlock(c_list[4], c_list[5], kernel_size, num_basis, dropout_rate, sa_kernel)

        # Deep Supervision
        if gt_ds:
            self.gt_conv1 = nn.Conv2d(c_list[4], num_classes, 1)
            self.gt_conv2 = nn.Conv2d(c_list[3], num_classes, 1)
            self.gt_conv3 = nn.Conv2d(c_list[2], num_classes, 1)
            self.gt_conv4 = nn.Conv2d(c_list[1], num_classes, 1)
            self.gt_conv5 = nn.Conv2d(c_list[0], num_classes, 1)

        # Decoder with Bridge
        self.decoder1 = DWBlock_Bridge(c_list[5], c_list[4], c_list[4], kernel_size, num_basis, dropout_rate, bridge_dropout, sa_kernel)
        self.decoder2 = DWBlock_Bridge(c_list[4], c_list[3], c_list[3], kernel_size, num_basis, dropout_rate, bridge_dropout, sa_kernel)
        self.decoder3 = DWBlock_Bridge(c_list[3], c_list[2], c_list[2], kernel_size, num_basis, dropout_rate, bridge_dropout, sa_kernel)
        self.decoder4 = DWBlock_Bridge(c_list[2], c_list[1], c_list[1], kernel_size, num_basis, dropout_rate, bridge_dropout, sa_kernel)
        self.decoder5 = DWBlock_Bridge(c_list[1], c_list[0], c_list[0], kernel_size, num_basis, dropout_rate, bridge_dropout, sa_kernel)

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)
        self.apply(_init_weights)

    def forward(self, x):
        is_eval = not self.training

        # Encoder with context extraction
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

        # Decoder with context bridge
        d5 = self.decoder1(e6, ctx_enc=ctx5) + e5
        d4 = F.interpolate(self.decoder2(d5, ctx_enc=ctx4), scale_factor=2, mode='bilinear', align_corners=True) + e4
        d3 = F.interpolate(self.decoder3(d4, ctx_enc=ctx3), scale_factor=2, mode='bilinear', align_corners=True) + e3
        d2 = F.interpolate(self.decoder4(d3, ctx_enc=ctx2), scale_factor=2, mode='bilinear', align_corners=True) + e2
        d1 = F.interpolate(self.decoder5(d2, ctx_enc=ctx1), scale_factor=2, mode='bilinear', align_corners=True) + e1

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
            if isinstance(m, (DWBlock, DWBlock_Bridge)):
                m.switch_to_deploy()


if __name__ == "__main__":
    model = JeongWonNet_CtxBridge_SA()
    x = torch.randn(2, 3, 256, 256)

    model.eval()
    with torch.no_grad():
        out = model(x)

    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
