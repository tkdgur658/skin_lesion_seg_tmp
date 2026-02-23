"""
JeongWonNet_CtxBridge_Coeff

기존 CtxBridge: GAP 출력 (ctx, 채널 수 만큼) 전달
변경: Basis 투영된 coeff (num_basis 차원) 전달

장점:
    - 차원이 훨씬 작음 (128ch → 8 dims)
    - 모든 encoder coeff 합쳐도 작음 (8 × 5 = 40 dims)
    - 이미 의미있는 특징으로 압축된 상태
    - 모든 encoder 정보를 한번에 전달 가능

구조:
    Encoder: ctx → coeff = ctx @ basis.T  (num_basis dims)
    All coeffs: [coeff1, coeff2, ..., coeff5] → concat (num_basis × 5 dims)
    Decoder: 모든 coeff를 받아서 활용
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


class PRCM_Coeff(nn.Module):
    """
    PRCM that returns basis-projected coefficients

    기존: ctx (channels dims) 반환
    변경: coeff (num_basis dims) 반환 - 훨씬 compact
    """
    def __init__(self, channels, num_basis=8, dropout_rate=0.5):
        super().__init__()
        self.num_basis = num_basis
        self.channels = channels

        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        self.fuser = nn.Linear(num_basis, channels, bias=False)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x, return_coeff=False):
        B, C, H, W = x.shape

        ctx = x.mean(dim=[2, 3])
        coeff = ctx @ self.basis.t()  # (B, num_basis)
        coeff_drop = self.coeff_dropout(coeff)

        w = self.fuser(coeff_drop).sigmoid().unsqueeze(-1).unsqueeze(-1)

        if return_coeff:
            return x * w, coeff  # coeff without dropout for bridge
        return x * w


class PRCM_Bridge_Coeff(nn.Module):
    """
    PRCM with All Encoder Coefficients

    모든 encoder의 coeff를 concat하여 활용
    coeff_all = [coeff1, coeff2, ..., coeff5]  (num_basis × 5 dims)

    구조:
        coeff_self = ctx_self @ basis.T
        coeff_bridge = Linear(coeff_all_enc)  → num_basis dims
        coeff_fused = coeff_self + coeff_bridge
        w = sigmoid(fuser(dropout(coeff_fused)))
    """
    def __init__(self, channels, num_basis=8, num_enc_stages=5, dropout_rate=0.5, bridge_dropout=0.2):
        super().__init__()
        self.num_basis = num_basis
        self.channels = channels

        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        self.fuser = nn.Linear(num_basis, channels, bias=False)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # All encoder coefficients → num_basis projection
        # Input: num_basis × num_enc_stages, Output: num_basis
        self.coeff_bridge = nn.Linear(num_basis * num_enc_stages, num_basis, bias=False)
        self.bridge_dropout = nn.Dropout(bridge_dropout) if bridge_dropout > 0 else nn.Identity()

    def forward(self, x, coeff_all_enc=None):
        B, C, H, W = x.shape

        ctx_self = x.mean(dim=[2, 3])
        coeff_self = ctx_self @ self.basis.t()  # (B, num_basis)

        if coeff_all_enc is not None:
            # coeff_all_enc: (B, num_basis × num_enc_stages)
            coeff_bridge = self.coeff_bridge(coeff_all_enc)  # (B, num_basis)
            coeff_bridge = self.bridge_dropout(coeff_bridge)  # bridge에만 dropout
            coeff_fused = coeff_self + coeff_bridge
        else:
            coeff_fused = self.coeff_dropout(coeff_self)  # bridge 없을 때만 dropout
        w = self.fuser(coeff_fused).sigmoid().unsqueeze(-1).unsqueeze(-1)
        return x * w


class DWBlock_Coeff(nn.Module):
    """Encoder DWBlock - returns coeff"""
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

        self.prcm = PRCM_Coeff(out_channels, num_basis=num_basis, dropout_rate=dropout_rate)

    def forward(self, x, return_coeff=False):
        if self.pw_conv is not None:
            x = self.pw_conv(x)

        x = self.dw_conv(x)

        if return_coeff:
            return self.prcm(x, return_coeff=True)
        return self.prcm(x)

    def switch_to_deploy(self):
        self.dw_conv.switch_to_deploy()


class DWBlock_Bridge_Coeff(nn.Module):
    """Decoder DWBlock - receives all encoder coeffs"""
    def __init__(self, in_channels, out_channels, kernel_size=7, num_basis=8,
                 num_enc_stages=5, dropout_rate=0.5, bridge_dropout=0.2):
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

        self.prcm = PRCM_Bridge_Coeff(
            out_channels, num_basis=num_basis, num_enc_stages=num_enc_stages,
            dropout_rate=dropout_rate, bridge_dropout=bridge_dropout
        )

    def forward(self, x, coeff_all_enc=None):
        if self.pw_conv is not None:
            x = self.pw_conv(x)

        x = self.dw_conv(x)
        return self.prcm(x, coeff_all_enc)

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


class JeongWonNet_CtxBridge_Coeff(nn.Module):
    """
    Context Bridge with Basis Coefficients

    기존: GAP ctx (24~128 dims) 전달
    변경: Basis coeff (8 dims each) 전달

    All encoder coeffs: [coeff1, ..., coeff5] = 8 × 5 = 40 dims
    → 모든 decoder에서 동일한 coeff_all 사용

    장점:
        - 매우 compact (40 dims vs 24+48+64+96+128=360 dims)
        - 이미 basis에 투영된 의미있는 특징
        - 모든 encoder 정보를 한번에 전달
    """
    def __init__(self,
                 num_classes=1,
                 input_channels=3,
                 c_list=[24, 48, 64, 96, 128, 192],
                 kernel_size=7,
                 num_basis=32,
                 dropout_rate=0.5,
                 bridge_dropout=0.2,
                 gt_ds=True):
        super().__init__()
        self.gt_ds = gt_ds
        self.num_basis = num_basis

        # Encoder (returns coeff)
        self.encoder1 = DWBlock_Coeff(input_channels, c_list[0], kernel_size, num_basis, dropout_rate)
        self.encoder2 = DWBlock_Coeff(c_list[0], c_list[1], kernel_size, num_basis, dropout_rate)
        self.encoder3 = DWBlock_Coeff(c_list[1], c_list[2], kernel_size, num_basis, dropout_rate)
        self.encoder4 = DWBlock_Coeff(c_list[2], c_list[3], kernel_size, num_basis, dropout_rate)
        self.encoder5 = DWBlock_Coeff(c_list[3], c_list[4], kernel_size, num_basis, dropout_rate)
        self.encoder6 = DWBlock_Coeff(c_list[4], c_list[5], kernel_size, num_basis, dropout_rate)

        # Deep Supervision
        if gt_ds:
            self.gt_conv1 = nn.Conv2d(c_list[4], num_classes, 1)
            self.gt_conv2 = nn.Conv2d(c_list[3], num_classes, 1)
            self.gt_conv3 = nn.Conv2d(c_list[2], num_classes, 1)
            self.gt_conv4 = nn.Conv2d(c_list[1], num_classes, 1)
            self.gt_conv5 = nn.Conv2d(c_list[0], num_classes, 1)

        # Decoder (receives all encoder coeffs)
        num_enc_stages = 5  # enc1 ~ enc5의 coeff 사용
        self.decoder1 = DWBlock_Bridge_Coeff(c_list[5], c_list[4], kernel_size, num_basis, num_enc_stages, dropout_rate, bridge_dropout)
        self.decoder2 = DWBlock_Bridge_Coeff(c_list[4], c_list[3], kernel_size, num_basis, num_enc_stages, dropout_rate, bridge_dropout)
        self.decoder3 = DWBlock_Bridge_Coeff(c_list[3], c_list[2], kernel_size, num_basis, num_enc_stages, dropout_rate, bridge_dropout)
        self.decoder4 = DWBlock_Bridge_Coeff(c_list[2], c_list[1], kernel_size, num_basis, num_enc_stages, dropout_rate, bridge_dropout)
        self.decoder5 = DWBlock_Bridge_Coeff(c_list[1], c_list[0], kernel_size, num_basis, num_enc_stages, dropout_rate, bridge_dropout)

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)
        self.apply(_init_weights)

    def forward(self, x):
        is_eval = not self.training

        # Encoder with coeff extraction
        e1_out, coeff1 = self.encoder1(x, return_coeff=True)
        e1 = F.max_pool2d(e1_out, 2)

        e2_out, coeff2 = self.encoder2(e1, return_coeff=True)
        e2 = F.max_pool2d(e2_out, 2)

        e3_out, coeff3 = self.encoder3(e2, return_coeff=True)
        e3 = F.max_pool2d(e3_out, 2)

        e4_out, coeff4 = self.encoder4(e3, return_coeff=True)
        e4 = F.max_pool2d(e4_out, 2)

        e5_out, coeff5 = self.encoder5(e4, return_coeff=True)
        e5 = F.max_pool2d(e5_out, 2)

        e6 = self.encoder6(e5)

        # Concatenate all encoder coefficients
        # coeff_all: (B, num_basis × 5) = (B, 40)
        coeff_all = torch.cat([coeff1, coeff2, coeff3, coeff4, coeff5], dim=1)

        # Decoder with all encoder coeffs
        d5 = self.decoder1(e6, coeff_all_enc=coeff_all) + e5
        d4 = F.interpolate(self.decoder2(d5, coeff_all_enc=coeff_all), scale_factor=2, mode='bilinear', align_corners=True) + e4
        d3 = F.interpolate(self.decoder3(d4, coeff_all_enc=coeff_all), scale_factor=2, mode='bilinear', align_corners=True) + e3
        d2 = F.interpolate(self.decoder4(d3, coeff_all_enc=coeff_all), scale_factor=2, mode='bilinear', align_corners=True) + e2
        d1 = F.interpolate(self.decoder5(d2, coeff_all_enc=coeff_all), scale_factor=2, mode='bilinear', align_corners=True) + e1

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
            if isinstance(m, (DWBlock_Coeff, DWBlock_Bridge_Coeff)):
                m.switch_to_deploy()


if __name__ == "__main__":
    print("=" * 60)
    print("JeongWonNet_CtxBridge_Coeff Test")
    print("Basis Coefficient Bridge (compact: 40 dims)")
    print("=" * 60)

    model = JeongWonNet_CtxBridge_Coeff(
        num_classes=1,
        input_channels=3,
        c_list=[24, 48, 64, 96, 128, 192],
        kernel_size=7,
        num_basis=8,
        dropout_rate=0.5,
        bridge_dropout=0.2,
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

    # Bridge parameters
    bridge_params = sum(
        sum(p.numel() for p in m.prcm.coeff_bridge.parameters())
        for m in [model.decoder1, model.decoder2, model.decoder3, model.decoder4, model.decoder5]
    )
    print(f"Coeff Bridge Parameters: {bridge_params:,}")

    print("\n[Coeff Bridge 구조]")
    print("  Encoder: coeff = ctx @ basis.T  (8 dims each)")
    print("  All coeffs: [c1, c2, c3, c4, c5] = 40 dims")
    print("  Decoder: coeff_bridge = Linear(40 → 8)")
    print("  coeff_fused = coeff_self + dropout(coeff_bridge)")
    print("\n[비교]")
    print("  기존 CtxBridge: ctx 전달 (24+48+64+96+128 = 360 dims)")
    print("  Coeff Bridge:   coeff 전달 (8×5 = 40 dims) ← 9배 compact!")
