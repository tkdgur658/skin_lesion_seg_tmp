import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_


class RepConv(nn.Module):
    """
    Re-parameterizable Convolution Block
    훈련: Conv + BN + Identity(or 1x1) branch
    추론: 단일 Conv로 융합
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

        self.use_identity = use_identity and (stride == 1) and (in_channels == out_channels)

        # 주 Branch: kernel_size Conv + BN
        self.conv_kxk = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride, padding, groups=groups, bias=False)
        self.bn_kxk = nn.BatchNorm2d(out_channels)

        # 1x1 Branch
        if kernel_size > 1:
            self.conv_1x1 = nn.Conv2d(in_channels, out_channels, 1,
                                      stride, 0, groups=groups, bias=False)
            self.bn_1x1 = nn.BatchNorm2d(out_channels)
        else:
            self.conv_1x1 = None

        # Identity Branch
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
        """추론 모드로 전환: 모든 branch를 단일 Conv로 융합"""
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

        # 훈련용 레이어 제거
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


class PRCM(nn.Module):
    """
    Pattern Recalibration Module
    Low-rank basis를 사용한 채널 재조정
    """
    def __init__(self, channels, num_basis=8, dropout_rate=0.5):
        super().__init__()
        self.num_basis = num_basis
        self.channels = channels

        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        self.fuser = nn.Linear(num_basis, channels, bias=False)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        ctx = x.mean(dim=[2, 3])
        coeff = ctx @ self.basis.t()
        coeff = self.coeff_dropout(coeff)

        w = self.fuser(coeff).sigmoid().unsqueeze(-1).unsqueeze(-1)
        return x * w


class DWBlock(nn.Module):
    """
    Depthwise Block (JeongWonNet77_Rep256Basis8S24Drop style)

    Structure:
        RepConv 7x7 DW (in_ch) -> [1x1 Conv] (if in_ch != out_ch) -> PRCM (out_ch)

    채널 확장 시 RepConv를 작은 채널에서 수행하여 파라미터 절약

    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        kernel_size: Depthwise Conv 커널 크기 (default: 7)
        num_basis: PRCM basis 개수 (default: 8)
        dropout_rate: PRCM dropout rate (default: 0.5)
    """
    def __init__(self, in_channels, out_channels, kernel_size=7, num_basis=8, dropout_rate=0.5):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Depthwise RepConv (입력 채널에서 수행 - 파라미터 절약)
        self.dw_conv = RepConv(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels
        )

        # 채널 조정이 필요할 때만 1x1 pointwise conv (RepConv 이후)
        if in_channels != out_channels:
            self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.pw_conv = None

        # Pattern Recalibration Module
        self.prcm = PRCM(out_channels, num_basis=num_basis, dropout_rate=dropout_rate)

    def forward(self, x):
        x = self.dw_conv(x)

        if self.pw_conv is not None:
            x = self.pw_conv(x)

        x = self.prcm(x)

        return x

    def switch_to_deploy(self):
        """추론 모드로 전환"""
        self.dw_conv.switch_to_deploy()


def _init_weights(m):
    """Weight initialization"""
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


if __name__ == "__main__":
    # Test DWBlock
    print("=" * 60)
    print("DWBlock Test")
    print("=" * 60)

    # Same channel
    block = DWBlock(64, 64, kernel_size=7, num_basis=8)
    block.apply(_init_weights)
    x = torch.randn(2, 64, 32, 32)

    block.train()
    out = block(x)
    print(f"Same channel (64->64): {x.shape} -> {out.shape}")

    # Channel expansion
    block = DWBlock(64, 128, kernel_size=7, num_basis=8)
    block.apply(_init_weights)
    x = torch.randn(2, 64, 32, 32)

    out = block(x)
    print(f"Channel expansion (64->128): {x.shape} -> {out.shape}")

    # Deploy mode test
    block = DWBlock(64, 64, kernel_size=7, num_basis=8)
    block.apply(_init_weights)
    block.eval()

    x = torch.randn(1, 64, 32, 32)

    with torch.no_grad():
        out_train = block(x)

    block.switch_to_deploy()

    with torch.no_grad():
        out_deploy = block(x)

    diff = (out_train - out_deploy).abs().max().item()
    print(f"Deploy mode diff: {diff:.6f}")

    # Parameter count
    params = sum(p.numel() for p in block.parameters())
    print(f"Parameters (64ch): {params:,}")

    print("\n[Structure]")
    print(block)
