"""
JeongWonNet_CtxBridge_Deep

스테이지 수를 줄이고 스테이지당 DWBlock 수를 늘림

기존: 6 stages × 1 block = 6 blocks
변경: 4 stages × 2 blocks = 8 blocks (더 깊음)

구조:
    Stage 1: 256→128, 2 blocks (32ch)
    Stage 2: 128→64,  2 blocks (64ch)
    Stage 3: 64→32,   2 blocks (128ch)
    Stage 4: 32→16,   2 blocks (256ch)
    Bottleneck: 16,   2 blocks (256ch)

장점:
    - 각 해상도에서 더 많은 처리
    - 더 풍부한 특징 학습
    - 총 파라미터는 비슷하게 유지 가능
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


class DWBlock(nn.Module):
    """Single DWBlock"""
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

        self.prcm = PRCM(out_channels, num_basis=num_basis, dropout_rate=dropout_rate)

    def forward(self, x, return_ctx=False):
        if self.pw_conv is not None:
            x = self.pw_conv(x)

        x = self.dw_conv(x)

        if return_ctx:
            return self.prcm(x, return_ctx=True)
        return self.prcm(x)

    def switch_to_deploy(self):
        self.dw_conv.switch_to_deploy()


class DWBlock_Bridge(nn.Module):
    """DWBlock with Context Bridge"""
    def __init__(self, in_channels, out_channels, enc_channels, kernel_size=7,
                 num_basis=8, dropout_rate=0.5, bridge_dropout=0.2):
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

        self.prcm = PRCM_Bridge(
            out_channels, enc_channels, num_basis=num_basis,
            dropout_rate=dropout_rate, bridge_dropout=bridge_dropout
        )

    def forward(self, x, ctx_enc=None):
        if self.pw_conv is not None:
            x = self.pw_conv(x)

        x = self.dw_conv(x)
        return self.prcm(x, ctx_enc)

    def switch_to_deploy(self):
        self.dw_conv.switch_to_deploy()


class EncoderStage(nn.Module):
    """
    Encoder Stage with multiple DWBlocks

    첫 번째 블록: 채널 변환 + ctx 추출
    나머지 블록: 동일 채널 유지
    """
    def __init__(self, in_channels, out_channels, num_blocks=2, kernel_size=7, num_basis=8, dropout_rate=0.5):
        super().__init__()

        blocks = []
        for i in range(num_blocks):
            if i == 0:
                blocks.append(DWBlock(in_channels, out_channels, kernel_size, num_basis, dropout_rate))
            else:
                blocks.append(DWBlock(out_channels, out_channels, kernel_size, num_basis, dropout_rate))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, return_ctx=False):
        for i, block in enumerate(self.blocks):
            if i == len(self.blocks) - 1 and return_ctx:
                # 마지막 블록에서 ctx 추출
                x, ctx = block(x, return_ctx=True)
                return x, ctx
            else:
                x = block(x)

        return x

    def switch_to_deploy(self):
        for block in self.blocks:
            block.switch_to_deploy()


class DecoderStage(nn.Module):
    """
    Decoder Stage with multiple DWBlocks + Context Bridge

    첫 번째 블록: 채널 변환 + context bridge
    나머지 블록: 동일 채널 유지 (bridge 없음)
    """
    def __init__(self, in_channels, out_channels, enc_channels, num_blocks=2,
                 kernel_size=7, num_basis=8, dropout_rate=0.5, bridge_dropout=0.2):
        super().__init__()

        blocks = []
        for i in range(num_blocks):
            if i == 0:
                # 첫 번째 블록: context bridge 포함
                blocks.append(DWBlock_Bridge(in_channels, out_channels, enc_channels,
                                             kernel_size, num_basis, dropout_rate, bridge_dropout))
            else:
                # 나머지 블록: 일반 DWBlock
                blocks.append(DWBlock(out_channels, out_channels, kernel_size, num_basis, dropout_rate))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, ctx_enc=None):
        for i, block in enumerate(self.blocks):
            if i == 0:
                x = block(x, ctx_enc=ctx_enc)
            else:
                x = block(x)
        return x

    def switch_to_deploy(self):
        for block in self.blocks:
            block.switch_to_deploy()


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


class JeongWonNet_CtxBridge_Deep(nn.Module):
    """
    Deep CtxBridge - 5 스테이지, 스테이지당 2 블록

    기존 (6 stages × 1 block):
        256→128→64→32→16→8, 채널: 24→48→64→96→128→192

    변경 (5 stages × 2 blocks):
        256→128→64→32→16→8, 채널: 24→48→96→128→192
        각 해상도에서 2개 블록 처리

    Args:
        num_blocks: 스테이지당 블록 수 (default: 2)
    """
    def __init__(self,
                 num_classes=1,
                 input_channels=3,
                 c_list=[24, 48, 96, 128, 192],
                 kernel_size=7,
                 num_basis=8,
                 num_blocks=2,
                 dropout_rate=0.5,
                 bridge_dropout=0.2,
                 gt_ds=True):
        super().__init__()
        self.gt_ds = gt_ds
        self.num_stages = len(c_list)

        # Encoder stages (5 stages)
        self.encoder1 = EncoderStage(input_channels, c_list[0], num_blocks, kernel_size, num_basis, dropout_rate)
        self.encoder2 = EncoderStage(c_list[0], c_list[1], num_blocks, kernel_size, num_basis, dropout_rate)
        self.encoder3 = EncoderStage(c_list[1], c_list[2], num_blocks, kernel_size, num_basis, dropout_rate)
        self.encoder4 = EncoderStage(c_list[2], c_list[3], num_blocks, kernel_size, num_basis, dropout_rate)
        self.encoder5 = EncoderStage(c_list[3], c_list[4], num_blocks, kernel_size, num_basis, dropout_rate)

        # Bottleneck
        self.bottleneck = EncoderStage(c_list[4], c_list[4], num_blocks, kernel_size, num_basis, dropout_rate)

        # Deep Supervision (5 outputs)
        if gt_ds:
            self.gt_conv1 = nn.Conv2d(c_list[4], num_classes, 1)
            self.gt_conv2 = nn.Conv2d(c_list[3], num_classes, 1)
            self.gt_conv3 = nn.Conv2d(c_list[2], num_classes, 1)
            self.gt_conv4 = nn.Conv2d(c_list[1], num_classes, 1)
            self.gt_conv5 = nn.Conv2d(c_list[0], num_classes, 1)

        # Decoder stages (5 stages)
        self.decoder1 = DecoderStage(c_list[4], c_list[4], c_list[4], num_blocks, kernel_size, num_basis, dropout_rate, bridge_dropout)
        self.decoder2 = DecoderStage(c_list[4], c_list[3], c_list[3], num_blocks, kernel_size, num_basis, dropout_rate, bridge_dropout)
        self.decoder3 = DecoderStage(c_list[3], c_list[2], c_list[2], num_blocks, kernel_size, num_basis, dropout_rate, bridge_dropout)
        self.decoder4 = DecoderStage(c_list[2], c_list[1], c_list[1], num_blocks, kernel_size, num_basis, dropout_rate, bridge_dropout)
        self.decoder5 = DecoderStage(c_list[1], c_list[0], c_list[0], num_blocks, kernel_size, num_basis, dropout_rate, bridge_dropout)

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)
        self.apply(_init_weights)

    def forward(self, x):
        is_eval = not self.training

        # Encoder (5 stages)
        e1_out, ctx1 = self.encoder1(x, return_ctx=True)
        e1 = F.max_pool2d(e1_out, 2)  # 128

        e2_out, ctx2 = self.encoder2(e1, return_ctx=True)
        e2 = F.max_pool2d(e2_out, 2)  # 64

        e3_out, ctx3 = self.encoder3(e2, return_ctx=True)
        e3 = F.max_pool2d(e3_out, 2)  # 32

        e4_out, ctx4 = self.encoder4(e3, return_ctx=True)
        e4 = F.max_pool2d(e4_out, 2)  # 16

        e5_out, ctx5 = self.encoder5(e4, return_ctx=True)
        e5 = F.max_pool2d(e5_out, 2)  # 8

        # Bottleneck
        bn = self.bottleneck(e5)  # 8

        # Decoder (5 stages)
        d5 = self.decoder1(bn, ctx_enc=ctx5) + e5  # 8
        d4 = F.interpolate(self.decoder2(d5, ctx_enc=ctx4), scale_factor=2, mode='bilinear', align_corners=True) + e4  # 16
        d3 = F.interpolate(self.decoder3(d4, ctx_enc=ctx3), scale_factor=2, mode='bilinear', align_corners=True) + e3  # 32
        d2 = F.interpolate(self.decoder4(d3, ctx_enc=ctx2), scale_factor=2, mode='bilinear', align_corners=True) + e2  # 64
        d1 = F.interpolate(self.decoder5(d2, ctx_enc=ctx1), scale_factor=2, mode='bilinear', align_corners=True) + e1  # 128

        out = F.interpolate(self.final(d1), scale_factor=2, mode='bilinear', align_corners=True)

        if self.gt_ds and not is_eval:
            h, w = x.shape[2], x.shape[3]
            return (
                F.interpolate(self.gt_conv1(d5), (h, w), mode='bilinear', align_corners=True),
                F.interpolate(self.gt_conv2(d4), (h, w), mode='bilinear', align_corners=True),
                F.interpolate(self.gt_conv3(d3), (h, w), mode='bilinear', align_corners=True),
                F.interpolate(self.gt_conv4(d2), (h, w), mode='bilinear', align_corners=True),
                F.interpolate(self.gt_conv5(d1), (h, w), mode='bilinear', align_corners=True),
            ), out
        else:
            return out

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, (EncoderStage, DecoderStage)):
                m.switch_to_deploy()


if __name__ == "__main__":
    print("=" * 60)
    print("JeongWonNet_CtxBridge_Deep Test")
    print("5 stages × 2 blocks")
    print("=" * 60)

    model = JeongWonNet_CtxBridge_Deep(
        num_classes=1,
        input_channels=3,
        c_list=[24, 48, 96, 128, 192],
        kernel_size=7,
        num_basis=8,
        num_blocks=2,
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

    # Block count
    total_blocks = 0
    for m in model.modules():
        if isinstance(m, (DWBlock, DWBlock_Bridge)):
            total_blocks += 1
    print(f"Total DWBlocks: {total_blocks}")

    print("\n[Deep 구조]")
    print("  Encoder: 5 stages × 2 blocks = 10 blocks")
    print("  Bottleneck: 2 blocks")
    print("  Decoder: 5 stages × 2 blocks = 10 blocks")
    print("  채널: 24 → 48 → 96 → 128 → 192")
    print("  해상도: 256 → 128 → 64 → 32 → 16 → 8")
