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

    return_ctx=True일 때 (x * w, ctx) 반환 → encoder에서 ctx 저장용
    """
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


class PRCM_CrossStage(nn.Module):
    """
    Cross-Stage Pattern Recalibration Module

    Encoder의 global context를 decoder에서 활용

    Structure:
        ctx_self = GAP(x)                    # 자신의 context
        ctx_fused = ctx_self + proj(ctx_enc) # encoder context 융합
        coeff = ctx_fused @ basis.t()
        w = fuser(coeff).sigmoid()
        return x * w

    이점:
        - Encoder의 멀티스케일 정보를 decoder에서 활용
        - 저해상도 encoder ctx → 병변 위치/크기 정보
        - 고해상도 encoder ctx → 경계/텍스처 정보
    """
    def __init__(self, channels, enc_channels, num_basis=8, dropout_rate=0.5):
        super().__init__()
        self.num_basis = num_basis
        self.channels = channels

        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        self.fuser = nn.Linear(num_basis, channels, bias=False)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Encoder context projection (채널 맞춤)
        if enc_channels != channels:
            self.ctx_proj = nn.Linear(enc_channels, channels, bias=False)
        else:
            self.ctx_proj = nn.Identity()

    def forward(self, x, ctx_enc=None):
        B, C, H, W = x.shape

        # Self context
        ctx_self = x.mean(dim=[2, 3])

        # Fuse with encoder context
        if ctx_enc is not None:
            ctx_enc_proj = self.ctx_proj(ctx_enc)
            ctx_fused = ctx_self + ctx_enc_proj
        else:
            ctx_fused = ctx_self

        coeff = ctx_fused @ self.basis.t()
        coeff = self.coeff_dropout(coeff)

        w = self.fuser(coeff).sigmoid().unsqueeze(-1).unsqueeze(-1)
        return x * w


class DWBlock(nn.Module):
    """
    Depthwise Block (Encoder용)

    Structure:
        [1x1 Conv] (if in_ch != out_ch) -> RepConv 7x7 DW -> PRCM

    forward(x, return_ctx=True)로 ctx 반환 가능
    """
    def __init__(self, in_channels, out_channels, kernel_size=7, num_basis=8, dropout_rate=0.5):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # 채널 조정이 필요할 때만 1x1 pointwise conv
        if in_channels != out_channels:
            self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.pw_conv = None

        # Depthwise RepConv
        self.dw_conv = RepConv(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=out_channels
        )

        # Pattern Recalibration Module
        self.prcm = PRCM(out_channels, num_basis=num_basis, dropout_rate=dropout_rate)

    def forward(self, x, return_ctx=False):
        if self.pw_conv is not None:
            x = self.pw_conv(x)

        x = self.dw_conv(x)

        if return_ctx:
            x, ctx = self.prcm(x, return_ctx=True)
            return x, ctx

        x = self.prcm(x)
        return x

    def switch_to_deploy(self):
        """추론 모드로 전환"""
        self.dw_conv.switch_to_deploy()


class DWBlock_CrossStage(nn.Module):
    """
    Depthwise Block with Cross-Stage Context (Decoder용)

    Structure:
        [1x1 Conv] (if in_ch != out_ch) -> RepConv 7x7 DW -> PRCM_CrossStage

    forward(x, ctx_enc)로 encoder context 입력
    """
    def __init__(self, in_channels, out_channels, enc_channels, kernel_size=7, num_basis=8, dropout_rate=0.5):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # 채널 조정이 필요할 때만 1x1 pointwise conv
        if in_channels != out_channels:
            self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.pw_conv = None

        # Depthwise RepConv
        self.dw_conv = RepConv(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=out_channels
        )

        # Cross-Stage Pattern Recalibration Module
        self.prcm = PRCM_CrossStage(out_channels, enc_channels, num_basis=num_basis, dropout_rate=dropout_rate)

    def forward(self, x, ctx_enc=None):
        if self.pw_conv is not None:
            x = self.pw_conv(x)

        x = self.dw_conv(x)
        x = self.prcm(x, ctx_enc)

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

class JeongWonNet_DWBlock_Swap_CtxBridge(nn.Module):
    """
    JeongWonNet with Cross-Stage Context Bridge

    Encoder의 PRCM global context를 Decoder에 전달하여
    멀티스케일 정보 활용

    Context Bridge 구조:
        Encoder: e1_ctx, e2_ctx, ..., e5_ctx 저장 (pooling 전)
        Decoder: d1(ctx=e5), d2(ctx=e4), ..., d5(ctx=e1)

    이점:
        - 저해상도 enc ctx → 병변 위치/크기 정보
        - 고해상도 enc ctx → 경계/텍스처 정보
        - Decoder에서 encoder 정보 직접 참조
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

        # Encoder blocks (ctx 반환)
        self.encoder1 = DWBlock(input_channels, c_list[0], kernel_size, num_basis, dropout_rate)
        self.encoder2 = DWBlock(c_list[0], c_list[1], kernel_size, num_basis, dropout_rate)
        self.encoder3 = DWBlock(c_list[1], c_list[2], kernel_size, num_basis, dropout_rate)
        self.encoder4 = DWBlock(c_list[2], c_list[3], kernel_size, num_basis, dropout_rate)
        self.encoder5 = DWBlock(c_list[3], c_list[4], kernel_size, num_basis, dropout_rate)
        self.encoder6 = DWBlock(c_list[4], c_list[5], kernel_size, num_basis, dropout_rate)

        # Deep Supervision heads
        if gt_ds:
            self.gt_conv1 = nn.Conv2d(c_list[4], num_classes, 1)
            self.gt_conv2 = nn.Conv2d(c_list[3], num_classes, 1)
            self.gt_conv3 = nn.Conv2d(c_list[2], num_classes, 1)
            self.gt_conv4 = nn.Conv2d(c_list[1], num_classes, 1)
            self.gt_conv5 = nn.Conv2d(c_list[0], num_classes, 1)

        # Decoder blocks with Cross-Stage Context
        # decoder1: 192→128, enc_ctx from encoder5 (128ch)
        # decoder2: 128→96,  enc_ctx from encoder4 (96ch)
        # decoder3: 96→64,   enc_ctx from encoder3 (64ch)
        # decoder4: 64→48,   enc_ctx from encoder2 (48ch)
        # decoder5: 48→24,   enc_ctx from encoder1 (24ch)
        self.decoder1 = DWBlock_CrossStage(c_list[5], c_list[4], c_list[4], kernel_size, num_basis, dropout_rate)
        self.decoder2 = DWBlock_CrossStage(c_list[4], c_list[3], c_list[3], kernel_size, num_basis, dropout_rate)
        self.decoder3 = DWBlock_CrossStage(c_list[3], c_list[2], c_list[2], kernel_size, num_basis, dropout_rate)
        self.decoder4 = DWBlock_CrossStage(c_list[2], c_list[1], c_list[1], kernel_size, num_basis, dropout_rate)
        self.decoder5 = DWBlock_CrossStage(c_list[1], c_list[0], c_list[0], kernel_size, num_basis, dropout_rate)

        # Final 1x1 conv
        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        # Initialize weights
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

        # Decoder with cross-stage context
        d5 = self.decoder1(e6, ctx_enc=ctx5) + e5
        d4 = F.interpolate(self.decoder2(d5, ctx_enc=ctx4), scale_factor=2, mode='bilinear', align_corners=True) + e4
        d3 = F.interpolate(self.decoder3(d4, ctx_enc=ctx3), scale_factor=2, mode='bilinear', align_corners=True) + e3
        d2 = F.interpolate(self.decoder4(d3, ctx_enc=ctx2), scale_factor=2, mode='bilinear', align_corners=True) + e2
        d1 = F.interpolate(self.decoder5(d2, ctx_enc=ctx1), scale_factor=2, mode='bilinear', align_corners=True) + e1

        # Final output
        out = F.interpolate(self.final(d1), scale_factor=2, mode='bilinear', align_corners=True)

        # Deep supervision
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
        """모든 DWBlock을 추론 모드로 전환"""
        for m in self.modules():
            if isinstance(m, (DWBlock, DWBlock_CrossStage)):
                m.switch_to_deploy()


if __name__ == "__main__":
    print("=" * 60)
    print("JeongWonNet_DWBlock_Swap_CtxBridge Test")
    print("=" * 60)

    model = JeongWonNet_DWBlock_Swap_CtxBridge(
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

    print("\nTraining Mode:")
    print(f"  Final Output: {final_out.shape}")
    print(f"  Deep Supervision: {len(ds_outputs)} levels")

    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"\nEvaluation Mode:")
    print(f"  Output: {out.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params:,}")

    # Context Bridge 파라미터
    ctx_bridge_params = 0
    for name, m in model.named_modules():
        if isinstance(m, PRCM_CrossStage):
            ctx_bridge_params += sum(p.numel() for p in m.parameters())
    print(f"Context Bridge (PRCM_CrossStage) Parameters: {ctx_bridge_params:,}")

    # Deploy mode test
    model.switch_to_deploy()
    with torch.no_grad():
        out_deploy = model(x)
    print(f"\nDeploy Mode Output: {out_deploy.shape}")

    print("\n[Cross-Stage Context Bridge 구조]")
    print("  Encoder: ctx1(24ch), ctx2(48ch), ctx3(64ch), ctx4(96ch), ctx5(128ch)")
    print("  Decoder: d1←ctx5, d2←ctx4, d3←ctx3, d4←ctx2, d5←ctx1")
    print("  ctx_fused = ctx_self + proj(ctx_enc)")
    print("  멀티스케일 encoder 정보를 decoder에서 직접 활용")
