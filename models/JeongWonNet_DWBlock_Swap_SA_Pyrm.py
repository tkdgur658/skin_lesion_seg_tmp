import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_

class SpatialAttention(nn.Module):
    """
    Lightweight Spatial Attention for Low-Resolution Feature Maps

    저해상도(8x8, 16x16)에서 효율적인 spatial attention
    Skin lesion이 뭉쳐있는 특성을 활용하여 병변 영역에 집중

    Structure:
        Pool(2x2) -> 1x1 Conv expand -> ReLU -> 1x1 Conv -> Upsample -> Sigmoid -> x * att

    Working Set 최소화:
        - 2x2 pooling으로 spatial 크기 축소
        - 1x1 conv로 파라미터 효율적
        - 저해상도에서만 사용 (encoder5, encoder6, bottleneck)
    """
    def __init__(self, channels, pool_size=2, expansion=2, dropout_rate=0.2):
        super().__init__()
        self.pool_size = pool_size
        hidden = channels * expansion

        # Pooled spatial processing
        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.ReLU(inplace=True)

        # Attention map generation (output: 1 channel)
        self.conv2 = nn.Conv2d(hidden, 1, 1, bias=False)

        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        # Pool to small size (use avg_pool2d for deterministic backward)
        if H >= self.pool_size and W >= self.pool_size:
            kernel_h = H // self.pool_size
            kernel_w = W // self.pool_size
            pooled = F.avg_pool2d(x, kernel_size=(kernel_h, kernel_w))
        else:
            pooled = x

        # Generate attention map
        att = self.conv1(pooled)
        att = self.bn1(att)
        att = self.act(att)
        att = self.dropout(att)
        att = self.conv2(att)

        # Upsample and apply sigmoid
        att = F.interpolate(att, size=(H, W), mode='bilinear', align_corners=True)
        att = torch.sigmoid(att)

        return x * att


class SpatialAttentionHeavy(nn.Module):
    """
    Original Spatial Attention with RepConv (for comparison)
    """
    def __init__(self, in_channels, reduction=8, dropout_rate=0.2):
        super(SpatialAttentionHeavy, self).__init__()
        intermediate_channels = max(in_channels // reduction, 8)

        self.conv1_rep = RepConv(in_channels, intermediate_channels, 7, 1, 3,
                                use_identity=True, use_activation=True)

        self.dropout_1 = nn.Dropout2d(dropout_rate)
        self.dropout_2 = nn.Dropout2d(dropout_rate)

        # Attention map generation
        self.conv2 = nn.Conv2d(intermediate_channels, 1, 3, 1, 1, bias=False)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x_att = self.conv1_rep(x)
        x_mod = x_att
        x_mod = self.dropout_1(x_mod)

        attention_map = self.conv2(x_mod)
        attention_map = self.act2(attention_map)
        output = x * attention_map
        output = self.dropout_2(output)

        return output

    def switch_to_deploy(self):
        """RepConv를 융합"""
        self.conv1_rep.switch_to_deploy()
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
        [1x1 Conv] (if in_ch != out_ch) -> RepConv 7x7 DW -> PRCM

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

    def forward(self, x):
        if self.pw_conv is not None:
            x = self.pw_conv(x)

        x = self.dw_conv(x)
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

class JeongWonNet_DWBlock_Swap_SA_Pyrm(nn.Module):
    """
    JeongWonNet with DWBlock + Spatial Attention at Low Resolution

    Skin lesion 특성:
    - 병변이 뭉쳐있음 → Spatial Attention으로 병변 영역 집중
    - 저해상도(8x8, 16x16)에서 SA 적용 → Working Set 오버헤드 최소화

    Structure:
    - DWBlock: [1x1 Conv] -> RepConv 7x7 DW -> PRCM
    - Spatial Attention: encoder5, encoder6 (bottleneck) 이후 적용
    - UCMNet 스타일의 단순한 skip connection (decoder + encoder)

    SA 적용 위치:
    - e5 (8x8): encoder5 출력에 SA 적용
    - e6 (8x8): bottleneck에 SA 적용 (가장 중요)
    """
    def __init__(self,
                 num_classes=1,
                 input_channels=3,
                 c_list=[16, 32, 64, 128, 196, 256],
                 kernel_size=7,
                 num_basis=8,
                 dropout_rate=0.5,
                 sa_pool_size=2,
                 sa_expansion=2,
                 sa_dropout=0.2,
                 gt_ds=True):
        super().__init__()
        self.gt_ds = gt_ds

        # Encoder blocks
        self.encoder1 = DWBlock(input_channels, c_list[0], kernel_size, num_basis, dropout_rate)
        self.encoder2 = DWBlock(c_list[0], c_list[1], kernel_size, num_basis, dropout_rate)
        self.encoder3 = DWBlock(c_list[1], c_list[2], kernel_size, num_basis, dropout_rate)
        self.encoder4 = DWBlock(c_list[2], c_list[3], kernel_size, num_basis, dropout_rate)
        self.encoder5 = DWBlock(c_list[3], c_list[4], kernel_size, num_basis, dropout_rate)
        self.encoder6 = DWBlock(c_list[4], c_list[5], kernel_size, num_basis, dropout_rate)

        # Spatial Attention at low resolution (8x8)
        # e5: 128ch @ 8x8, e6: 192ch @ 8x8
        self.sa_e5 = SpatialAttention(c_list[4], pool_size=sa_pool_size,
                                       expansion=sa_expansion, dropout_rate=sa_dropout)
        self.sa_e6 = SpatialAttention(c_list[5], pool_size=sa_pool_size,
                                       expansion=sa_expansion, dropout_rate=sa_dropout)

        # Deep Supervision heads
        if gt_ds:
            self.gt_conv1 = nn.Conv2d(c_list[4], num_classes, 1)
            self.gt_conv2 = nn.Conv2d(c_list[3], num_classes, 1)
            self.gt_conv3 = nn.Conv2d(c_list[2], num_classes, 1)
            self.gt_conv4 = nn.Conv2d(c_list[1], num_classes, 1)
            self.gt_conv5 = nn.Conv2d(c_list[0], num_classes, 1)

        # Decoder blocks
        self.decoder1 = DWBlock(c_list[5], c_list[4], kernel_size, num_basis, dropout_rate)
        self.decoder2 = DWBlock(c_list[4], c_list[3], kernel_size, num_basis, dropout_rate)
        self.decoder3 = DWBlock(c_list[3], c_list[2], kernel_size, num_basis, dropout_rate)
        self.decoder4 = DWBlock(c_list[2], c_list[1], kernel_size, num_basis, dropout_rate)
        self.decoder5 = DWBlock(c_list[1], c_list[0], kernel_size, num_basis, dropout_rate)

        # Final 1x1 conv
        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        # Initialize weights
        self.apply(_init_weights)

    def forward(self, x):
        is_eval = not self.training

        # Encoder
        e1 = F.max_pool2d(self.encoder1(x), 2)   # 128x128
        e2 = F.max_pool2d(self.encoder2(e1), 2)  # 64x64
        e3 = F.max_pool2d(self.encoder3(e2), 2)  # 32x32
        e4 = F.max_pool2d(self.encoder4(e3), 2)  # 16x16
        e5 = F.max_pool2d(self.encoder5(e4), 2)  # 8x8

        # Spatial Attention at e5 (low resolution)
        e5 = self.sa_e5(e5)

        # Bottleneck with Spatial Attention
        e6 = self.encoder6(e5)  # 8x8
        e6 = self.sa_e6(e6)

        # Decoder with skip connections
        d5 = self.decoder1(e6) + e5
        d4 = F.interpolate(self.decoder2(d5), scale_factor=2, mode='bilinear', align_corners=True) + e4
        d3 = F.interpolate(self.decoder3(d4), scale_factor=2, mode='bilinear', align_corners=True) + e3
        d2 = F.interpolate(self.decoder4(d3), scale_factor=2, mode='bilinear', align_corners=True) + e2
        d1 = F.interpolate(self.decoder5(d2), scale_factor=2, mode='bilinear', align_corners=True) + e1

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
            if isinstance(m, DWBlock):
                m.switch_to_deploy()


if __name__ == "__main__":
    print("=" * 60)
    print("JeongWonNet_DWBlock_Swap_SA Test")
    print("=" * 60)

    model = JeongWonNet_DWBlock_Swap_SA(
        num_classes=1,
        input_channels=3,
        c_list=[24, 48, 64, 96, 128, 192],
        kernel_size=7,
        num_basis=8,
        dropout_rate=0.5,
        sa_pool_size=2,
        sa_expansion=2,
        sa_dropout=0.2,
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

    # Spatial Attention params
    sa_params = sum(p.numel() for p in model.sa_e5.parameters())
    sa_params += sum(p.numel() for p in model.sa_e6.parameters())
    print(f"Spatial Attention Parameters: {sa_params:,}")

    # Deploy mode test
    model.switch_to_deploy()
    with torch.no_grad():
        out_deploy = model(x)
    print(f"\nDeploy Mode Output: {out_deploy.shape}")

    print("\n[Spatial Attention 구조]")
    print("  - SA @ e5: 128ch @ 8x8 (pool_size=2 → 2x2)")
    print("  - SA @ e6: 192ch @ 8x8 (pool_size=2 → 2x2)")
    print("  - Working Set 추가: ~0.02MB (매우 작음)")
    print("  - 병변 영역에 집중하는 attention map 생성")
