import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_

# ============================================================
# 1. IO-Optimized PRCM: 메모리 접근 횟수 최소화
# ============================================================
class PRCM_IOAware(nn.Module):
    """
    기존 PRCM 문제점:
    - x.mean() → DRAM read (B*C*H*W)
    - matmul → DRAM read/write
    - Linear → DRAM read/write  
    - sigmoid → DRAM read/write
    - broadcast multiply → DRAM read (B*C*H*W)
    총 5회 이상의 별도 kernel launch → cache miss 폭발
    
    개선안: 단일 Conv 연산으로 fusion
    """
    def __init__(self, channels, num_basis=8, reduction=4):
        super().__init__()
        # SE-style bottleneck으로 basis projection 대체
        # 모든 연산을 Conv로 통일 → cuDNN kernel fusion 가능
        mid_ch = max(num_basis, channels // reduction)
        self.squeeze_excite = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B,C,H,W] → [B,C,1,1]
            nn.Conv2d(channels, mid_ch, 1, bias=False),
            nn.ReLU(inplace=True),  # GELU 대신 ReLU (더 빠름)
            nn.Conv2d(mid_ch, channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 단일 forward로 처리, intermediate tensor 최소화
        return x * self.squeeze_excite(x)


# ============================================================
# 2. Strided Convolution으로 Pooling 대체
# ============================================================
class StridedDWConv(nn.Module):
    """
    F.max_pool2d의 문제점:
    - 별도의 kernel launch → cache miss
    - depthwise conv와 분리되어 중간 결과 저장 필요
    
    개선안: stride=2 depthwise conv로 downsampling 통합
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2):
        super().__init__()
        padding = kernel_size // 2
        
        # Pointwise → Depthwise (stride) → Pointwise
        self.pw1 = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.dw = nn.Conv2d(out_ch, out_ch, kernel_size, stride=stride, 
                           padding=padding, groups=out_ch, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.pw1(x)
        x = self.dw(x)
        x = self.bn(x)
        return self.act(x)


# ============================================================
# 3. Lightweight RepConv (Inference 최적화)
# ============================================================
class RepConv_Lite(nn.Module):
    """
    기존 RepConv의 문제:
    - Training 시 3개 branch → 메모리 footprint 3배
    - 7x7 kernel → 5x5로 축소 (receptive field는 stacking으로 확보)
    """
    def __init__(self, channels, kernel_size=5, groups=None):
        super().__init__()
        if groups is None:
            groups = channels  # depthwise
        
        self.channels = channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # Main branch만 유지 (1x1, identity는 제거)
        self.conv = nn.Conv2d(channels, channels, kernel_size,
                             padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ============================================================
# 4. Pixel Shuffle Upsampling (Bilinear 대체)
# ============================================================
class PixelShuffleUp(nn.Module):
    """
    F.interpolate(mode='bilinear')의 문제:
    - 런타임에 weight 계산 → 캐시 불가
    - memory-bound operation
    
    개선안: 학습 가능한 sub-pixel convolution
    - weight는 학습 후 고정 → L1 캐시 적중률 극대화
    """
    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * (scale ** 2), 1, bias=False)
        self.shuffle = nn.PixelShuffle(scale)
        self.bn = nn.BatchNorm2d(out_ch)
    
    def forward(self, x):
        return self.bn(self.shuffle(self.conv(x)))


# ============================================================
# 5. In-place Skip Connection
# ============================================================
class FusedSkipConnection(nn.Module):
    """
    기존: d = decoder(x); d = d + skip → 2번 메모리 쓰기
    개선: in-place addition으로 1번만 쓰기
    """
    def forward(self, decoder_out, skip):
        # PyTorch의 in-place add
        return decoder_out.add_(skip)  # 메모리 할당 없음


# ============================================================
# 6. Grouped Deep Supervision (메모리 재사용)
# ============================================================
class GroupedDeepSupervision(nn.Module):
    """
    기존: 5개 독립 GT head → 5번 별도 interpolation
    개선: shared conv + single interpolation pass
    """
    def __init__(self, channels_list, num_classes=1):
        super().__init__()
        # 모든 feature를 single tensor로 concat → 단일 interpolation
        total_ch = sum(channels_list)
        self.reduce = nn.Conv2d(total_ch, num_classes * len(channels_list), 1)
        self.num_outputs = len(channels_list)
        self.num_classes = num_classes
        
    def forward(self, features, target_size):
        # features: list of [d5, d4, d3, d2, d1]
        # 1. Resize all to same spatial size (nearest neighbor)
        min_h = min(f.shape[2] for f in features)
        min_w = min(f.shape[3] for f in features)
        
        resized = []
        for f in features:
            if f.shape[2] != min_h:
                f = F.interpolate(f, size=(min_h, min_w), mode='nearest')
            resized.append(f)
        
        # 2. Concat and process
        concat = torch.cat(resized, dim=1)  # [B, total_ch, H, W]
        out = self.reduce(concat)  # [B, num_classes*5, H, W]
        
        # 3. Single interpolation to target size
        out = F.interpolate(out, size=target_size, mode='nearest')
        
        # 4. Split into individual outputs
        return torch.split(out, self.num_classes, dim=1)


# ============================================================
# 7. IO-Aware Full Model
# ============================================================
class JeongWonNet_IOOptimized(nn.Module):
    """
    IO-Aware 최적화 핵심 원칙:
    1. Kernel Fusion: 연속된 연산을 단일 CUDA kernel로 병합
    2. In-place Operations: 불필요한 메모리 할당 제거
    3. Cache-Friendly Patterns: stride conv > pooling, PixelShuffle > interpolate
    4. Reduced Branching: RepConv 간소화
    """
    def __init__(self, num_classes=1, input_channels=3, 
                 c_list=[24, 48, 64, 96, 128, 192], gt_ds=True):
        super().__init__()
        self.gt_ds = gt_ds
        
        # ===== Encoder with Strided Convs =====
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(c_list[0]),
            nn.ReLU(inplace=True)
        )
        
        # Stride=2로 downsampling (max_pool 제거)
        self.enc1 = self._make_encoder_block(c_list[0], c_list[0])
        self.down1 = StridedDWConv(c_list[0], c_list[1])
        
        self.enc2 = self._make_encoder_block(c_list[1], c_list[1])
        self.down2 = StridedDWConv(c_list[1], c_list[2])
        
        self.enc3 = self._make_encoder_block(c_list[2], c_list[2])
        self.down3 = StridedDWConv(c_list[2], c_list[3])
        
        self.enc4 = self._make_encoder_block(c_list[3], c_list[3])
        self.down4 = StridedDWConv(c_list[3], c_list[4])
        
        self.enc5 = self._make_encoder_block(c_list[4], c_list[4])
        self.down5 = StridedDWConv(c_list[4], c_list[5])
        
        self.bottleneck = self._make_encoder_block(c_list[5], c_list[5])
        
        # ===== Decoder with PixelShuffle =====
        self.up5 = PixelShuffleUp(c_list[5], c_list[4])
        self.dec5 = self._make_decoder_block(c_list[4], c_list[4])
        
        self.up4 = PixelShuffleUp(c_list[4], c_list[3])
        self.dec4 = self._make_decoder_block(c_list[3], c_list[3])
        
        self.up3 = PixelShuffleUp(c_list[3], c_list[2])
        self.dec3 = self._make_decoder_block(c_list[2], c_list[2])
        
        self.up2 = PixelShuffleUp(c_list[2], c_list[1])
        self.dec2 = self._make_decoder_block(c_list[1], c_list[1])
        
        self.up1 = PixelShuffleUp(c_list[1], c_list[0])
        self.dec1 = self._make_decoder_block(c_list[0], c_list[0])
        
        # Final head
        self.final_up = PixelShuffleUp(c_list[0], c_list[0])
        self.final_conv = nn.Conv2d(c_list[0], num_classes, 1)
        
        # Deep Supervision (optional)
        if gt_ds:
            self.ds_head = GroupedDeepSupervision(
                [c_list[4], c_list[3], c_list[2], c_list[1], c_list[0]],
                num_classes
            )
        
        self.apply(self._init_weights)
    
    def _make_encoder_block(self, in_ch, out_ch):
        """Encoder block: PW → DW → PRCM"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity(),
            RepConv_Lite(out_ch, kernel_size=5),
            PRCM_IOAware(out_ch, num_basis=8)
        )
    
    def _make_decoder_block(self, in_ch, out_ch):
        """Decoder block: 동일 구조"""
        return nn.Sequential(
            RepConv_Lite(in_ch, kernel_size=5),
            PRCM_IOAware(in_ch, num_basis=8),
            nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        )
    
    def _init_weights(self, m):
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
    
    def forward(self, x):
        # ===== Encoder =====
        e0 = self.stem(x)
        e1 = self.enc1(e0)
        e1_down = self.down1(e1)
        
        e2 = self.enc2(e1_down)
        e2_down = self.down2(e2)
        
        e3 = self.enc3(e2_down)
        e3_down = self.down3(e3)
        
        e4 = self.enc4(e3_down)
        e4_down = self.down4(e4)
        
        e5 = self.enc5(e4_down)
        e5_down = self.down5(e5)
        
        bottleneck = self.bottleneck(e5_down)
        
        # ===== Decoder with In-place Skip Connections =====
        d5 = self.up5(bottleneck)
        d5 = self.dec5(d5.add_(e5))  # in-place add
        
        d4 = self.up4(d5)
        d4 = self.dec4(d4.add_(e4))
        
        d3 = self.up3(d4)
        d3 = self.dec3(d3.add_(e3))
        
        d2 = self.up2(d3)
        d2 = self.dec2(d2.add_(e2))
        
        d1 = self.up1(d2)
        d1 = self.dec1(d1.add_(e1))
        
        # Final output
        out = self.final_up(d1)
        out = self.final_conv(out)
        
        # Deep Supervision
        if self.gt_ds and self.training:
            ds_outputs = self.ds_head([d5, d4, d3, d2, d1], x.shape[2:])
            return ds_outputs, out
        else:
            return out