import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_

# ============================================================
# 1. Efficient SCFFM (Spatial-Channel Feature Fusion Module)
# ============================================================
class EfficientSCFFM(nn.Module):
    """
    AMNet의 SCFFM을 경량화
    """
    def __init__(self, in_ch_low, in_ch_high, out_ch, reduction=16):
        super().__init__()
        
        # Low-level feature projection
        self.conv_low = nn.Conv2d(in_ch_low, out_ch, 1, bias=False)
        
        # High-level feature projection  
        self.conv_high = nn.Conv2d(in_ch_high, out_ch, 1, bias=False)
        
        # Efficient attention
        self.attention = EfficientSCModule(out_ch, reduction)
        
    def forward(self, x_low, x_high):
        """
        x_low: encoder feature (skip connection)
        x_high: decoder feature
        """
        # Channel alignment
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        
        # Global context from low-level
        low_context = F.adaptive_avg_pool2d(x_low, 1)
        
        # Attention on high-level
        x_high = self.attention(x_high)
        
        # Context modulation
        high_context = F.adaptive_avg_pool2d(x_high, 1)
        x_low = x_low * high_context
        
        # Fusion
        return x_low + x_high


class EfficientSCModule(nn.Module):
    """
    Spatial-Channel attention module (캐시 친화적)
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention (7x7 대신 depthwise separable)
        self.spatial_attn = nn.Sequential(
            # Depthwise 3x3
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # Pointwise to 1 channel
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        x_ch = x * self.channel_attn(x)
        # Spatial attention
        x_sp = x * self.spatial_attn(x)
        return x_ch + x_sp


# ============================================================
# 2. Efficient PAM (Position Attention Module)
# ============================================================
class EfficientPAM(nn.Module):
    """
    Window-based local attention (메모리 효율적)
    """
    def __init__(self, channels, reduction=8, window_size=8):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.reduction = reduction
        
        # Q, K, V projections
        self.query = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.key = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        
        # Learnable scale
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        
        # Strided attention으로 간소화 (window 대신)
        # Key와 Value를 downsampling
        stride = 2
        
        Q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C_r]
        
        # Strided K, V
        K_down = F.avg_pool2d(x, kernel_size=stride, stride=stride)
        K = self.key(K_down)
        K = K.view(B, -1, (H // stride) * (W // stride))  # [B, C_r, HW/4]
        
        V_down = F.avg_pool2d(x, kernel_size=stride, stride=stride)
        V = self.value(V_down)
        V = V.view(B, -1, (H // stride) * (W // stride))  # [B, C, HW/4]
        
        # Attention
        attn = torch.bmm(Q, K)  # [B, HW, HW/4]
        attn = F.softmax(attn / (self.channels // self.reduction) ** 0.5, dim=-1)
        
        # Output
        out = torch.bmm(attn, V.permute(0, 2, 1))  # [B, HW, C]
        out = out.permute(0, 2, 1).view(B, C, H, W)
        
        return x + self.gamma * out


# ============================================================
# 3. Efficient CAM (Channel Attention Module)
# ============================================================
class EfficientCAM(nn.Module):
    """
    Channel attention with low-rank approximation
    """
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Channel-wise attention
        # Q = K = spatial average
        proj_query = x.view(B, C, -1)  # [B, C, HW]
        proj_key = x.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        
        # Attention
        energy = torch.bmm(proj_query, proj_key)  # [B, C, C]
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new, dim=-1)
        
        # Apply attention
        proj_value = x.view(B, C, -1)  # [B, C, HW]
        out = torch.bmm(attention, proj_value)
        out = out.view(B, C, H, W)
        
        return x + self.gamma * out


# ============================================================
# 4. Combined PAM+CAM Layer
# ============================================================
class EfficientPAM_CAM_Layer(nn.Module):
    """
    MAResUNet 스타일의 PAM+CAM 결합 (경량화)
    """
    def __init__(self, channels, reduction=8, window_size=8):
        super().__init__()
        
        # Input processing
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention modules
        self.pam = EfficientPAM(channels, reduction, window_size)
        self.cam = EfficientCAM(channels)
        
        # Output projections
        self.conv_pam = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_cam = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Final fusion
        self.conv_final = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        
        # Parallel attention
        x_pam = self.conv_pam(self.pam(x))
        x_cam = self.conv_cam(self.cam(x))
        
        # Fusion
        out = x_pam + x_cam
        return self.conv_final(out)


# ============================================================
# 5. Simple Channel Attention (for encoder/decoder blocks)
# ============================================================
class SimpleChannelAttention(nn.Module):
    """
    간단한 SE-style attention (FLOPs 에러 방지)
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, _, _ = x.shape
        # Squeeze
        y = self.avg_pool(x).view(B, C)
        # Excitation
        y = self.fc(y).view(B, C, 1, 1)
        # Scale
        return x * y.expand_as(x)


# ============================================================
# 6. JeongWonNet with SCFFM + PAM/CAM
# ============================================================
class JeongWonNet_WithAttention(nn.Module):
    """
    JeongWonNet + SCFFM + PAM/CAM (경량화)
    """
    def __init__(self, num_classes=1, input_channels=3,
                 c_list=[24, 48, 64, 96, 128, 192],
                 gt_ds=True,
                 use_pam_cam=False,
                 use_scffm=True,
                 window_size=8):
        super().__init__()
        self.gt_ds = gt_ds
        self.use_pam_cam = use_pam_cam
        self.use_scffm = use_scffm

        def make_encoder_block(in_ch, out_ch):
            """Encoder/Decoder block with depthwise conv + attention"""
            layers = []
            if in_ch != out_ch:
                layers.append(nn.Conv2d(in_ch, out_ch, 1, bias=False))
            
            # Depthwise conv
            layers.extend([
                nn.Conv2d(out_ch, out_ch, 7, padding=3, groups=out_ch, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            
            # Channel attention
            layers.append(SimpleChannelAttention(out_ch, reduction=8))
            
            return nn.Sequential(*layers)

        # Encoder
        self.encoder1 = make_encoder_block(input_channels, c_list[0])
        self.encoder2 = make_encoder_block(c_list[0], c_list[1])
        self.encoder3 = make_encoder_block(c_list[1], c_list[2])
        self.encoder4 = make_encoder_block(c_list[2], c_list[3])
        self.encoder5 = make_encoder_block(c_list[3], c_list[4])
        self.encoder6 = make_encoder_block(c_list[4], c_list[5])

        # PAM+CAM on important layers
        if use_pam_cam:
            self.attn_bottleneck = EfficientPAM_CAM_Layer(c_list[5], window_size=window_size)
            self.attn_dec3 = EfficientPAM_CAM_Layer(c_list[3], window_size=window_size)

        # Decoder
        self.decoder1 = make_encoder_block(c_list[5], c_list[4])
        self.decoder2 = make_encoder_block(c_list[4], c_list[3])
        self.decoder3 = make_encoder_block(c_list[3], c_list[2])
        self.decoder4 = make_encoder_block(c_list[2], c_list[1])
        self.decoder5 = make_encoder_block(c_list[1], c_list[0])

        # SCFFM for skip connections
        if use_scffm:
            self.scffm5 = EfficientSCFFM(c_list[4], c_list[4], c_list[4], reduction=16)
            self.scffm4 = EfficientSCFFM(c_list[3], c_list[3], c_list[3], reduction=16)
            self.scffm3 = EfficientSCFFM(c_list[2], c_list[2], c_list[2], reduction=16)
            self.scffm2 = EfficientSCFFM(c_list[1], c_list[1], c_list[1], reduction=16)
            self.scffm1 = EfficientSCFFM(c_list[0], c_list[0], c_list[0], reduction=16)

        # Deep supervision
        if gt_ds:
            self.gt_conv1 = nn.Conv2d(c_list[4], num_classes, 1)
            self.gt_conv2 = nn.Conv2d(c_list[3], num_classes, 1)
            self.gt_conv3 = nn.Conv2d(c_list[2], num_classes, 1)
            self.gt_conv4 = nn.Conv2d(c_list[1], num_classes, 1)
            self.gt_conv5 = nn.Conv2d(c_list[0], num_classes, 1)

        # Final
        self.final = nn.Conv2d(c_list[0], num_classes, 1)

        self.apply(self._init_weights)

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
        # Encoder
        e1 = F.max_pool2d(self.encoder1(x), 2)
        e2 = F.max_pool2d(self.encoder2(e1), 2)
        e3 = F.max_pool2d(self.encoder3(e2), 2)
        e4 = F.max_pool2d(self.encoder4(e3), 2)
        e5 = F.max_pool2d(self.encoder5(e4), 2)
        e6 = self.encoder6(e5)

        # Bottleneck attention
        if self.use_pam_cam:
            e6 = self.attn_bottleneck(e6)

        # Decoder
        d5 = self.decoder1(e6)
        if self.use_scffm:
            d5 = self.scffm5(e5, d5)
        else:
            d5 = d5 + e5

        d4 = F.interpolate(self.decoder2(d5), scale_factor=2, mode='bilinear', align_corners=True)
        if self.use_scffm:
            d4 = self.scffm4(e4, d4)
        else:
            d4 = d4 + e4

        d3 = F.interpolate(self.decoder3(d4), scale_factor=2, mode='bilinear', align_corners=True)
        if self.use_pam_cam:
            d3 = self.attn_dec3(d3)
        if self.use_scffm:
            d3 = self.scffm3(e3, d3)
        else:
            d3 = d3 + e3

        d2 = F.interpolate(self.decoder4(d3), scale_factor=2, mode='bilinear', align_corners=True)
        if self.use_scffm:
            d2 = self.scffm2(e2, d2)
        else:
            d2 = d2 + e2

        d1 = F.interpolate(self.decoder5(d2), scale_factor=2, mode='bilinear', align_corners=True)
        if self.use_scffm:
            d1 = self.scffm1(e1, d1)
        else:
            d1 = d1 + e1

        # Final
        out = F.interpolate(self.final(d1), scale_factor=2, mode='bilinear', align_corners=True)

        # Deep supervision
        if self.gt_ds and self.training:
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


# ============================================================
# 7. 테스트용 간단한 버전
# ============================================================
if __name__ == "__main__":
    # SCFFM만 추가 (가장 안전)
    model1 = JeongWonNet_WithAttention(
        num_classes=1,
        use_pam_cam=False,
        use_scffm=True
    )
    
    # SCFFM + PAM/CAM (성능 최대화)
    model2 = JeongWonNet_WithAttention(
        num_classes=1,
        use_pam_cam=True,
        use_scffm=True,
        window_size=8
    )
    
    x = torch.randn(1, 3, 256, 256)
    
    print("Testing SCFFM only:")
    out1 = model1(x)
    print(f"Output shape: {out1.shape}")
    
    print("\nTesting SCFFM + PAM/CAM:")
    model2.eval()
    out2 = model2(x)
    print(f"Output shape: {out2.shape}")