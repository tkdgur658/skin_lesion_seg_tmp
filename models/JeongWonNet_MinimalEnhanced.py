import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_

# ============================================================
# 1. Zero-Cost Skip Connection Enhancement
# ============================================================
class ZeroCostSkipFusion(nn.Module):
    """
    기존: d + e (단순 덧셈)
    개선: Learnable weighted sum
    
    추가 FLOPs: 0 (단순 scalar multiplication)
    성능 향상: Skip connection의 기여도 학습
    """
    def __init__(self):
        super().__init__()
        # Learnable weight (초기값 0.5)
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, decoder_feat, encoder_feat):
        # alpha * decoder + (1-alpha) * encoder
        return self.alpha * decoder_feat + (1 - self.alpha) * encoder_feat


# ============================================================
# 2. Efficient Boundary-Aware PRCM
# ============================================================
class PRCM_BoundaryAware(nn.Module):
    """
    기존 PRCM에 경계 정보 추가하되, 추가 연산 최소화
    
    아이디어: Channel attention 계산 시 max와 avg를 모두 사용
    (SE-Net 변형, 추가 연산 거의 없음)
    """
    def __init__(self, channels, num_basis=8):
        super().__init__()
        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        self.fuser = nn.Linear(num_basis, channels, bias=False)
        
        # Dual pooling weight (avg vs max)
        self.pool_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Dual global pooling (avg + max)
        ctx_avg = x.mean(dim=[2, 3])  # 기존
        ctx_max = x.flatten(2).max(dim=2)[0]  # 추가 (경계 정보)
        
        # Weighted combination
        ctx = self.pool_weight * ctx_avg + (1 - self.pool_weight) * ctx_max
        
        # 나머지는 기존과 동일
        coeff = ctx @ self.basis.t()
        w = self.fuser(coeff).sigmoid().unsqueeze(-1).unsqueeze(-1)
        return x * w


# ============================================================
# 3. Deep Supervision Improvement (Zero Cost)
# ============================================================
class ImprovedDeepSupervision(nn.Module):
    """
    기존: 각 decoder 출력을 독립적으로 supervision
    개선: Decoder 간 consistency regularization
    
    추가 loss term만 추가, inference 시 overhead 없음
    """
    def __init__(self, channels_list, num_classes=1):
        super().__init__()
        self.gt_convs = nn.ModuleList([
            nn.Conv2d(ch, num_classes, 1) for ch in channels_list
        ])
    
    def forward(self, features, target_size):
        outputs = []
        for conv, feat in zip(self.gt_convs, features):
            out = conv(feat)
            out = F.interpolate(out, target_size, mode='bilinear', align_corners=True)
            outputs.append(out)
        return outputs
    
    def consistency_loss(self, outputs):
        """
        Deep supervision outputs 간의 consistency 강제
        Training loss에 추가
        """
        loss = 0
        for i in range(len(outputs) - 1):
            loss += F.mse_loss(outputs[i], outputs[i+1].detach())
        return loss


# ============================================================
# 4. Attention Distillation (Training Trick)
# ============================================================
class AttentionDistillation(nn.Module):
    """
    Encoder의 attention을 decoder로 전달
    
    추가 연산: Training 시에만, Inference는 동일
    """
    def __init__(self, channels):
        super().__init__()
        # 1x1 conv for attention transfer
        self.transfer = nn.Conv2d(channels, channels, 1, bias=False)
    
    def forward(self, encoder_feat, decoder_feat, training=True):
        if training:
            # Encoder의 attention map 생성
            enc_attn = encoder_feat.pow(2).mean(1, keepdim=True)
            enc_attn = F.softmax(enc_attn.flatten(2), dim=2).view_as(enc_attn)
            
            # Decoder에 전달
            decoder_feat = decoder_feat + self.transfer(encoder_feat) * enc_attn
        
        return decoder_feat


# ============================================================
# 5. Minimal-Overhead Enhanced Model
# ============================================================
class JeongWonNet_MinimalEnhanced(nn.Module):
    """
    성능 개선 전략:
    1. Zero-cost skip fusion
    2. PRCM → PRCM_BoundaryAware (dual pooling)
    3. Deep supervision consistency loss
    4. (Optional) Attention distillation
    
    목표:
    - FLOPs: +5% 이하
    - Latency: +10% 이하  
    - Dice: +1-2% 개선
    """
    def __init__(self, num_classes=1, input_channels=3, 
                 c_list=[24, 48, 64, 96, 128, 192], 
                 gt_ds=True,
                 use_distillation=False):
        super().__init__()
        self.gt_ds = gt_ds
        self.use_distillation = use_distillation

        def make_block(in_ch, out_ch, use_boundary_aware=True):
            layers = []
            if in_ch != out_ch:
                layers.append(nn.Conv2d(in_ch, out_ch, 1, bias=False))
            
            # 7x7 depthwise conv
            layers.append(nn.Conv2d(out_ch, out_ch, 7, padding=3, groups=out_ch, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            
            # PRCM variant
            if use_boundary_aware:
                layers.append(PRCM_BoundaryAware(out_ch, num_basis=8))
            else:
                # 기존 PRCM 유지 가능
                from your_original_code import PRCM
                layers.append(PRCM(out_ch, num_basis=8))
            
            return nn.Sequential(*layers)

        # Encoder
        self.encoder1 = make_block(input_channels, c_list[0])
        self.encoder2 = make_block(c_list[0], c_list[1])
        self.encoder3 = make_block(c_list[1], c_list[2])
        self.encoder4 = make_block(c_list[2], c_list[3])
        self.encoder5 = make_block(c_list[3], c_list[4])
        self.encoder6 = make_block(c_list[4], c_list[5])

        # Decoder
        self.decoder1 = make_block(c_list[5], c_list[4])
        self.decoder2 = make_block(c_list[4], c_list[3])
        self.decoder3 = make_block(c_list[3], c_list[2])
        self.decoder4 = make_block(c_list[2], c_list[1])
        self.decoder5 = make_block(c_list[1], c_list[0])

        # Zero-cost skip fusion
        self.skip5 = ZeroCostSkipFusion()
        self.skip4 = ZeroCostSkipFusion()
        self.skip3 = ZeroCostSkipFusion()
        self.skip2 = ZeroCostSkipFusion()
        self.skip1 = ZeroCostSkipFusion()

        # Attention distillation (optional)
        if use_distillation:
            self.distill5 = AttentionDistillation(c_list[4])
            self.distill4 = AttentionDistillation(c_list[3])
            self.distill3 = AttentionDistillation(c_list[2])
            self.distill2 = AttentionDistillation(c_list[1])
            self.distill1 = AttentionDistillation(c_list[0])

        # Deep Supervision
        if gt_ds:
            self.ds = ImprovedDeepSupervision(
                [c_list[4], c_list[3], c_list[2], c_list[1], c_list[0]],
                num_classes
            )

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
        # === Encoder ===
        e1 = F.max_pool2d(self.encoder1(x), 2)
        e2 = F.max_pool2d(self.encoder2(e1), 2)
        e3 = F.max_pool2d(self.encoder3(e2), 2)
        e4 = F.max_pool2d(self.encoder4(e3), 2)
        e5 = F.max_pool2d(self.encoder5(e4), 2)
        e6 = self.encoder6(e5)

        # === Decoder ===
        d5 = self.decoder1(e6)
        if self.use_distillation and self.training:
            d5 = self.distill5(e5, d5, training=True)
        d5 = self.skip5(d5, e5)

        d4 = F.interpolate(self.decoder2(d5), scale_factor=2, mode='bilinear', align_corners=True)
        if self.use_distillation and self.training:
            d4 = self.distill4(e4, d4, training=True)
        d4 = self.skip4(d4, e4)

        d3 = F.interpolate(self.decoder3(d4), scale_factor=2, mode='bilinear', align_corners=True)
        if self.use_distillation and self.training:
            d3 = self.distill3(e3, d3, training=True)
        d3 = self.skip3(d3, e3)

        d2 = F.interpolate(self.decoder4(d3), scale_factor=2, mode='bilinear', align_corners=True)
        if self.use_distillation and self.training:
            d2 = self.distill2(e2, d2, training=True)
        d2 = self.skip2(d2, e2)

        d1 = F.interpolate(self.decoder5(d2), scale_factor=2, mode='bilinear', align_corners=True)
        if self.use_distillation and self.training:
            d1 = self.distill1(e1, d1, training=True)
        d1 = self.skip1(d1, e1)

        # Final
        out = F.interpolate(self.final(d1), scale_factor=2, mode='bilinear', align_corners=True)

        # Deep supervision
        if self.gt_ds and self.training:
            ds_outputs = self.ds([d5, d4, d3, d2, d1], x.shape[2:])
            return ds_outputs, out
        else:
            return out
    
    def compute_loss(self, outputs, target, criterion):
        """
        Enhanced training loss
        """
        if isinstance(outputs, tuple):
            ds_outputs, main_output = outputs
            
            # Main loss
            main_loss = criterion(main_output, target)
            
            # Deep supervision loss
            ds_loss = sum(criterion(out, target) for out in ds_outputs) / len(ds_outputs)
            
            # Consistency loss (추가)
            consistency_loss = self.ds.consistency_loss(ds_outputs)
            
            # Total loss
            total_loss = main_loss + 0.4 * ds_loss + 0.1 * consistency_loss
            return total_loss
        else:
            return criterion(outputs, target)