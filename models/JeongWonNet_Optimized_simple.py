import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_


# ============================================================================
# 기본 RepConv (재사용)
# ============================================================================
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
            out += self.bn_1x1(self.conv_1x1(x))
        if self.use_identity:
            out += self.bn_identity(x)
        
        return self.activation(out)


# ============================================================================
# 최적화 버전 1: Fused Affine PRCM (Linear 1개로 통합)
# ============================================================================
class FusedAffinePRCM(nn.Module):
    """
    최적화: Scale과 Shift를 하나의 Linear로 처리
    [B, K] -> [B, 2C] 한 번에 생성 후 split
    """
    def __init__(self, channels, num_basis=8, dropout_rate=0.5):
        super().__init__()
        self.num_basis = num_basis
        self.channels = channels
        
        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        # 하나의 Linear로 Scale + Shift 동시 생성
        self.affine_proj = nn.Linear(num_basis, channels * 2, bias=False)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        ctx = x.mean(dim=[2, 3])
        coeff = ctx @ self.basis.t()
        coeff = self.coeff_dropout(coeff)
        
        # 한 번에 생성 후 split
        affine_params = self.affine_proj(coeff)  # [B, 2C]
        alpha, beta = affine_params.chunk(2, dim=1)  # [B, C], [B, C]
        
        alpha = alpha.sigmoid().unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        return x * alpha + beta


# ============================================================================
# 최적화 버전 2: Lightweight Affine PRCM (Grouped)
# ============================================================================
class LightweightAffinePRCM(nn.Module):
    """
    최적화: 채널을 그룹으로 나눠서 처리 (복잡도 감소)
    전체 채널에 동일한 affine 적용하는 대신, 그룹별로 다른 affine 적용
    """
    def __init__(self, channels, num_basis=8, num_groups=4, dropout_rate=0.5):
        super().__init__()
        self.num_basis = num_basis
        self.channels = channels
        self.num_groups = num_groups
        self.channels_per_group = channels // num_groups
        
        assert channels % num_groups == 0, f"channels {channels} must be divisible by num_groups {num_groups}"
        
        # 그룹별 basis
        self.basis = nn.Parameter(torch.randn(num_groups, num_basis, self.channels_per_group))
        self.scale_proj = nn.Linear(num_basis, self.channels_per_group, bias=False)
        self.shift_proj = nn.Linear(num_basis, self.channels_per_group, bias=False)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 그룹별로 처리
        x_groups = x.reshape(B, self.num_groups, self.channels_per_group, H, W)
        
        alphas = []
        betas = []
        
        for g in range(self.num_groups):
            ctx_g = x_groups[:, g].mean(dim=[2, 3])  # [B, C/G]
            coeff_g = ctx_g @ self.basis[g].t()  # [B, K]
            coeff_g = self.coeff_dropout(coeff_g)
            
            alpha_g = self.scale_proj(coeff_g).sigmoid()
            beta_g = self.shift_proj(coeff_g)
            
            alphas.append(alpha_g)
            betas.append(beta_g)
        
        alpha = torch.stack(alphas, dim=1).reshape(B, C, 1, 1)
        beta = torch.stack(betas, dim=1).reshape(B, C, 1, 1)
        
        return x * alpha + beta


# ============================================================================
# 최적화 버전 3: No-Split Block (메모리 복사 제거)
# ============================================================================
class NoSplitAffinePRCM(nn.Module):
    """
    최적화: Channel Split 없이 전체 채널에 대해 선택적으로 처리
    Split/Concat 오버헤드 제거
    """
    def __init__(self, channels, num_basis=8, dropout_rate=0.5, selection_ratio=0.5):
        super().__init__()
        self.num_basis = num_basis
        self.channels = channels
        self.num_selected = int(channels * selection_ratio)
        
        # 학습 가능한 채널 선택 마스크
        self.channel_importance = nn.Parameter(torch.randn(channels))
        
        self.basis = nn.Parameter(torch.randn(num_basis, self.num_selected))
        self.affine_proj = nn.Linear(num_basis, self.num_selected * 2, bias=False)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Top-K 채널 선택 (backward 가능)
        _, top_indices = torch.topk(self.channel_importance, self.num_selected, dim=0)
        top_indices_sorted, _ = torch.sort(top_indices)
        
        # 선택된 채널만 처리
        x_selected = x[:, top_indices_sorted]  # [B, K_selected, H, W]
        
        ctx = x_selected.mean(dim=[2, 3])
        coeff = ctx @ self.basis.t()
        coeff = self.coeff_dropout(coeff)
        
        affine_params = self.affine_proj(coeff)
        alpha, beta = affine_params.chunk(2, dim=1)
        
        alpha = alpha.sigmoid().unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        x_modulated = x_selected * alpha + beta
        
        # 원래 위치에 다시 배치
        out = x.clone()
        out[:, top_indices_sorted] = x_modulated
        
        return out


# ============================================================================
# 최적화 버전 4: Depthwise Affine (가장 경량)
# ============================================================================
class DepthwiseAffinePRCM(nn.Module):
    """
    최적화: 각 채널이 독립적인 Scale/Shift 학습
    Basis projection 없이 직접 학습 (가장 빠름)
    """
    def __init__(self, channels, dropout_rate=0.5):
        super().__init__()
        self.channels = channels
        
        # 채널별 독립적인 scale/shift
        self.scale = nn.Parameter(torch.ones(channels))
        self.shift = nn.Parameter(torch.zeros(channels))
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Global context
        ctx = x.mean(dim=[2, 3])  # [B, C]
        
        # Context-aware scale/shift
        alpha = (self.scale * ctx).sigmoid().unsqueeze(-1).unsqueeze(-1)
        beta = (self.shift * ctx).unsqueeze(-1).unsqueeze(-1)
        
        x = x * alpha + beta
        return self.dropout(x)


# ============================================================================
# 최적화 버전 5: 원본 유지 + Affine만 적용
# ============================================================================
class SimpleAffinePRCM(nn.Module):
    """
    최적화: Split 없이 기존 구조 유지, Affine만 추가
    가장 간단한 개선
    """
    def __init__(self, channels, num_basis=8, dropout_rate=0.5):
        super().__init__()
        self.num_basis = num_basis
        
        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        self.affine_proj = nn.Linear(num_basis, channels * 2, bias=False)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        ctx = x.mean(dim=[2, 3])
        coeff = ctx @ self.basis.t()
        coeff = self.coeff_dropout(coeff)
        
        affine_params = self.affine_proj(coeff)
        alpha, beta = affine_params.chunk(2, dim=1)
        
        alpha = alpha.sigmoid().unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        return x * alpha + beta


# ============================================================================
# 최적화된 블록들
# ============================================================================
def make_optimized_block(in_ch, out_ch, prcm_type='fused', **kwargs):
    """
    prcm_type:
        - 'fused': FusedAffinePRCM (Linear 1개)
        - 'lightweight': LightweightAffinePRCM (그룹별 처리)
        - 'depthwise': DepthwiseAffinePRCM (가장 빠름)
        - 'simple': SimpleAffinePRCM (기본)
    """
    layers = []
    
    if in_ch != out_ch:
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))
    
    layers.append(RepConv(out_ch, out_ch, kernel_size=7, padding=3, groups=out_ch))
    
    if prcm_type == 'fused':
        layers.append(FusedAffinePRCM(out_ch, **kwargs))
    elif prcm_type == 'lightweight':
        layers.append(LightweightAffinePRCM(out_ch, **kwargs))
    elif prcm_type == 'depthwise':
        layers.append(DepthwiseAffinePRCM(out_ch, **kwargs))
    elif prcm_type == 'simple':
        layers.append(SimpleAffinePRCM(out_ch, **kwargs))
    else:
        raise ValueError(f"Unknown prcm_type: {prcm_type}")
    
    return nn.Sequential(*layers)


# ============================================================================
# 최적화된 메인 모델
# ============================================================================
class JeongWonNet_Optimized_simple(nn.Module):
    """
    Latency 최적화 버전
    - Split/Concat 제거
    - Fused Affine operation
    - 선택 가능한 PRCM 타입
    """
    def __init__(self, 
                 num_classes=1, 
                 input_channels=3, 
                 c_list=[24, 48, 64, 96, 128, 192],
                 prcm_type='simple',  # 'fused', 'lightweight', 'depthwise', 'simple'
                 num_basis=8,
                 dropout_rate=0.5,
                 gt_ds=True):
        super().__init__()
        self.gt_ds = gt_ds
        
        prcm_kwargs = {
            'num_basis': num_basis,
            'dropout_rate': dropout_rate
        }
        
        if prcm_type == 'lightweight':
            prcm_kwargs['num_groups'] = 4
        elif prcm_type == 'depthwise':
            prcm_kwargs = {'dropout_rate': dropout_rate}
        
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_list[0]),
            nn.ReLU(inplace=True)
        )
        
        self.encoder1 = make_optimized_block(c_list[0], c_list[0], prcm_type, **prcm_kwargs)
        self.encoder2 = make_optimized_block(c_list[0], c_list[1], prcm_type, **prcm_kwargs)
        self.encoder3 = make_optimized_block(c_list[1], c_list[2], prcm_type, **prcm_kwargs)
        self.encoder4 = make_optimized_block(c_list[2], c_list[3], prcm_type, **prcm_kwargs)
        self.encoder5 = make_optimized_block(c_list[3], c_list[4], prcm_type, **prcm_kwargs)
        self.encoder6 = make_optimized_block(c_list[4], c_list[5], prcm_type, **prcm_kwargs)
        
        if gt_ds:
            self.gt_conv1 = nn.Conv2d(c_list[4], num_classes, 1)
            self.gt_conv2 = nn.Conv2d(c_list[3], num_classes, 1)
            self.gt_conv3 = nn.Conv2d(c_list[2], num_classes, 1)
            self.gt_conv4 = nn.Conv2d(c_list[1], num_classes, 1)
            self.gt_conv5 = nn.Conv2d(c_list[0], num_classes, 1)
        
        self.decoder1 = make_optimized_block(c_list[5], c_list[4], prcm_type, **prcm_kwargs)
        self.decoder2 = make_optimized_block(c_list[4], c_list[3], prcm_type, **prcm_kwargs)
        self.decoder3 = make_optimized_block(c_list[3], c_list[2], prcm_type, **prcm_kwargs)
        self.decoder4 = make_optimized_block(c_list[2], c_list[1], prcm_type, **prcm_kwargs)
        self.decoder5 = make_optimized_block(c_list[1], c_list[0], prcm_type, **prcm_kwargs)
        
        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)
        
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
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        is_eval = not self.training
        
        x = self.stem(x)
        
        e1 = F.max_pool2d(self.encoder1(x), 2)
        e2 = F.max_pool2d(self.encoder2(e1), 2)
        e3 = F.max_pool2d(self.encoder3(e2), 2)
        e4 = F.max_pool2d(self.encoder4(e3), 2)
        e5 = F.max_pool2d(self.encoder5(e4), 2)
        e6 = self.encoder6(e5)
        
        d5 = self.decoder1(e6) + e5
        d4 = F.interpolate(self.decoder2(d5), scale_factor=2, mode='bilinear', align_corners=True) + e4
        d3 = F.interpolate(self.decoder3(d4), scale_factor=2, mode='bilinear', align_corners=True) + e3
        d2 = F.interpolate(self.decoder4(d3), scale_factor=2, mode='bilinear', align_corners=True) + e2
        d1 = F.interpolate(self.decoder5(d2), scale_factor=2, mode='bilinear', align_corners=True) + e1
        
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


if __name__ == "__main__":
    print("=== Latency 최적화 버전 테스트 ===\n")
    
    x = torch.randn(2, 3, 256, 256)
    
    models = {
        'fused': JeongWonNet_Optimized(prcm_type='fused'),
        'lightweight': JeongWonNet_Optimized(prcm_type='lightweight'),
        'depthwise': JeongWonNet_Optimized(prcm_type='depthwise'),
        'simple': JeongWonNet_Optimized(prcm_type='simple'),
    }
    
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            out = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:12s}: Output {out.shape}, Params: {params:,}")
