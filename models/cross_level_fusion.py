"""
Cross-Level Fusion Modules

High-level과 Low-level 특징 융합을 위한 모듈들
"""
import torch
from torch import nn
import torch.nn.functional as F


class DiffAwareFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 두 피처 간의 차이(경계선 정보)에서 유의미한 패턴을 찾는 레이어
        self.diff_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, dec_feat, enc_feat):
        # 채널 수가 다르다면 맞춰주는 과정 선행 필요
        
        # High-level과 Low-level의 차이 계산 (Semantic 갭과 공간적 불일치 포착)
        diff = torch.abs(dec_feat - enc_feat)
        
        # 차이 정보를 기반으로 엣지를 강조하는 Attention Map 생성
        boundary_attn = self.diff_conv(diff)
        
        # 원본 피처들에 바운더리 정보를 부각시켜 융합
        fused = dec_feat + enc_feat * boundary_attn
        return self.out_conv(fused)


class SemanticModulationFusion(nn.Module):
    def __init__(self, dec_channels, enc_channels):
        super().__init__()
        # High-level 피처로 Low-level 피처를 컨트롤할 파라미터 생성
        self.modulator = nn.Sequential(
            nn.Conv2d(dec_channels, enc_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(enc_channels * 2)
        )
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(enc_channels, dec_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(dec_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, dec_feat, enc_feat):
        # 1. High-level에서 modulation 파라미터(gamma, beta) 추출
        # (해상도를 맞춰주는 interpolate 과정이 필요하다면 추가)
        mod_params = self.modulator(dec_feat)
        gamma, beta = torch.chunk(mod_params, 2, dim=1)
        
        # 2. Low-level 특징을 High-level 정보로 공간적 조절(Modulation)
        # 엣지 주변을 강화하거나 노이즈를 누르는 역할 수행
        enc_modulated = enc_feat * (1 + gamma) + beta
        
        # 3. 조절된 특징을 출력 채널에 맞게 변환 (Add 방식과 결합 가능)
        return self.out_conv(enc_modulated) + dec_feat


class AttentionFusion(nn.Module):
    def __init__(self, dec_channels, enc_channels):
        super().__init__()
        # gating_channels는 보통 dec_channels와 같음
        inter_channels = dec_channels // 2 

        self.theta = nn.Conv2d(enc_channels, inter_channels, 1, bias=False)
        self.phi = nn.Conv2d(dec_channels, inter_channels, 1, bias=False)
        self.psi = nn.Conv2d(inter_channels, 1, 1, bias=True)
        self.act = nn.Sigmoid()

        self.mix_conv = nn.Sequential(
            nn.Conv2d(dec_channels + enc_channels, dec_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, dec_feat, enc_feat):
        # 1. Attention Map 생성 (High-level을 가이드로 사용)
        theta_x = self.theta(enc_feat)
        phi_g = F.interpolate(self.phi(dec_feat), size=enc_feat.shape[2:], mode='bilinear')
        
        # 2. 픽셀별 중요도 계산 (Spatial Attention)
        f = F.relu(theta_x + phi_g, inplace=True)
        alpha = self.act(self.psi(f))
        
        # 3. Low-level 특징 필터링 (불필요한 배경 제거, 엣지 보존)
        enc_filtered = enc_feat * alpha
        
        # 4. 필터링된 특징과 High-level 융합
        # (원한다면 dec_feat 크기를 enc_feat에 맞추는 과정 추가 가능)
        fused = torch.cat([dec_feat, enc_filtered], dim=1)
        return self.mix_conv(fused)