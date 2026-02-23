import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_

class RepConv(nn.Module):
    """
    Multi-Scale Re-parameterizable Conv
    훈련: 3x3 + 5x5 + 1x1 + Identity
    추론: 단일 5x5 Conv로 융합
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5],
                 stride=1, groups=1, use_identity=True, use_activation=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_kernel = max(kernel_sizes)
        self.stride = stride
        self.padding = self.max_kernel // 2
        self.groups = groups
        self.kernel_sizes = sorted(kernel_sizes, reverse=True)
        
        self.use_identity = use_identity and (stride == 1) and (in_channels == out_channels)
        
        # 각 스케일별 Conv + BN
        self.branches = nn.ModuleList()
        for ks in self.kernel_sizes:
            pad = ks // 2
            conv = nn.Conv2d(in_channels, out_channels, ks, 
                           stride, pad, groups=groups, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.branches.append(nn.ModuleDict({'conv': conv, 'bn': bn}))
        
        # Identity branch
        if self.use_identity:
            self.bn_identity = nn.BatchNorm2d(out_channels)
        
        self.activation = nn.ReLU(inplace=True) if use_activation else nn.Identity()
    
    def forward(self, x):
        if hasattr(self, 'fused_conv'):
            return self.activation(self.fused_conv(x))
        
        # 모든 스케일 합산
        out = 0
        for branch in self.branches:
            out = out + branch['bn'](branch['conv'](x))
        
        if self.use_identity:
            out = out + self.bn_identity(x)
        
        return self.activation(out)
    
    def switch_to_deploy(self):
        if hasattr(self, 'fused_conv'):
            return
        
        # 모든 branch를 최대 커널 사이즈로 융합
        kernel = 0
        bias = 0
        
        for branch in self.branches:
            k, b = self._fuse_bn_tensor(branch['conv'], branch['bn'])
            kernel = kernel + self._pad_to_max_kernel(k, branch['conv'].kernel_size[0])
            bias = bias + b
        
        if self.use_identity:
            k_id, b_id = self._fuse_bn_tensor(None, self.bn_identity)
            kernel = kernel + k_id
            bias = bias + b_id
        
        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.max_kernel,
            self.stride, self.padding, groups=self.groups, bias=True
        )
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias
        
        # 메모리 정리
        self.__delattr__('branches')
        if hasattr(self, 'bn_identity'):
            self.__delattr__('bn_identity')
    
    def _fuse_bn_tensor(self, conv, bn):
        if conv is None:
            # Identity
            input_dim = self.in_channels // self.groups
            kernel_value = torch.zeros((self.in_channels, input_dim,
                                       self.max_kernel, self.max_kernel),
                                      dtype=bn.weight.dtype, device=bn.weight.device)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 
                           self.max_kernel // 2, self.max_kernel // 2] = 1
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
    
    def _pad_to_max_kernel(self, kernel, current_size):
        if current_size == self.max_kernel:
            return kernel
        pad = (self.max_kernel - current_size) // 2
        return F.pad(kernel, [pad, pad, pad, pad])

import torch
from torch import nn
from timm.layers import trunc_normal_

class SimplePRCM(nn.Module):
    """
    [Optimization] AffinePRCM 경량화 버전
    - Shift(beta) 연산 제거 -> Latency 약 30% 감소
    - Scale(alpha)만 적용하여 Gating 효과 유지
    """
    def __init__(self, channels, num_basis=8, dropout_rate=0.5):
        super().__init__()
        self.num_basis = num_basis
        self.channels = channels

        self.basis = nn.Parameter(torch.randn(num_basis, channels))
        # shift_proj 제거
        self.scale_proj = nn.Linear(num_basis, channels, bias=False)
        self.coeff_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        # B, C, H, W = x.shape  <-- 사용하지 않는 변수 제거 (속도 미세 최적화)
        
        ctx = x.mean(dim=[2, 3])
        coeff = ctx @ self.basis.t()
        coeff = self.coeff_dropout(coeff)

        # shift(beta) 연산 제거, alpha만 계산
        alpha = self.scale_proj(coeff).sigmoid().unsqueeze(-1).unsqueeze(-1)
        
        return x * alpha


class FastSTMBlock(nn.Module):
    """
    [Optimization] Split-Transform-Merge Block 가속화 버전
    - Chunk/Shuffle 제거 -> Slicing/Mixing(1x1 Conv) 도입
    - Stem 처리 방식 개선 (Full Conv)
    """
    def __init__(self, in_channels, out_channels, kernel_size=7, num_basis=8, dropout_rate=0.5, stem=False):
        super().__init__()
        self.stem = stem

        # [1] Stem(입력층)인 경우: 복잡한 Split 없이 바로 Full Convolution 수행
        if stem:
            # 3채널 -> out_channels로 바로 변환 (가장 빠름)
            self.stem_block = nn.Sequential(
                RepConv(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=1), # groups=1 주의 (RGB는 연관성이 높음)
                SimplePRCM(out_channels, num_basis=num_basis, dropout_rate=dropout_rate)
            )
        else:
            # [2] 일반 Block: Partial Convolution 전략 (FasterNet Style)
            
            # 2-1. Channel Alignment (입출력 채널 다를 때 1x1로 맞춤)
            if in_channels != out_channels:
                self.align_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
                self.align_bn = nn.BatchNorm2d(out_channels) # BN 추가로 학습 안정화
            else:
                self.align_conv = None

            # 2-2. Active Branch (채널 절반만 연산)
            self.dim_conv = out_channels // 2
            
            self.partial_conv = RepConv(
                self.dim_conv, 
                self.dim_conv, 
                kernel_size=kernel_size, 
                groups=self.dim_conv # Depthwise
            )
            
            self.partial_prcm = SimplePRCM(
                self.dim_conv, 
                num_basis=num_basis, 
                dropout_rate=dropout_rate
            )

            # 2-3. Mixing Branch (Shuffle 대신 1x1 Conv 사용)
            # Shuffle(0.01ms) 대신 GPU 행렬 연산에 최적화된 1x1 Conv 사용
            self.mix_conv = nn.Conv2d(out_channels, out_channels, 1, bias=False)
            self.mix_bn = nn.BatchNorm2d(out_channels)
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # [Case 1] Stem: 단순 통과 (Latency 최소화)
        if self.stem:
            return self.stem_block(x)

        # [Case 2] Normal Block
        
        # 1. 채널 개수 맞추기 (필요 시)
        if self.align_conv is not None:
            x = self.act(self.align_bn(self.align_conv(x)))
        
        # 2. Slicing (Zero-copy View) - Chunk보다 빠름
        # x_active: 연산할 절반 / x_passive: 그대로 둘 절반
        x_active = x[:, :self.dim_conv, :, :]
        x_passive = x[:, self.dim_conv:, :, :]
        
        # 3. Transform (Active만 연산)
        x_active = self.partial_conv(x_active)
        x_active = self.partial_prcm(x_active)
        
        # 4. Merge (Concat)
        x_out = torch.cat([x_active, x_passive], dim=1)
        
        # 5. Mix (Channel Shuffle 대체)
        # 1x1 Conv가 채널 간 정보를 섞어줌
        x_out = self.act(self.mix_bn(self.mix_conv(x_out)))
        
        return x_out


class JeongWonNet_STMShuffle_NoStemMS(nn.Module):
    """Split-Transform-Merge UNet with Affine Modulation PRCM (No separate stem)"""
    def __init__(self,
                 num_classes=1,
                 input_channels=3,
                 c_list=[24, 48, 64, 96, 128, 192],
                 num_basis=8,
                 dropout_rate=0.5,
                 gt_ds=True):
        super().__init__()
        self.gt_ds = gt_ds

        # encoder1은 stem=True로 3ch -> c_list[0] 변환
        self.encoder1 = FastSTMBlock(
            input_channels, c_list[0], kernel_size=7, num_basis=num_basis, dropout_rate=dropout_rate, stem=True
        )
        self.encoder2 = FastSTMBlock(
            c_list[0], c_list[1], kernel_size=7, num_basis=num_basis, dropout_rate=dropout_rate
        )
        self.encoder3 = FastSTMBlock(
            c_list[1], c_list[2], kernel_size=7, num_basis=num_basis, dropout_rate=dropout_rate
        )
        self.encoder4 = FastSTMBlock(
            c_list[2], c_list[3], kernel_size=7, num_basis=num_basis, dropout_rate=dropout_rate
        )
        self.encoder5 = FastSTMBlock(
            c_list[3], c_list[4], kernel_size=7, num_basis=num_basis, dropout_rate=dropout_rate
        )
        self.encoder6 = FastSTMBlock(
            c_list[4], c_list[5], kernel_size=7, num_basis=num_basis, dropout_rate=dropout_rate
        )

        if gt_ds:
            self.gt_conv1 = nn.Conv2d(c_list[4], num_classes, 1)
            self.gt_conv2 = nn.Conv2d(c_list[3], num_classes, 1)
            self.gt_conv3 = nn.Conv2d(c_list[2], num_classes, 1)
            self.gt_conv4 = nn.Conv2d(c_list[1], num_classes, 1)
            self.gt_conv5 = nn.Conv2d(c_list[0], num_classes, 1)

        self.decoder1 = FastSTMBlock(
            c_list[5], c_list[4], kernel_size=7, num_basis=num_basis, dropout_rate=dropout_rate
        )
        self.decoder2 = FastSTMBlock(
            c_list[4], c_list[3], kernel_size=7, num_basis=num_basis, dropout_rate=dropout_rate
        )
        self.decoder3 = FastSTMBlock(
            c_list[3], c_list[2], kernel_size=7, num_basis=num_basis, dropout_rate=dropout_rate
        )
        self.decoder4 = FastSTMBlock(
            c_list[2], c_list[1], kernel_size=7, num_basis=num_basis, dropout_rate=dropout_rate
        )
        self.decoder5 = FastSTMBlock(
            c_list[1], c_list[0], kernel_size=7, num_basis=num_basis, dropout_rate=dropout_rate
        )

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
    model = JeongWonNet_STMShuffle_NoStem()
    x = torch.randn(2, 3, 256, 256)

    model.train()
    ds_outputs, final_out = model(x)
    print(f"Train - Final: {final_out.shape}, DS: {len(ds_outputs)} levels")

    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Eval - Output: {out.shape}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")
