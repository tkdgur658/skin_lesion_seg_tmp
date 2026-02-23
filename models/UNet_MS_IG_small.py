# ablation_unet_ms_ig.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utils
# -----------------------------
def group_norm(num_channels, eps=1e-5):
    g = num_channels // 4 if num_channels >= 4 else 1
    return nn.GroupNorm(num_groups=g, num_channels=num_channels, eps=eps, affine=True)

def to_gray(x):
    return x if x.size(1) == 1 else x.mean(dim=1, keepdim=True)

# -----------------------------
# Input-Guided Gate (파라미터 0)
# mode: 'both' | 'intensity' | 'grad'
# op:   'mul'  | 'res'         (res: feat + feat*gate)
# -----------------------------
class InputGuidedGate(nn.Module):
    def __init__(self, mode='both', op='mul', eps=1e-6):
        super().__init__()
        self.mode = mode
        self.op = op
        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        sobel_y = sobel_x.transpose(-1,-2).contiguous()
        self.register_buffer('kx', sobel_x)
        self.register_buffer('ky', sobel_y)
        self.eps = eps

    def _gate_from_ref(self, ref):
        with torch.no_grad():
            # ref:[B,1,H,W]
            comp = []
            if self.mode in ('intensity', 'both'):
                m = ref.mean(dim=[2,3], keepdim=True)
                v = ref.var(dim=[2,3], keepdim=True)
                i_norm = (ref - m) / (v.add(self.eps).sqrt() + self.eps)
                comp.append(i_norm)
            if self.mode in ('grad', 'both'):
                gx = F.conv2d(ref, self.kx, padding=1)
                gy = F.conv2d(ref, self.ky, padding=1)
                comp.append(gx.abs() + gy.abs())
            s = comp[0] if len(comp) == 1 else sum(comp)
            gate = torch.sigmoid(s)  # [B,1,H,W], 0~1
        return gate

    def forward(self, ref, feat):
        gate = self._gate_from_ref(ref)
        if self.op == 'mul':
            return feat * gate
        elif self.op == 'res':
            return feat + feat * gate
        else:
            raise ValueError(f"Unknown gate op: {self.op}")

# -----------------------------
# Multi-Scale Large-Kernel Bottleneck (MS-LK)
# depthwise dilated branches (경량 전역 문맥)
# -----------------------------
class MultiScaleLargeKernelBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_sizes=(3,7,11)):
        super().__init__()
        mid = out_ch
        self.pre = nn.Sequential(
            nn.Conv2d(in_ch, mid * len(kernel_sizes), 1, bias=False),
            group_norm(mid * len(kernel_sizes)),
            nn.GELU(approximate='tanh')
        )
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            d = (k - 1) // 2
            self.branches.append(nn.Sequential(
                nn.Conv2d(mid, mid, 3, padding=d, dilation=d, groups=mid, bias=False),
                group_norm(mid),
                nn.GELU(approximate='tanh')
            ))
        self.post = nn.Sequential(
            nn.Conv2d(mid * len(kernel_sizes), out_ch, 1, bias=False),
            group_norm(out_ch)
        )
        self.down = None
        if in_ch != out_ch:
            self.down = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), group_norm(out_ch))
        self.act = nn.GELU(approximate='tanh')

    def forward(self, x):
        idt = x
        x = self.pre(x)
        chunks = torch.chunk(x, len(self.branches), dim=1)
        ys = [b(c) for b, c in zip(self.branches, chunks)]
        y = self.post(torch.cat(ys, dim=1))
        if self.down is not None:
            idt = self.down(idt)
        return self.act(y + idt)

# -----------------------------
# Plain Bottleneck (Baseline)
# -----------------------------
class PlainBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = out_ch
        self.conv1 = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.gn1 = group_norm(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
        self.gn2 = group_norm(mid)
        self.conv3 = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.gn3 = group_norm(out_ch)
        self.down = None
        if in_ch != out_ch:
            self.down = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), group_norm(out_ch))
        self.act = nn.GELU(approximate='tanh')

    def forward(self, x):
        idt = x
        x = self.act(self.gn1(self.conv1(x)))
        x = self.act(self.gn2(self.conv2(x)))
        x = self.gn3(self.conv3(x))
        if self.down is not None:
            idt = self.down(idt)
        return self.act(x + idt)

# -----------------------------
# Stage builder: MS-LK on/off
# -----------------------------
def make_stage(in_ch, out_ch, num_blocks=2, use_ms=True, kernel_sizes=(3,7,11)):
    layers = []
    for i in range(num_blocks):
        ic = in_ch if i == 0 else out_ch
        if use_ms:
            layers.append(MultiScaleLargeKernelBottleneck(ic, out_ch, kernel_sizes=kernel_sizes))
        else:
            layers.append(PlainBottleneck(ic, out_ch))
    return nn.Sequential(*layers)

# -----------------------------
# UNet Ablation Model
# -----------------------------
__all__ = ['UNet_MS_IG_small']
class UNet_MS_IG_small(nn.Module):
    """
    핵심 아이디어 ablation 및 스케일링이 가능한 모델:
     - init_features: 모델의 전체적인 너비(채널 수)를 조절
     - depths: 각 스테이지의 깊이(블록 수)를 조절

     Ablation용 파라미터:
     - use_ms: 멀티스케일 대형커널 병목 on/off
     - use_gate: 입력-유도 게이팅 on/off
     - gate_mode: 'both'|'intensity'|'grad'
     - gate_stages: (s1,s2,s3,s4) True/False
    """
    def __init__(self,
                 in_channels=3, out_channels=1,
                 # --- 모델 스케일링 파라미터 ---
                 init_features=28, depths=(1, 2, 2, 1),
                 # --- Ablation용 파라미터 ---
                 use_ms=True,
                 ms_kernel_sizes=(3,7,11),
                 use_gate=True,
                 gate_mode='both', gate_op='mul',
                 gate_stages=(True, True, True, True)
                 ):
        super().__init__()

        if not isinstance(depths, (list, tuple)) or len(depths) != 4:
            raise ValueError(f"depths는 4개의 정수를 담은 리스트/튜플이어야 합니다. (현재: {depths})")

        f = init_features
        self.use_gate = use_gate
        self.gate_stages = gate_stages
        if use_gate:
            self.gate = InputGuidedGate(mode=gate_mode, op=gate_op)

        # Encoder
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, f, 3, padding=1, bias=False),
            group_norm(f),
            nn.GELU(approximate='tanh')
        )
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)

        # 깊이 파라미터를 사용하여 스테이지 생성
        self.enc1 = make_stage(f,   f,   num_blocks=depths[0], use_ms=use_ms, kernel_sizes=ms_kernel_sizes)
        self.enc2 = make_stage(f,   2*f, num_blocks=depths[1], use_ms=use_ms, kernel_sizes=ms_kernel_sizes)
        self.enc3 = make_stage(2*f, 4*f, num_blocks=depths[2], use_ms=use_ms, kernel_sizes=ms_kernel_sizes)
        self.enc4 = make_stage(4*f, 8*f, num_blocks=depths[3], use_ms=use_ms, kernel_sizes=ms_kernel_sizes)

        # Decoder (baseline 유지, 인코더와 대칭적인 깊이 적용)
        self.up3 = nn.ConvTranspose2d(8*f, 4*f, 2, stride=2)
        self.dec3 = make_stage(8*f, 4*f, num_blocks=depths[2], use_ms=False) # 디코더는 Plain Bottleneck 고정
        self.up2 = nn.ConvTranspose2d(4*f, 2*f, 2, stride=2)
        self.dec2 = make_stage(4*f, 2*f, num_blocks=depths[1], use_ms=False)
        self.up1 = nn.ConvTranspose2d(2*f, f, 2, stride=2)
        self.dec1 = make_stage(2*f, f, num_blocks=depths[0], use_ms=False)

        self.head = nn.Conv2d(f, out_channels, 1)

    def _apply_gate(self, x_ref, feat, flag):
        if self.use_gate and flag:
            return self.gate(x_ref, feat)
        return feat

    def forward(self, x):
        # ... (forward 메소드는 기존과 동일) ...
        # 입력 참조(그레이 + 다운샘플)
        gx  = to_gray(x)
        gx2 = F.avg_pool2d(gx, 2)
        gx4 = F.avg_pool2d(gx, 4)
        gx8 = F.avg_pool2d(gx, 8)

        # Encoder (+ optional gate)
        x0 = self.stem(x)
        e1 = self.enc1(x0)
        e1 = self._apply_gate(gx, e1, self.gate_stages[0])

        e2 = self.enc2(self.pool1(e1))
        e2 = self._apply_gate(gx2, e2, self.gate_stages[1])

        e3 = self.enc3(self.pool2(e2))
        e3 = self._apply_gate(gx4, e3, self.gate_stages[2])

        e4 = self.enc4(self.pool3(e3))
        e4 = self._apply_gate(gx8, e4, self.gate_stages[3])

        # Decoder (baseline)
        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)
