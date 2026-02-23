# JeongWonNet Context Bridge Variants

## Overview

PRCM의 Global Average Pooled context를 Encoder→Decoder 간 연결하여 멀티스케일 정보 활용

## Best Model: JeongWonNet_CtxBridge

```
Encoder에서 추출한 ctx를 대응하는 Decoder의 PRCM에 전달

enc1_ctx(24ch)  ────────────────────────────→  dec5
enc2_ctx(48ch)  ──────────────────────→  dec4
enc3_ctx(64ch)  ────────────────→  dec3
enc4_ctx(96ch)  ──────────→  dec2
enc5_ctx(128ch) ────→  dec1
```

### PRCM_Bridge 구조

```
ctx_self = GAP(x)                      # Decoder 자신의 context
ctx_enc_proj = Dropout(Linear(ctx_enc)) # Encoder context projection
ctx_fused = ctx_self + ctx_enc_proj    # Add fusion

coeff = ctx_fused @ basis.T
w = sigmoid(fuser(dropout(coeff)))
output = x * w
```

### 특징
- **Add Fusion**: 단순 덧셈으로 정보 결합
- **Linear Projection**: enc_channels → dec_channels 변환
- **Bridge Dropout (0.2)**: encoder context 의존도 조절, 과적합 방지

---

## Variant Comparison

| Model | Fusion | Projection | 결과 |
|-------|--------|------------|------|
| **CtxBridge** | Add | Linear | **Best** |
| CtxBridge_Concat | Concat→Linear | Linear | - |
| CtxBridge_NonLinear | Add | Linear→LN→GELU | - |
| CtxBridge_ConcatNL | Concat | Linear→LN→GELU | - |

### 왜 CtxBridge가 가장 좋은가?

1. **단순함**: Add fusion은 gradient flow가 깔끔함
2. **적절한 정규화**: Dropout만으로 충분한 regularization
3. **파라미터 효율**: Concat이나 NonLinear보다 적은 파라미터
4. **과적합 방지**: 복잡한 fusion은 오히려 과적합 유발

---

## Architecture

```
Input (3, 256, 256)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Encoder                                                │
├─────────────────────────────────────────────────────────┤
│  encoder1: DWBlock(3→24)   → e1, ctx1 → MaxPool → 128  │
│  encoder2: DWBlock(24→48)  → e2, ctx2 → MaxPool → 64   │
│  encoder3: DWBlock(48→64)  → e3, ctx3 → MaxPool → 32   │
│  encoder4: DWBlock(64→96)  → e4, ctx4 → MaxPool → 16   │
│  encoder5: DWBlock(96→128) → e5, ctx5 → MaxPool → 8    │
│  encoder6: DWBlock(128→192) → e6                → 8    │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Decoder with Context Bridge                            │
├─────────────────────────────────────────────────────────┤
│  decoder1: DWBlock_Bridge(192→128, ctx5) + e5    → 8   │
│  decoder2: DWBlock_Bridge(128→96, ctx4)  + e4    → 16  │
│  decoder3: DWBlock_Bridge(96→64, ctx3)   + e3    → 32  │
│  decoder4: DWBlock_Bridge(64→48, ctx2)   + e2    → 64  │
│  decoder5: DWBlock_Bridge(48→24, ctx1)   + e1    → 128 │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Final Conv (24→1) + Upsample → Output (1, 256, 256)
```

---

## DWBlock Structure

```
DWBlock (Encoder):
    [1x1 Conv] (if in_ch != out_ch)
        ↓
    RepConv 7x7 DW (depthwise)
        ↓
    PRCM (return_ctx=True)
        ↓
    output, ctx

DWBlock_Bridge (Decoder):
    [1x1 Conv] (if in_ch != out_ch)
        ↓
    RepConv 7x7 DW (depthwise)
        ↓
    PRCM_Bridge (ctx_enc 입력)
        ↓
    output
```

---

## Hyperparameters

```python
JeongWonNet_CtxBridge(
    num_classes=1,
    input_channels=3,
    c_list=[24, 48, 64, 96, 128, 192],
    kernel_size=7,          # RepConv kernel size
    num_basis=8,            # PRCM low-rank basis 개수
    dropout_rate=0.5,       # PRCM coefficient dropout
    bridge_dropout=0.2,     # Context bridge dropout
    gt_ds=True              # Deep supervision
)
```

---

## Context Bridge 효과

### Skin Lesion Segmentation에서의 이점

1. **저해상도 ctx (enc5, enc4)**: 병변의 위치와 크기 정보
2. **고해상도 ctx (enc1, enc2)**: 경계와 텍스처 정보
3. **Decoder에서 직접 참조**: Skip connection 외에 global context도 활용

### Information Flow

```
Feature Map Skip Connection:  e1, e2, e3, e4, e5 → spatial detail
Context Bridge:               ctx1~ctx5 → global semantic info
```

---

## Usage

```python
from models.JeongWonNet_CtxBridge import JeongWonNet_CtxBridge

model = JeongWonNet_CtxBridge(
    num_classes=1,
    bridge_dropout=0.2
)

# Training
model.train()
ds_outputs, final_out = model(x)

# Inference
model.eval()
out = model(x)

# Deploy mode (RepConv fusion)
model.switch_to_deploy()
```
