# Loss Functions

## 개요

모든 loss는 Deep Supervision wrapper를 포함하며, `forward(gt_pre, out, target)` 인터페이스로 `GT_BceDiceLoss`와 호환된다.

```python
# main.ipynb Config에서 모델별 loss 지정
model_loss_map = {
    'ModelA': 'boundary_focal',
    'ModelB': 'lovasz',
    'ModelC': 'uncertainty',
}
# 매핑에 없는 모델 → 기존 GT_BceDiceLoss (디폴트)
```

---

## 1. GT_BceDiceLoss (디폴트)

**파일:** `utils.py`

가장 기본적인 segmentation loss 조합.

$$L = BCE + Dice$$

| 구성 | 설명 |
|---|---|
| BCE | Binary Cross-Entropy. 픽셀별 분류 확률 최적화 |
| Dice | Dice coefficient 기반. 영역 overlap 최적화 |

- 모든 픽셀을 동등하게 취급
- 경계/어려운 샘플에 대한 특별한 처리 없음

---

## 2. GT_BoundaryFocalLoss — `'boundary_focal'`

**파일:** `utils_loss.py` | **키워드:** 경계 정밀도, hard example mining

$$L = w_b \cdot BCE + w_d \cdot Dice + \lambda_{bd} \cdot BoundaryDice + \lambda_{ft} \cdot FocalTversky$$

### BoundaryDice

GT에서 boundary mask를 추출하여 **경계 영역에서만** Dice를 계산한다.

```
1) erosion: eroded = 1 - max_pool2d(1 - mask)
2) boundary = mask - eroded
3) boundary 영역에서 Dice 계산
```

- `max_pool2d`로 morphological erosion 구현 (GPU 친화적, 추가 라이브러리 불필요)
- `kernel_size`로 boundary 두께 조절 (기본값: 3)
- 소형 모델의 제한된 capacity를 경계에 집중시킴

### FocalTversky

Tversky Index에 Focal 변형을 적용하여 FN(미탐지)에 더 큰 페널티를 부여한다.

$$TI = \frac{TP + \epsilon}{TP + \alpha \cdot FP + \beta \cdot FN + \epsilon}$$

$$L_{FT} = (1 - TI)^{\gamma}$$

| 파라미터 | 기본값 | 역할 |
|---|---|---|
| `alpha` | 0.3 | FP 가중치 (낮을수록 FP에 관대) |
| `beta` | 0.7 | FN 가중치 (높을수록 미탐지 엄격) |
| `gamma` | 0.75 | Focal exponent (높을수록 어려운 샘플 집중) |

### 사용

```python
model_loss_map = {
    'JeongWonNet_CtxBridge_StdExp': 'boundary_focal',
}
```

### 적합한 경우

- 경계가 불분명한 병변 (피부 병변 등)
- 소형 모델에서 boundary quality 개선이 필요할 때
- FN을 줄여 recall을 높이고 싶을 때

---

## 3. GT_LovaszLoss — `'lovasz'`

**파일:** `utils_loss.py` | **키워드:** IoU 직접 최적화

$$L = w_b \cdot BCE + w_l \cdot Lov\acute{a}sz\text{-}Hinge$$

### Lovasz-Hinge

IoU(Jaccard Index)의 **convex surrogate**로, 평가 지표인 IoU를 직접 최적화한다.

- Lovasz extension을 통해 submodular set function (IoU)의 tight convex relaxation 제공
- BCE가 픽셀별 분류를 안정화하고, Lovasz가 전체 IoU를 끌어올림
- 추가 하이퍼파라미터 없이 간결한 구성

**Reference:** Berman et al., "The Lovasz-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks", CVPR 2018.

### 사용

```python
model_loss_map = {
    'EGEUNet': 'lovasz',
}
```

### 적합한 경우

- IoU 지표를 직접 올리고 싶을 때
- 하이퍼파라미터 튜닝 부담을 줄이고 싶을 때
- 경계보다는 전체 영역 정확도가 중요할 때

---

## 4. GT_UncertaintyLoss — `'uncertainty'`

**파일:** `utils_loss.py` | **키워드:** 자동 가중치 학습

$$L = \frac{1}{2\sigma_1^2} \cdot BCE + \frac{1}{2\sigma_2^2} \cdot Dice + \frac{1}{2\sigma_3^2} \cdot BoundaryDice + \frac{1}{2}\log(\sigma_1^2 \sigma_2^2 \sigma_3^2)$$

각 loss 항의 가중치를 수동으로 정하지 않고, **학습 가능한 파라미터** $\log \sigma^2$로 자동 조절한다.

| 학습 파라미터 | 대상 loss | 초기값 |
|---|---|---|
| `log_var_bce` | BCE | 0 (σ=1) |
| `log_var_dice` | Dice | 0 (σ=1) |
| `log_var_boundary` | BoundaryDice | 0 (σ=1) |

- σ가 커지면 해당 loss의 가중치가 줄어듦 (불확실한 항은 자동으로 덜 반영)
- σ가 작아지면 가중치가 커짐 (확실한 항에 더 집중)
- `log(σ)` regularization term이 σ가 무한히 커지는 것을 방지

**Reference:** Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics", CVPR 2018.

### 사용

```python
model_loss_map = {
    'MALUNet': 'uncertainty',
}
```

### 적합한 경우

- 여러 loss 간 가중치 튜닝이 어려울 때
- 학습 과정에서 loss 균형이 자동으로 잡히길 원할 때
- 실험적으로 최적 가중치를 탐색하는 대신 모델이 스스로 찾도록 할 때

### 주의사항

- σ 파라미터가 optimizer에 포함되어야 하므로, 현재 코드에서 `model.parameters()`만 optimizer에 넘기는 경우 σ의 gradient가 업데이트되지 않음
- `criterion.parameters()`도 optimizer에 포함해야 정상 동작 (확인 필요)

---

## Deep Supervision 구조

모든 GT-level loss는 동일한 DS 가중치를 사용한다:

```
L_total = L_main(out) + Σ(w_i · L(gt_pre_i))
```

| 출력 | 가중치 | 설명 |
|---|---|---|
| `out` (최종) | 1.0 | Main output |
| `gt_pre1` | 0.5 | 최종 출력에 가장 가까운 intermediate |
| `gt_pre2` | 0.4 | |
| `gt_pre3` | 0.3 | |
| `gt_pre4` | 0.2 | |
| `gt_pre5` | 0.1 | 가장 깊은 (초기) intermediate |

---

## 비교 요약

| Loss | 구성 | 경계 집중 | IoU 직접 최적화 | 자동 가중치 | 하이퍼파라미터 수 |
|---|---|---|---|---|---|
| GT_BceDiceLoss (디폴트) | BCE + Dice | X | X | X | 2 (wb, wd) |
| GT_BoundaryFocalLoss | BCE + Dice + BdDice + FocalTversky | O | X | X | 7 |
| GT_LovaszLoss | BCE + Lovasz-Hinge | X | O | X | 2 (wb, wl) |
| GT_UncertaintyLoss | BCE + Dice + BdDice | O | X | O | 1 (kernel) |
