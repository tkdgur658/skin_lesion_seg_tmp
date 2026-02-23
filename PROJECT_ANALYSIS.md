# Skin Lesion Segmentation 프로젝트 분석

## 1. 프로젝트 개요

이 프로젝트는 **피부 병변 세분화(Skin Lesion Segmentation)**를 위한 딥러닝 프레임워크입니다.
다양한 UNet 기반 모델들을 지원하며, ISIC 피부암 데이터셋에서 병변 영역을 자동으로 분할합니다.

---

## 2. 실행 코드

### 기본 실행
```bash
python train.py
```

### 실행 전 설정
`configs/config_setting.py` 파일에서 다음 항목들을 수정합니다:

| 설정 항목 | 설명 | 예시 |
|----------|------|------|
| `network` | 사용할 모델 선택 | `'ucmnet'`, `'egeunet'`, `'unet'` 등 |
| `datasets` | 데이터셋 선택 | `'isic17'`, `'isic18'`, `'PH2'` |
| `batch_size` | 배치 크기 | `8` |
| `epochs` | 학습 에폭 수 | `300` |
| `gpu_id` | 사용할 GPU | `'0'` |

---

## 3. 프로젝트 구조

```
Skin_Lesion_Segmentation/
├── train.py                 # 메인 학습 스크립트 (실행 코드)
├── engine.py                # 학습/검증/테스트 루프
├── utils.py                 # 유틸리티 함수들
├── count_params.py          # 모델 파라미터 수 계산
├── configs/
│   └── config_setting.py    # 학습 설정 파일
├── datasets/
│   └── dataset.py           # 데이터셋 로더
├── models/                  # 모델 아키텍처들
│   ├── __init__.py
│   ├── UNet.py
│   ├── egeunet.py
│   ├── UCMNet.py
│   ├── MALUNet.py
│   ├── AttU_Net.py
│   ├── CMUNeXt.py
│   ├── MHorUNet.py
│   ├── HF_UNet.py
│   ├── MAResUNet.py
│   ├── MHA_UNet.py
│   ├── AMNet.py
│   ├── TinyUNet.py
│   ├── UltraLight_VM_UNet.py
│   └── propose.py
└── data/                    # 데이터셋 폴더
    ├── isic2017/
    ├── isic2018/
    └── PH2/
```

---

## 4. 지원 모델 목록

| 모델명 | config 설정값 | 특징 |
|--------|--------------|------|
| UNet | `'unet'` | 기본 UNet |
| EGE-UNet | `'egeunet'` | Efficient Group Enhanced UNet (MICCAI 2023) |
| UCMNet | `'ucmnet'` | U-shaped Context Mining Network |
| MALUNet | `'malunet'` | Multi-Attention Lightweight UNet |
| Attention U-Net | `'AttU_Net'` | Attention Gate 적용 |
| CMUNeXt | `'CMUNeXt'` | ConvNeXt 기반 |
| MHorUNet | `'MHorUNet'` | Multi-Head Horizontal UNet |
| HF-UNet | `'HFUNet'` | High-Frequency UNet |
| MAResUNet | `'MAResUNet'` | Multi-Attention Residual UNet |
| MHA-UNet | `'MHA_UNet'` | Multi-Head Attention UNet |
| AMNet | `'amnet'` | Attention Module Network |
| TinyUNet | `'tinyunet'` | 경량화 UNet |
| UltraLight VM-UNet | `'UltraLight_VM_UNet'` | Mamba 기반 초경량 모델 |
| Propose | `'propose'` | 제안 모델 |

---

## 5. 데이터셋 구조

각 데이터셋은 10-fold 교차 검증을 위한 구조로 구성됩니다:

```
data/isic2017/
├── train_1/
│   ├── images/
│   └── masks/
├── val_1/
│   ├── images/
│   └── masks/
├── test_1/
│   ├── images/
│   └── masks/
├── train_2/
...
└── test_10/
```

---

## 6. 학습 파이프라인

### 6.1 전체 흐름
```
1. 설정 로드 (config_setting.py)
2. 10회 반복 실험 시작 (exp 1~10)
   ├── 데이터셋 준비 (train/val/test)
   ├── 모델 초기화
   ├── 학습 루프 (epochs)
   │   ├── train_one_epoch()
   │   ├── val_one_epoch()
   │   └── 최적 모델 저장
   └── test_one_epoch() (best 모델로 테스트)
```

### 6.2 손실 함수
- **GT_BceDiceLoss**: Binary Cross Entropy + Dice Loss 조합

### 6.3 최적화
- **Optimizer**: AdamW (기본값)
- **Scheduler**: CosineAnnealingLR (기본값)
- **Learning Rate**: 0.001

### 6.4 평가 지표
- **Dice Score (DSC)**: F1 Score와 동일
- **IoU (mIoU)**: Intersection over Union
- **Accuracy**: 전체 정확도
- **Sensitivity**: 재현율 (True Positive Rate)
- **Specificity**: 특이도 (True Negative Rate)

---

## 7. 출력 결과

학습 결과는 `results/` 폴더에 저장됩니다:

```
results/{network}_{dataset}_{timestamp}/
├── exp_1/
│   ├── log/              # 학습 로그
│   ├── checkpoints/      # 모델 가중치 (best-epoch{N}.pth)
│   ├── outputs/          # 시각화 결과
│   └── summary/          # TensorBoard 로그
├── exp_2/
...
└── exp_10/
```

---

## 8. 필수 의존성

```
python==3.8
pytorch==1.8.0 (CUDA 11.1)
torchvision==0.9.0
tensorboardX
ptflops
scikit-learn
numpy
Pillow
tqdm
```

### 선택적 의존성
```
mamba_ssm  # UltraLight_VM_UNet 사용 시 필요
```

---

## 9. 빠른 시작 가이드

### Step 1: 데이터 준비
데이터를 `./data/isic2017/` 또는 `./data/isic2018/`에 배치

### Step 2: 설정 수정
`configs/config_setting.py` 열어서 수정:
```python
network = 'ucmnet'     # 원하는 모델
datasets = 'isic17'    # 사용할 데이터셋
batch_size = 8
epochs = 300
gpu_id = '0'
```

### Step 3: 학습 실행
```bash
python train.py
```

### Step 4: 결과 확인
- 로그: `results/{실험폴더}/exp_N/log/`
- TensorBoard: `tensorboard --logdir results/{실험폴더}/exp_N/summary/`

---

## 10. 주요 특징

1. **10-Fold 교차 검증**: 자동으로 10회 반복 실험 수행
2. **Deep Supervision**: 일부 모델에서 다중 출력 학습 지원
3. **자동 모델 저장**: Validation Loss 기준 최적 모델 저장
4. **데이터 증강**: 회전, 뒤집기 등 다양한 증강 적용
5. **TensorBoard 지원**: 실시간 학습 모니터링

---

## 11. 참고 문헌

- EGE-UNet: MICCAI 2023 ("EGE-UNet: an Efficient Group Enhanced UNet for skin lesion segmentation")
