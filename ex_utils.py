"""
실험 유틸리티 함수 모음
main.ipynb에서 사용
"""
import os
import random
import inspect
import warnings
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime

# Warning 억제
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', category=FutureWarning, module='timm')
warnings.filterwarnings('ignore', message='.*upsample_bilinear2d_backward.*')
warnings.filterwarnings('ignore', message='.*upsample_bicubic2d_backward.*')

from datasets.dataset import NPY_datasets, Test_datasets
from tensorboardX import SummaryWriter
from engine import train_one_epoch, val_one_epoch, test_one_epoch
from utils import (
    get_logger, log_config_info, get_optimizer, get_scheduler,
    myNormalize, myToTensor, myRandomHorizontalFlip, myRandomVerticalFlip,
    myRandomRotation, myResizeKeepRatio, GT_BceDiceLoss
)
from utils_loss import GT_BCEOnlyLoss, GT_BceDiceSquaredLoss, GT_BoundaryFocalLoss, GT_LovaszLoss, GT_UncertaintyLoss  # noqa: E501
from models import *
import matplotlib.pyplot as plt
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def switch_to_deploy(model):
    """RepConv 등 reparameterizable 모듈을 추론 모드로 전환 (영구 변경)"""
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()


def visualize_predictions(model, test_loader, device, save_dir, dataset_name, model_name, seed, max_samples=10):
    """
    테스트 결과 시각화 및 저장

    Args:
        model: 평가할 모델
        test_loader: 테스트 데이터로더
        device: 디바이스
        save_dir: 시각화 저장 디렉토리
        dataset_name: 데이터셋 이름
        model_name: 모델 이름
        seed: 시드 번호
        max_samples: 저장할 최대 샘플 수
    """
    model.eval()

    # 시각화 폴더 생성
    vis_folder = os.path.join(save_dir, 'visualizations', f'{model_name}_{dataset_name}_seed{seed}')
    os.makedirs(vis_folder, exist_ok=True)

    sample_idx = 0

    with torch.no_grad():
        for data in test_loader:
            if sample_idx >= max_samples:
                break

            img, msk = data
            img = img.to(device).float()
            outputs = model(img)

            if isinstance(outputs, dict):
                outputs = outputs['out']
            while isinstance(outputs, tuple):
                outputs = outputs[0]

            # 모든 모델은 raw logits 출력 -> sigmoid 적용
            pred = torch.sigmoid(outputs)
            pred_binary = (pred > 0.5).float()

            batch_size = img.size(0)
            for i in range(batch_size):
                if sample_idx >= max_samples:
                    break

                # 원본 이미지 (denormalize 필요)
                img_np = img[i].cpu().numpy().transpose(1, 2, 0)
                # 간단한 denormalize (min-max)
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

                gt_np = msk[i].cpu().numpy().squeeze()
                pred_np = pred[i].cpu().numpy().squeeze()
                pred_bin_np = pred_binary[i].cpu().numpy().squeeze()

                # IoU 계산
                intersection = np.sum(pred_bin_np * gt_np)
                union = np.sum(pred_bin_np) + np.sum(gt_np) - intersection
                iou = intersection / (union + 1e-6)

                # 오버레이 생성 (TP: 초록, FP: 빨강, FN: 파랑)
                overlay = img_np.copy()
                tp_mask = (pred_bin_np > 0.5) & (gt_np > 0.5)
                fp_mask = (pred_bin_np > 0.5) & (gt_np <= 0.5)
                fn_mask = (pred_bin_np <= 0.5) & (gt_np > 0.5)

                alpha = 0.4
                overlay[tp_mask] = overlay[tp_mask] * (1 - alpha) + np.array([0, 1, 0]) * alpha  # 초록
                overlay[fp_mask] = overlay[fp_mask] * (1 - alpha) + np.array([1, 0, 0]) * alpha  # 빨강
                overlay[fn_mask] = overlay[fn_mask] * (1 - alpha) + np.array([0, 0, 1]) * alpha  # 파랑

                # 시각화
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))

                axes[0].imshow(img_np)
                axes[0].set_title('Input Image')
                axes[0].axis('off')

                axes[1].imshow(gt_np, cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')

                axes[2].imshow(overlay)
                axes[2].set_title('Overlay (G:TP, R:FP, B:FN)')
                axes[2].axis('off')

                axes[3].imshow(pred_bin_np, cmap='gray')
                axes[3].set_title(f'Prediction (IoU: {iou:.4f})')
                axes[3].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(vis_folder, f'sample_{sample_idx:03d}_iou{iou:.3f}.png'),
                           dpi=150, bbox_inches='tight')
                plt.close()

                sample_idx += 1

    print(f"  Saved {sample_idx} visualizations to: {vis_folder}")

# cuBLAS 결정론적 설정
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# 모든 모델은 raw logits 출력, sigmoid는 학습 코드에서 적용


def set_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available()==True:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
    except:
        if torch.cuda.is_available()==True:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        pass




def seed_worker(worker_id):
    """DataLoader worker 시드 고정"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_transformers(dataset_name, config):
    """데이터셋별 transformer 생성 (비율 유지 리사이즈)"""
    train_transformer = transforms.Compose([
        myNormalize(dataset_name, train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResizeKeepRatio(config.input_size_h, config.input_size_w)
    ])

    test_transformer = transforms.Compose([
        myNormalize(dataset_name, train=False),
        myToTensor(),
        myResizeKeepRatio(config.input_size_h, config.input_size_w)
    ])

    return train_transformer, test_transformer


def compute_dice(pred, target, smooth=1e-6):
    """Dice coefficient 계산"""
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def compute_iou(pred, target, smooth=1e-6):
    """IoU 계산"""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def evaluate_metrics(model, dataloader, device, config=None):
    """모델 평가 메트릭 계산 (샘플별 평균)"""
    model.eval()
    dice_total, iou_total, count = 0.0, 0.0, 0

    with torch.no_grad():
        for data in dataloader:
            images, targets = data
            images = images.to(device).float()
            targets = targets.to(device)
            outputs = model(images)

            if isinstance(outputs, dict):
                outputs = outputs['out']
            while isinstance(outputs, tuple):
                outputs = outputs[0]

            # 모든 모델은 raw logits 출력 -> sigmoid 적용
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # 샘플별로 메트릭 계산
            batch_size = images.size(0)
            for i in range(batch_size):
                pred_i = preds[i]
                target_i = targets[i]
                dice_total += compute_dice(pred_i, target_i).item()
                iou_total += compute_iou(pred_i, target_i).item()
                count += 1

            # BETA TEST: 1 batch만
            if config is not None and config.BETA_TEST:
                break

    model.train()
    return dice_total / count, iou_total / count


def create_model(network_name):
    """
    모델 자동 생성 - 클래스명으로 동적 생성
    num_classes/out_channels를 자동 감지하여 1로 설정
    """
    if network_name not in globals():
        raise ValueError(f"Model '{network_name}' not found. Did you add it to models/ folder?")

    model_class = globals()[network_name]

    # __init__ 파라미터 분석
    sig = inspect.signature(model_class.__init__)
    params = sig.parameters

    # num_classes 또는 out_channels 자동 설정
    kwargs = {}
    if 'num_classes' in params:
        kwargs['num_classes'] = 1
    if 'out_channels' in params:
        kwargs['out_channels'] = 1

    return model_class(**kwargs)


def run_experiment(config, seed, dataset_name, network_name, device):
    """
    단일 실험 수행 (재현성 보장)
    순서: seed -> dataset -> model
    """
    # 훈련 시작 시간
    train_time = datetime.now().strftime('%y%m%d_%H%M%S')

    # 폴더명: 모델_데이터셋_seed_시드번호
    exp_name = f"{network_name}_{dataset_name}_seed_{seed}"

    beta_str = " [BETA]" if config.BETA_TEST else ""
    print(f'{exp_name}{beta_str}')

    # 시드 설정 (transformer 생성 전에 설정해야 함!)
    set_seed_all(seed)

    # config 설정 (engine.py 호환용)
    config.network = network_name
    config.datasets = dataset_name
    config.data_path = config.data_paths[dataset_name]
    config.train_transformer, config.test_transformer = get_transformers(dataset_name, config)
    # 모델별 loss 선택
    LOSS_MAP = {
        'bce': GT_BCEOnlyLoss,
        'bce_dice_squared': GT_BceDiceSquaredLoss,
        'boundary_focal': GT_BoundaryFocalLoss,
        'lovasz': GT_LovaszLoss,
        'uncertainty': GT_UncertaintyLoss,
    }
    loss_name = getattr(config, 'model_loss_map', {}).get(network_name, None)
    if loss_name and loss_name in LOSS_MAP:
        config.criterion = LOSS_MAP[loss_name](wb=1, wd=1)
    else:
        config.criterion = GT_BceDiceLoss(wb=1, wd=1)

    # 실험 디렉토리 설정 (플랫 구조)
    exp_work_dir = os.path.join(config.results_root, exp_name)
    log_dir = os.path.join(exp_work_dir, 'log')
    checkpoint_dir = os.path.join(exp_work_dir, 'checkpoints')
    outputs_dir = os.path.join(exp_work_dir, 'outputs')

    for d in [exp_work_dir, log_dir, checkpoint_dir, outputs_dir]:
        os.makedirs(d, exist_ok=True)

    # 모델 py 파일 복사
    model_src_path = os.path.join('models', f'{network_name}.py')
    if os.path.exists(model_src_path):
        shutil.copy(model_src_path, os.path.join(exp_work_dir, f'{network_name}.py'))

    # Logger & Writer
    logger = get_logger(exp_name, log_dir)
    writer = SummaryWriter(os.path.join(exp_work_dir, 'summary'))
    log_config_info(config, logger)
    logger.info(f"Dataset: {dataset_name}, Model: {network_name}, Seed: {seed}")

    torch.cuda.empty_cache()

    # 데이터 로더 (num_workers=0으로 완전한 재현성 테스트)
    train_dataset = NPY_datasets(config.data_path, config, train=True, exp_idx=seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )

    val_dataset = NPY_datasets(config.data_path, config, train=False, exp_idx=seed)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    test_dataset = Test_datasets(config.data_path, config, exp_idx=seed)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    # 모델 생성
    model = create_model(network_name).to(device)

    # FLOPs & Params 계산
    try:
        from thop import profile
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        flops_str = f"{flops/1e9:.2f} GFLOPs"
    except:
        flops = 0
        flops_str = "N/A"

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model FLOPs: {flops_str}, Params: {num_params:,}")

    # Optimizer & Scheduler
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    # 학습
    min_loss = float('inf')
    min_epoch = 1
    step = 0
    import time
    train_start_time = time.time()

    # 훈련 직전 시드 재설정 (241228 방식)
    set_seed_all(seed)

    for epoch in range(1, config.epochs + 1):
        torch.cuda.empty_cache()

        # Train
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for iter_idx, data in enumerate(train_loader):
            step += 1
            optimizer.zero_grad()
            images, targets = data
            images = images.cuda(non_blocking=True).float()
            targets = targets.cuda(non_blocking=True).float()

            outputs = model(images)

            # 모델이 자체 compute_loss 메서드가 있으면 사용
            if hasattr(model, 'compute_loss'):
                # compute_loss는 raw logits을 받아서 내부적으로 처리
                loss = model.compute_loss(outputs, targets, torch.nn.BCEWithLogitsLoss())
                # NaN 체크용 out 추출
                if isinstance(outputs, tuple):
                    out = torch.sigmoid(outputs[-1])
                elif isinstance(outputs, dict):
                    out = torch.sigmoid(outputs['out'])
                else:
                    out = torch.sigmoid(outputs)
            else:
                # 기존 로직: 출력 형태 처리
                if isinstance(outputs, dict):
                    out = outputs['out']
                    gt_pre = (out, out, out, out, out)
                elif isinstance(outputs, tuple):
                    gt_pre, out = outputs
                else:
                    out = outputs
                    gt_pre = (out, out, out, out, out)

                # 모든 모델은 raw logits 출력 -> sigmoid 적용
                gt_pre = tuple(torch.sigmoid(x) for x in gt_pre)
                out = torch.sigmoid(out)

                # 수치 안정성: [0, 1] 범위로 clamp
                gt_pre = tuple(torch.clamp(x, 1e-7, 1 - 1e-7) for x in gt_pre)
                out = torch.clamp(out, 1e-7, 1 - 1e-7)

                loss = criterion(gt_pre, out, targets)

            # NaN 체크 및 처리
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError(f"NaN/Inf detected in model output at epoch {epoch}, iter {iter_idx}")

            # Loss NaN 체크
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"NaN/Inf loss at epoch {epoch}, iter {iter_idx}")
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss_sum += loss.item()
            train_count += 1

            # Step logging
            if step % config.print_interval == 0:
                logger.info(f"Step {step}: loss = {loss.item():.4f}")
                writer.add_scalar("Train/Loss", loss.item(), step)

            # BETA TEST: 1 step만 실행
            if config.BETA_TEST:
                break

        scheduler.step()
        train_loss = train_loss_sum / train_count if train_count > 0 else 0.0

        # Validation
        val_loss = val_one_epoch(val_loader, model, criterion, epoch, logger, config, writer)

        # Metrics
        val_dice, val_iou = evaluate_metrics(model, val_loader, device, config)

        # Best model 저장
        is_best = val_loss < min_loss
        if is_best:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = val_loss
            min_epoch = epoch

        # Epoch summary print
        date = datetime.now().strftime('%m/%d %H:%M:%S')
        best_str = f" *Best V_Loss: {val_loss:.6f}" if is_best else ""
        print(f"{epoch}EP({date}): T_Loss: {train_loss:.6f} V_Loss: {val_loss:.6f} IoU: {val_iou:.4f}{best_str}")

        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, IoU: {val_iou:.4f}{best_str}")
        writer.add_scalar("Train/EpochLoss", train_loss, epoch)
        writer.add_scalar("Val/Dice", val_dice, epoch)
        writer.add_scalar("Val/IoU", val_iou, epoch)

    # 훈련 시간 계산
    train_end_time = time.time()
    total_training_time = train_end_time - train_start_time

    # 테스트 결과 저장용
    test_results = {
        'IoU': None,
        'Dice': None,
        'Precision': None,
        'Recall': None
    }

    # 테스트
    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best.pth')))
        switch_to_deploy(model)  # RepConv 등 reparameterization (추론 최적화)

        # 테스트 메트릭 계산
        test_dice, test_iou = evaluate_metrics(model, test_loader, device, config)

        # 상세 메트릭 계산
        # 샘플별 Precision, Recall 계산
        model.eval()
        precision_total, recall_total, sample_count = 0.0, 0.0, 0
        with torch.no_grad():
            for data in test_loader:
                img, msk = data
                img = img.to(device).float()
                outputs = model(img)

                if isinstance(outputs, dict):
                    outputs = outputs['out']
                while isinstance(outputs, tuple):
                    outputs = outputs[0]

                # 모든 모델은 raw logits 출력 -> sigmoid 적용
                pred = (torch.sigmoid(outputs) > 0.5).float()

                # 샘플별로 계산
                batch_size = img.size(0)
                for i in range(batch_size):
                    pred_i = pred[i].cpu().numpy().flatten()
                    gt_i = msk[i].numpy().flatten()

                    y_pred = (pred_i >= 0.5).astype(int)
                    y_true = (gt_i >= 0.5).astype(int)

                    TP = np.sum((y_pred == 1) & (y_true == 1))
                    FP = np.sum((y_pred == 1) & (y_true == 0))
                    FN = np.sum((y_pred == 0) & (y_true == 1))

                    prec = TP / (TP + FP) if (TP + FP) != 0 else 0
                    rec = TP / (TP + FN) if (TP + FN) != 0 else 0

                    precision_total += prec
                    recall_total += rec
                    sample_count += 1

                # BETA TEST: 1 batch만
                if config.BETA_TEST:
                    break

        precision = precision_total / sample_count if sample_count > 0 else 0
        recall = recall_total / sample_count if sample_count > 0 else 0

        test_results = {
            'IoU': test_iou,
            'Dice': test_dice,
            'Precision': precision,
            'Recall': recall
        }

        print(f'Test: IoU {test_iou:.3f} | Dice {test_dice:.3f} | Prec {precision:.3f} | Recall {recall:.3f}')
        logger.info(f"Test - IoU: {test_iou:.4f}, Dice: {test_dice:.4f}, Prec: {precision:.4f}, Recall: {recall:.4f}")

        # 기존 test_one_epoch 호출 (시각화용) - BETA 모드에서는 스킵
        if not config.BETA_TEST:
            test_one_epoch(test_loader, model, criterion, logger, config, outputs_dir, writer)

        # 시각화 수행 (VISUALIZE_OUTPUT=True인 경우)
        if hasattr(config, 'VISUALIZE_OUTPUT') and config.VISUALIZE_OUTPUT:
            visualize_predictions(model, test_loader, device, config.results_root, dataset_name, network_name, seed)

        os.rename(os.path.join(checkpoint_dir, 'best.pth'),
                  os.path.join(checkpoint_dir, f'best-epoch{min_epoch}.pth'))

    writer.close()

    # 훈련 시간 포맷 (HH:MM:SS)
    hours, remainder = divmod(int(total_training_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    training_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # GPU 메모리 해제 (누수 방지)
    del model, optimizer, scheduler, criterion
    del train_loader, val_loader, test_loader
    del train_dataset, val_dataset, test_dataset
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return {
        'Experiment Time': config.experiment_time,
        'Train Time': train_time,
        'Model Name': network_name,
        'Dataset Name': dataset_name,
        'Iteration': seed,
        'Val Loss': min_loss,
        **test_results,
        'Params': f"{num_params:,}",
        'FLOPs': f"{int(flops):,}" if flops > 0 else "N/A",
        'Total Training Time': training_time_str,
        'Best Epoch': min_epoch,
        'Path': os.path.abspath(exp_work_dir)
    }


def run_all_experiments(config, start_from=None):
    """
    전체 실험 실행

    Args:
        config: 설정 객체
        start_from: (seed_idx, dataset_idx, model_idx) 튜플 (1-indexed)
                   예: (2, 2, 10) = 2번째 iteration, 2번째 데이터셋, 10번째 모델부터 시작
                   None이면 처음부터 시작
    """
    import pandas as pd

    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 전체 실험 결과 저장
    all_results = []

    # 시드 리스트
    seeds = list(range(config.seed_range[0], config.seed_range[1] + 1))
    total = len(seeds) * len(config.datasets_list) * len(config.networks)
    current = 0

    # start_from 처리 (1-indexed -> 0-indexed)
    skip_until = None
    if start_from is not None:
        seed_idx, dataset_idx, model_idx = start_from
        skip_until = (seed_idx - 1, dataset_idx - 1, model_idx - 1)
        print(f"Resuming from: seed[{seed_idx}]={seeds[seed_idx-1]}, dataset[{dataset_idx}]={config.datasets_list[dataset_idx-1]}, model[{model_idx}]={config.networks[model_idx-1]}")

    print(f"Total: {total} experiments | Seeds: {seeds} | Datasets: {config.datasets_list} | Models: {config.networks}")

    # 결과 폴더 생성
    os.makedirs(config.results_root, exist_ok=True)

    # 순서: seed -> dataset -> model
    for seed_i, seed in enumerate(seeds):
        for dataset_i, dataset_name in enumerate(config.datasets_list):
            for model_i, network_name in enumerate(config.networks):
                current += 1

                # start_from 스킵 로직
                if skip_until is not None:
                    if (seed_i, dataset_i, model_i) < skip_until:
                        print(f"[{current}/{total}] SKIP {network_name}/{dataset_name}/seed{seed}")
                        continue

                print(f"[{current}/{total}] ", end="")

                try:
                    result = run_experiment(config, seed, dataset_name, network_name, device)
                    # CUDA 비동기 에러 체크
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    all_results.append(result)
                except Exception as e:
                    # 에러 발생 시 에러 row 생성
                    error_msg = str(e)

                    # CUDA 에러 처리
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                        except RuntimeError as cuda_err:
                            error_msg = f"CUDA Error: {str(cuda_err)}"

                    error_result = {
                        'Experiment Time': config.experiment_time,
                        'Train Time': datetime.now().strftime('%y%m%d_%H%M%S'),
                        'Model Name': network_name,
                        'Dataset Name': dataset_name,
                        'Iteration': seed,
                        'Val Loss': None,
                        'IoU': None,
                        'Dice': None,
                        'Precision': None,
                        'Recall': None,
                        'Params': 'ERROR',
                        'FLOPs': 'ERROR',
                        'Total Training Time': 'ERROR',
                        'Best Epoch': None,
                        'Path': 'ERROR',
                        'Error': error_msg[:200]  # 에러 메시지 (200자 제한)
                    }
                    all_results.append(error_result)
                    print(f"\n  [ERROR] {network_name}/{dataset_name}/seed{seed}: {error_msg[:100]}")

                    # CUDA 상태 초기화 + 메모리 해제
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        try:
                            torch.cuda.reset_peak_memory_stats()
                        except:
                            pass

                # 중간 저장 (실험이 중단되어도 결과 보존)
                df_temp = pd.DataFrame(all_results)
                df_temp.to_csv(os.path.join(config.results_root, 'results.csv'), index=False)

    # 최종 결과 DataFrame 생성
    df = pd.DataFrame(all_results)

    # CSV 저장
    csv_path = os.path.join(config.results_root, 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")

    # 요약 생성 (에러 row 제외)
    print("--- Summary ---")
    summary_data = []
    error_count = df['IoU'].isna().sum() if 'IoU' in df.columns else 0
    if error_count > 0:
        print(f"(Skipping {error_count} failed experiments)")

    for dataset in config.datasets_list:
        for model in config.networks:
            subset = df[(df['Dataset Name'] == dataset) & (df['Model Name'] == model)]
            # 에러가 아닌 row만 필터링 (IoU가 있는 경우)
            subset = subset[subset['IoU'].notna()]
            if len(subset) > 0:
                iou_mean = subset['IoU'].mean()
                iou_std = subset['IoU'].std()
                dice_mean = subset['Dice'].mean()
                dice_std = subset['Dice'].std()
                print(f"{dataset}/{model}: IoU {iou_mean:.4f}±{iou_std:.4f} | Dice {dice_mean:.4f}±{dice_std:.4f}")

                summary_data.append({
                    'Dataset Name': dataset,
                    'Model Name': model,
                    'IoU Mean': iou_mean,
                    'IoU Std': iou_std,
                    'Dice Mean': dice_mean,
                    'Dice Std': dice_std,
                    'Precision Mean': subset['Precision'].mean(),
                    'Recall Mean': subset['Recall'].mean()
                })

    # 요약 CSV 저장
    df_summary = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(config.results_root, 'summary.csv')
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"\nSummary saved to: {summary_csv_path}")

    return df, df_summary


def run_eval_from_csv(config, csv_path):
    """
    기존 학습된 모델의 CSV를 읽어서 추론만 수행

    Args:
        config: 설정 객체 (gpu_id, img_size 등 필요)
        csv_path: 기존 결과 CSV 경로 (예: results/260209_013501/results.csv)
    """
    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # CSV 읽기
    df = pd.read_csv(csv_path)
    checkpoint_dir = os.path.dirname(csv_path)

    # 새 결과 폴더 생성 (training과 동일한 구조)
    eval_time = datetime.now().strftime('%y%m%d_%H%M%S')
    eval_results_dir = f'results/eval_{eval_time}/'
    os.makedirs(eval_results_dir, exist_ok=True)

    print(f"Loading experiments from: {csv_path}")
    print(f"Total: {len(df)} experiments")
    print(f"Results will be saved to: {eval_results_dir}")

    # 새 결과 저장
    all_results = []

    for idx, row in df.iterrows():
        # CSV 컬럼명 호환 (이전 버전: Model/Dataset, 현재: Model Name/Dataset Name)
        model_name = row.get('Model Name', row.get('Model', None))
        dataset_name = row.get('Dataset Name', row.get('Dataset', None))
        iteration = int(row.get('Iteration', 1))
        best_epoch = int(row.get('Best Epoch', 0))
        experiment_time = row.get('Experiment Time', None)

        if model_name is None or dataset_name is None:
            print(f"[{idx+1}/{len(df)}] Skipping - missing Model/Dataset info")
            continue

        # 체크포인트 경로 찾기 (Experiment Time 기반)
        exp_folder = f"{model_name}_{dataset_name}_seed_{iteration}"

        # Experiment Time이 있으면 해당 폴더에서 찾기
        if experiment_time:
            checkpoint_path = os.path.join('results', str(experiment_time), exp_folder, 'checkpoints', f'best-epoch{best_epoch}.pth')
        else:
            # 없으면 CSV 파일 위치 기준
            checkpoint_path = os.path.join(checkpoint_dir, exp_folder, 'checkpoints', f'best-epoch{best_epoch}.pth')

        if not os.path.exists(checkpoint_path):
            print(f"[{idx+1}/{len(df)}] Skipping {model_name}/{dataset_name}/seed{iteration} - checkpoint not found: {checkpoint_path}")
            continue

        print(f"[{idx+1}/{len(df)}] Evaluating {model_name} on {dataset_name} (seed={iteration})")

        try:
            result = eval_single_experiment(
                config, iteration, dataset_name, model_name, device, checkpoint_path, row, eval_results_dir
            )
            all_results.append(result)

            # 중간 저장
            df_temp = pd.DataFrame(all_results)
            df_temp.to_csv(os.path.join(eval_results_dir, 'results.csv'), index=False)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # 최종 결과 저장
    if all_results:
        df_results = pd.DataFrame(all_results)
        csv_save_path = os.path.join(eval_results_dir, 'results.csv')
        df_results.to_csv(csv_save_path, index=False)
        print(f"\nResults saved to: {csv_save_path}")

        # 요약 생성
        print("\n--- Summary ---")
        summary_data = []
        datasets = df_results['Dataset Name'].unique()
        models = df_results['Model Name'].unique()

        for dataset in datasets:
            for model in models:
                subset = df_results[(df_results['Dataset Name'] == dataset) & (df_results['Model Name'] == model)]
                if len(subset) > 0:
                    iou_mean = subset['IoU'].mean()
                    iou_std = subset['IoU'].std()
                    dice_mean = subset['Dice'].mean()
                    dice_std = subset['Dice'].std()
                    print(f"{dataset}/{model}: IoU {iou_mean:.4f}±{iou_std:.4f} | Dice {dice_mean:.4f}±{dice_std:.4f}")

                    summary_data.append({
                        'Dataset Name': dataset,
                        'Model Name': model,
                        'IoU Mean': iou_mean,
                        'IoU Std': iou_std,
                        'Dice Mean': dice_mean,
                        'Dice Std': dice_std,
                        'Precision Mean': subset['Precision'].mean(),
                        'Recall Mean': subset['Recall'].mean()
                    })

        # 요약 CSV 저장
        df_summary = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(eval_results_dir, 'summary.csv')
        df_summary.to_csv(summary_csv_path, index=False)
        print(f"Summary saved to: {summary_csv_path}")

        return df_results
    else:
        print("No successful evaluations.")
        return None


def eval_single_experiment(config, seed, dataset_name, network_name, device, checkpoint_path, original_row=None, results_dir=None):
    """단일 모델 추론 평가"""
    set_seed_all(seed)

    # config 설정
    config.network = network_name
    config.data_path = config.data_paths[dataset_name]

    # transformer 설정
    config.test_transformer = transforms.Compose([
        myNormalize(dataset_name, train=False),
        myToTensor(),
        myResizeKeepRatio(config.input_size_h, config.input_size_w)
    ])

    # 데이터 로더 생성 (테스트만, 241228 방식)
    test_dataset = Test_datasets(config.data_path, config, exp_idx=seed)
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )

    # 모델 생성 및 로드
    model = create_model(network_name).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    switch_to_deploy(model)  # RepConv 등 reparameterization
    model.eval()

    # 모든 메트릭을 한 번의 추론으로 계산
    dice_total, iou_total = 0.0, 0.0
    precision_total, recall_total = 0.0, 0.0
    sample_count = 0

    with torch.no_grad():
        for data in test_loader:
            img, msk = data
            img = img.to(device).float()
            msk = msk.to(device)
            outputs = model(img)

            if isinstance(outputs, dict):
                outputs = outputs['out']
            while isinstance(outputs, tuple):
                outputs = outputs[0]

            # 모든 모델은 raw logits 출력 -> sigmoid 적용
            pred = (torch.sigmoid(outputs) > 0.5).float()

            # 샘플별 메트릭 계산
            batch_size = img.size(0)
            for i in range(batch_size):
                pred_i = pred[i]
                target_i = msk[i]

                # IoU, Dice (GPU tensor)
                dice_total += compute_dice(pred_i, target_i).item()
                iou_total += compute_iou(pred_i, target_i).item()

                # Precision, Recall (numpy)
                pred_np = pred_i.cpu().numpy().flatten()
                gt_np = target_i.cpu().numpy().flatten()

                y_pred = (pred_np >= 0.5).astype(int)
                y_true = (gt_np >= 0.5).astype(int)

                TP = np.sum((y_pred == 1) & (y_true == 1))
                FP = np.sum((y_pred == 1) & (y_true == 0))
                FN = np.sum((y_pred == 0) & (y_true == 1))

                prec = TP / (TP + FP) if (TP + FP) != 0 else 0
                rec = TP / (TP + FN) if (TP + FN) != 0 else 0

                precision_total += prec
                recall_total += rec
                sample_count += 1

    test_iou = iou_total / sample_count if sample_count > 0 else 0
    test_dice = dice_total / sample_count if sample_count > 0 else 0
    precision = precision_total / sample_count if sample_count > 0 else 0
    recall = recall_total / sample_count if sample_count > 0 else 0

    print(f"  IoU: {test_iou:.4f} | Dice: {test_dice:.4f} | Prec: {precision:.4f} | Recall: {recall:.4f}")

    # 시각화 수행 (VISUALIZE_OUTPUT=True인 경우)
    if hasattr(config, 'VISUALIZE_OUTPUT') and config.VISUALIZE_OUTPUT and results_dir:
        visualize_predictions(model, test_loader, device, results_dir, dataset_name, network_name, seed)

    # 결과 딕셔너리 (training CSV와 동일한 컬럼 순서)
    result = {
        'Experiment Time': original_row.get('Experiment Time', '') if original_row is not None else '',
        'Train Time': original_row.get('Train Time', '') if original_row is not None else '',
        'Model Name': network_name,
        'Dataset Name': dataset_name,
        'Iteration': seed,
        'Val Loss': original_row.get('Val Loss', '') if original_row is not None else '',
        'IoU': test_iou,
        'Dice': test_dice,
        'Precision': precision,
        'Recall': recall,
        'Params': original_row.get('Params', '') if original_row is not None else '',
        'FLOPs': original_row.get('FLOPs', '') if original_row is not None else '',
        'Total Training Time': original_row.get('Total Training Time', '') if original_row is not None else '',
        'Best Epoch': original_row.get('Best Epoch', '') if original_row is not None else '',
    }

    return result
