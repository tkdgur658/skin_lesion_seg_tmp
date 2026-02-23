"""
효율성 측정 유틸리티 함수 모음
measure_efficiency.ipynb에서 사용
"""
import torch
import torch.nn as nn
import numpy as np
import time
import inspect
import warnings
warnings.filterwarnings('ignore')

from thop import profile
from models import *


def create_model(network_name):
    """모델 자동 생성 - 클래스명으로 동적 생성"""
    if network_name not in globals():
        raise ValueError(f"Model '{network_name}' not found.")

    model_class = globals()[network_name]

    # __init__ 파라미터 분석
    sig = inspect.signature(model_class.__init__)
    params = sig.parameters

    kwargs = {}
    if 'num_classes' in params:
        kwargs['num_classes'] = 1
    if 'out_channels' in params:
        kwargs['out_channels'] = 1

    return model_class(**kwargs)


def count_parameters(model):
    """파라미터 수 계산"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def switch_to_deploy(model):
    """RepConv 등 reparameterizable 모듈을 추론 모드로 전환"""
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()


def measure_flops(model, input_size):
    """FLOPs 계산 (thop 사용)"""
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_size).to(device)

    try:
        model.eval()
        switch_to_deploy(model)  # RepConv 등 reparameterization
        with torch.no_grad():
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return flops, params
    except Exception as e:
        print(f"    FLOPs calculation failed: {e}")
        return None, None


def measure_gpu_inference_time(model, input_size, num_warmup=100, num_iterations=100):
    """GPU Inference Time 측정 (ms)"""
    if not torch.cuda.is_available():
        return None, None

    device = torch.device('cuda')
    model = model.to(device)
    model.eval()
    switch_to_deploy(model)  # RepConv 등 reparameterization

    dummy_input = torch.randn(1, *input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    return np.mean(times), np.std(times)


def measure_cpu_inference_time(model, input_size, num_warmup=100, num_iterations=100):
    """CPU Inference Time 측정 (ms)"""
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    switch_to_deploy(model)  # RepConv 등 reparameterization

    dummy_input = torch.randn(1, *input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    return np.mean(times), np.std(times)


def format_params(num):
    """파라미터 수 포맷팅"""
    if num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)


def format_flops(num):
    """FLOPs 포맷팅"""
    if num is None:
        return "N/A"
    if num >= 1e9:
        return f"{num/1e9:.2f}G"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)


def measure_all_models(models, input_size, num_warmup=100, num_iterations=100):
    """전체 모델 측정"""
    results = []

    print(f"Measuring {len(models)} models...")
    print("=" * 60)

    for i, model_name in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {model_name}")

        try:
            # 모델 생성
            model = create_model(model_name)

            # Parameters
            total_params, _ = count_parameters(model)
            print(f"  Params: {format_params(total_params)}")

            # FLOPs
            flops, _ = measure_flops(model, input_size)
            print(f"  FLOPs: {format_flops(flops)}")

            # GPU Inference Time
            model_gpu = create_model(model_name)
            gpu_mean, gpu_std = measure_gpu_inference_time(
                model_gpu, input_size, num_warmup, num_iterations
            )
            if gpu_mean is not None:
                print(f"  GPU: {gpu_mean:.2f} ± {gpu_std:.2f} ms")
            del model_gpu
            torch.cuda.empty_cache()

            # CPU Inference Time
            model_cpu = create_model(model_name)
            cpu_mean, cpu_std = measure_cpu_inference_time(
                model_cpu, input_size, num_warmup, num_iterations
            )
            print(f"  CPU: {cpu_mean:.2f} ± {cpu_std:.2f} ms")
            del model_cpu

            results.append({
                'Model Name': model_name,
                'Params': total_params,
                'Params (fmt)': format_params(total_params),
                'FLOPs': flops if flops else 0,
                'FLOPs (fmt)': format_flops(flops),
                'GPU Mean (ms)': gpu_mean,
                'GPU Std (ms)': gpu_std,
                'CPU Mean (ms)': cpu_mean,
                'CPU Std (ms)': cpu_std,
            })

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'Model Name': model_name,
                'Params': 0,
                'Params (fmt)': 'N/A',
                'FLOPs': 0,
                'FLOPs (fmt)': 'N/A',
                'GPU Mean (ms)': None,
                'GPU Std (ms)': None,
                'CPU Mean (ms)': None,
                'CPU Std (ms)': None,
            })

    print("\n" + "=" * 60)
    print("Measurement completed!")

    return results
