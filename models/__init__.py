"""
모델 자동 import - 새 모델 파일 추가 시 자동으로 인식됨
models/ 폴더에 .py 파일만 추가하면 끝!

규칙: 파일명과 동일한 클래스명이 메인 모델
예) EGEUNet.py -> EGEUNet 클래스가 메인 모델
"""
import os
import importlib
import inspect
import torch.nn as nn

# 현재 디렉토리의 모든 .py 파일 찾기
_current_dir = os.path.dirname(os.path.abspath(__file__))
_model_files = [f[:-3] for f in os.listdir(_current_dir)
                if f.endswith('.py') and f != '__init__.py']

# 각 파일에서 nn.Module을 상속받는 클래스 자동 import
_all_models = {}
_main_models = []  # 메인 모델 (파일명 == 클래스명)

for _module_name in _model_files:
    try:
        _module = importlib.import_module(f'.{_module_name}', package='models')

        # 모듈에서 nn.Module 상속 클래스 찾기
        for _name, _obj in inspect.getmembers(_module, inspect.isclass):
            if issubclass(_obj, nn.Module) and _obj is not nn.Module:
                # 해당 모듈에서 정의된 클래스만 (import된 것 제외)
                if _obj.__module__ == _module.__name__:
                    _all_models[_name] = _obj

                    # 파일명과 클래스명이 동일하면 메인 모델
                    if _name == _module_name:
                        _main_models.append(_name)

    except Exception as e:
        print(f"Warning: Failed to import {_module_name}: {e}")

# 메인 모델만 현재 네임스페이스에 등록 (파일명 == 클래스명)
# 헬퍼 클래스(conv_block, Attention_block 등)는 export하지 않음
_main_model_classes = {k: v for k, v in _all_models.items() if k in _main_models}
globals().update(_main_model_classes)

# __all__ 설정 (from models import * 지원) - 메인 모델만
__all__ = list(_main_model_classes.keys())

# 사용 가능한 메인 모델 출력 함수
def list_models():
    """실험에 사용 가능한 메인 모델 목록 반환 (파일명 == 클래스명)"""
    return sorted(_main_models)
