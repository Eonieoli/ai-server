"""
모델 관련 유틸리티 함수
"""
import os
import sys
import logging
import contextlib
from pathlib import Path
from typing import Optional, Dict, Any, Union, Callable

import torch
from PIL import Image

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_transformers_warnings():
    """
    Transformers 경고 일시적으로 억제하는 컨텍스트 매니저
    """
    import warnings
    from transformers import logging as transformers_logging
    
    # 현재 로깅 레벨 저장
    original_tf_level = transformers_logging.get_verbosity()
    original_py_level = logging.getLogger().level
    
    try:
        # 경고 억제
        warnings.filterwarnings("ignore")
        transformers_logging.set_verbosity_error()
        logging.getLogger().setLevel(logging.ERROR)
        
        yield
    finally:
        # 원래 레벨로 복원
        transformers_logging.set_verbosity(original_tf_level)
        logging.getLogger().setLevel(original_py_level)


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    이미지 파일 로드
    
    Args:
        image_path (Union[str, Path]): 이미지 파일 경로
        
    Returns:
        Image.Image: 로드된 PIL 이미지
    """
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise


def get_available_memory() -> Dict[str, Any]:
    """
    사용 가능한 메모리 정보 수집
    
    Returns:
        Dict[str, Any]: 메모리 정보
    """
    info = {
        "free_cpu_memory": None,
        "free_gpu_memory": None,
        "total_cpu_memory": None,
        "total_gpu_memory": None
    }
    
    # CPU 메모리 정보
    try:
        # psutil 모듈이 없을 수도 있으므로 조건부 가져오기
        try:
            import psutil
            has_psutil = True
        except ImportError:
            has_psutil = False
        
        if has_psutil:
            vm = psutil.virtual_memory()
            info["free_cpu_memory"] = vm.available / (1024 ** 3)  # GB 단위
            info["total_cpu_memory"] = vm.total / (1024 ** 3)     # GB 단위
        else:
            # 대안: 시스템 콜 사용 (Linux/Windows)
            if sys.platform == "linux":
                with open('/proc/meminfo', 'r') as f:
                    lines = f.readlines()
                    
                total = None
                available = None
                
                for line in lines:
                    if 'MemTotal' in line:
                        total = int(line.split()[1])
                    elif 'MemAvailable' in line:
                        available = int(line.split()[1])
                
                if total is not None:
                    info["total_cpu_memory"] = total / (1024 * 1024)  # GB 단위
                if available is not None:
                    info["free_cpu_memory"] = available / (1024 * 1024)  # GB 단위
            elif sys.platform == "win32":
                # Windows - 강제로 값 설정 (Docker 컨테이너에서는 신뢰할 수 없음)
                info["free_cpu_memory"] = 4.0  # 기본값 4GB 가정
                info["total_cpu_memory"] = 8.0  # 기본값 8GB 가정
    except Exception as e:
        logger.warning(f"Error getting CPU memory info: {e}, assuming 4GB free memory")
        info["free_cpu_memory"] = 4.0  # 기본값 4GB 가정
        info["total_cpu_memory"] = 8.0  # 기본값 8GB 가정
    
    # GPU 메모리 정보
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            info["free_gpu_memory"] = (torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)) / (1024 ** 3)
            info["total_gpu_memory"] = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        except Exception as e:
            logger.warning(f"Error getting GPU memory info: {e}")
    
    return info


def is_memory_sufficient(required_cpu_gb: float = 8.0, required_gpu_gb: Optional[float] = None) -> bool:
    """
    필요한 메모리가 충분한지 확인
    
    Args:
        required_cpu_gb (float): 필요한 CPU 메모리 (GB)
        required_gpu_gb (Optional[float]): 필요한 GPU 메모리 (GB), None이면 GPU 체크 무시
        
    Returns:
        bool: 메모리 충분 여부
    """
    memory_info = get_available_memory()
    
    # CPU 메모리 확인
    if memory_info["free_cpu_memory"] is not None and memory_info["free_cpu_memory"] < required_cpu_gb:
        logger.warning(f"Insufficient CPU memory: {memory_info['free_cpu_memory']:.2f}GB available, {required_cpu_gb}GB required")
        return False
    
    # GPU 메모리 확인 (GPU 사용 시)
    if required_gpu_gb is not None and torch.cuda.is_available():
        if memory_info["free_gpu_memory"] is not None and memory_info["free_gpu_memory"] < required_gpu_gb:
            logger.warning(f"Insufficient GPU memory: {memory_info['free_gpu_memory']:.2f}GB available, {required_gpu_gb}GB required")
            return False
    
    return True
