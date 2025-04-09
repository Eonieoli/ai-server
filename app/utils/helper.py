"""
유틸리티 함수 모음
"""
import os
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta

from app.core.config import settings

def get_system_info():
    """시스템 정보를 반환합니다."""
    import platform
    import torch
    
    system_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        system_info.update({
            "cuda_version": torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown",
            "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "None",
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB" if torch.cuda.device_count() > 0 else "None"
        })
    
    return system_info

def setup_logging():
    """로깅 설정"""
    # 로그 디렉토리 생성
    log_dir = settings.ROOT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 현재 날짜로 로그 파일 이름 지정
    log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"

    # 로그 포맷 설정
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 루트 로거 설정
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def cleanup_temp_files(max_age_hours=24):
    """
    임시 파일 정리
    
    Args:
        max_age_hours (int): 이 시간(시간)보다 오래된 파일 삭제
    """
    temp_dir = settings.TEMP_DIR

    if not temp_dir.exists():
        return

    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    for file_path in temp_dir.glob("*"):
        if file_path.is_file():
            file_age = current_time - os.path.getmtime(file_path)
            if max_age_hours == 0 or file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    logging.info(f"임시 파일 삭제: {file_path}")
                except Exception as e:
                    logging.error(f"임시 파일 삭제 실패: {file_path} - {e}")
                    

def scale_score(score, input_range=(1, 10), output_range=(1, 100), high_score_threshold=8):
    """
    점수를 재조정하는 함수
    
    Args:
        score (int, float): 조정할 점수
        input_range (tuple): 입력 점수 범위 (최소, 최대)
        output_range (tuple): 출력 점수 범위 (최소, 최대)
        high_score_threshold (int): 높은 점수로 간주할 임계값
        
    Returns:
        int: 조정된 점수
    """
    import random
    
    # 입력이 숫자가 아니면 기본값 반환
    if not isinstance(score, (int, float)):
        return (output_range[0] + output_range[1]) // 2  # 중간값 반환
    
    # 입력 범위 내에 있는지 확인
    in_min, in_max = input_range
    if not (in_min <= score <= in_max):
        # 범위를 벗어난 경우, 범위 내로 조정
        score = max(in_min, min(in_max, score))
    
    # 기본 점수 계산 (선형 변환)
    out_min, out_max = output_range
    base_score = int((score - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)
    
    # 점수에 따라 다른 변동 적용
    if score < high_score_threshold:
        variation = random.randint(-5, 5)  # -5 ~ +5 사이의 변동
        final_score = max(out_min, min(out_max, base_score + variation))
    else:
        # 높은 점수의 경우 작은 범위의 변동 적용 (높은 점수 유지)
        variation = random.randint(-3, 3)  # -3 ~ +3 사이의 작은 변동
        final_score = max(max(out_min, 70), min(out_max, base_score + variation))
    
    return final_score


def get_system_info():
    """
    시스템 정보 가져오기
    
    Returns:
        dict: 시스템 정보를 담은 딕셔너리
    """
    import platform
    import torch
    
    system_info = {
        "os": platform.system(),
        "python_version": platform.python_version(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
    }
    
    return system_info