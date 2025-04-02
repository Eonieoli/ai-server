"""
유틸리티 및 헬퍼 함수
"""
import os
import logging
import asyncio
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

import torch

from app.core.config import settings

logger = logging.getLogger(__name__)


def setup_logging(log_level=logging.INFO):
    """
    로깅 설정
    
    Args:
        log_level: 로그 레벨 (기본: INFO)
    """
    # 로그 포맷 설정
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 핸들러 설정
    handlers = [logging.StreamHandler()]
    
    # 로그 디렉토리 설정 (선택적)
    log_dir = Path(settings.ROOT_DIR) / "logs"
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일 핸들러 추가 (선택적)
    log_file = log_dir / f"ai_server_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    handlers.append(file_handler)
    
    # 로깅 설정 적용
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )


def get_system_info() -> Dict[str, Any]:
    """
    시스템 정보 수집
    
    Returns:
        Dict[str, Any]: 시스템 정보
    """
    info = {
        "cpu_count": os.cpu_count(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "temp_dir_exists": Path(settings.TEMP_DIR).exists(),
        "model_cache_exists": Path(settings.MODEL_CACHE_DIR).exists(),
    }
    
    # GPU 정보 추가
    if info["gpu_available"]:
        info["gpu_info"] = []
        for i in range(info["gpu_count"]):
            info["gpu_info"].append({
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory / (1024 ** 3),  # GB 단위
            })
    
    return info


async def check_url_accessibility(url: str) -> bool:
    """
    URL 접근성 확인
    
    Args:
        url (str): 확인할 URL
        
    Returns:
        bool: 접근 가능 여부
    """
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(url, timeout=5) as response:
                return response.status == 200
    except Exception:
        return False


def cleanup_temp_files(max_age_hours: int = 24):
    """
    오래된 임시 파일 정리
    
    Args:
        max_age_hours (int): 최대 파일 수명 (시간)
    """
    temp_dir = Path(settings.TEMP_DIR)
    if not temp_dir.exists():
        return
    
    # 현재 시간
    now = datetime.now().timestamp()
    max_age_seconds = max_age_hours * 3600
    
    # 임시 디렉토리 내 파일 확인
    for file_path in temp_dir.glob("*"):
        if file_path.is_file():
            file_age = now - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                try:
                    file_path.unlink()
                    logger.info(f"Removed old temporary file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to remove temporary file {file_path}: {e}")
