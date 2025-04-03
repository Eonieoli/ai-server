"""
유틸리티 함수 모음
"""
import os
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta

from app.core.config import settings


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