# app/utils/helper.py
import os
import uuid
import shutil
import asyncio
from fastapi import UploadFile
import cv2
import numpy as np
from PIL import Image
import io
import logging
from app.core.config import settings

# 로깅 설정
logger = logging.getLogger(__name__)

# 임시 디렉토리 설정
TEMP_DIR = os.path.join(os.getcwd(), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)


async def save_upload_file_temporarily(upload_file: UploadFile) -> str:
    """
    업로드된 파일을 임시 디렉토리에 저장
    
    Args:
        upload_file: FastAPI UploadFile 객체
        
    Returns:
        임시 파일 경로
    """
    try:
        # 고유한 파일명 생성
        file_extension = os.path.splitext(upload_file.filename)[1] if upload_file.filename else ".jpg"
        temp_filename = f"{uuid.uuid4()}{file_extension}"
        temp_file_path = os.path.join(TEMP_DIR, temp_filename)
        
        # 파일 내용 읽기
        contents = await upload_file.read()
        
        # 파일 저장
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(contents)
            
        return temp_file_path
        
    except Exception as e:
        logger.error(f"Error saving temporary file: {str(e)}")
        raise e


def remove_file(file_path: str) -> None:
    """
    파일 제거
    
    Args:
        file_path: 제거할 파일 경로
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Error removing file {file_path}: {str(e)}")


def read_image(image_path: str) -> Image.Image:
    """
    이미지 파일을 PIL Image 객체로 읽기
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        PIL Image 객체
    """
    try:
        # PIL로 이미지 읽기
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Error reading image {image_path}: {str(e)}")
        return None


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    이미지 전처리
    
    Args:
        image: PIL Image 객체
        
    Returns:
        전처리된 PIL Image 객체
    """
    try:
        # 이미지 크기 조정 (필요한 경우)
        max_size = settings.MAX_IMAGE_SIZE
        if max(image.size) > max_size:
            # 비율 유지하면서 크기 조정
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
            
        return image
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return image  # 오류 시 원본 이미지 반환

import aiohttp

async def download_image_from_url(image_url: str) -> str:
    """
    URL에서 이미지를 다운로드하여 임시 파일로 저장합니다.
    
    Args:
        image_url: 이미지 URL (S3 또는 다른 URL)
        
    Returns:
        임시 저장된 이미지 파일 경로
    """
    try:
        # 임시 파일 경로 생성
        temp_filename = f"{uuid.uuid4()}.jpg"
        temp_file_path = os.path.join(TEMP_DIR, temp_filename)
        
        # 비동기적으로 이미지 다운로드
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download image: HTTP {response.status}")
                
                # 이미지 데이터 읽기
                image_data = await response.read()
                
                # 임시 파일로 저장
                with open(temp_file_path, "wb") as f:
                    f.write(image_data)
                
        return temp_file_path
    
    except Exception as e:
        logger.error(f"Error downloading image from URL {image_url}: {str(e)}")
        raise e