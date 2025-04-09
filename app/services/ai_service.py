"""
AI 서비스 관련 비즈니스 로직
"""
import os
import logging
import aiohttp
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

from app.core.config import settings
from app.models.ai_model import model_instance

logger = logging.getLogger(__name__)


class AIService:
    """AI 서비스 클래스"""
    
    @staticmethod
    async def download_image(image_url: str) -> Optional[Path]:
        """
        이미지 URL에서 이미지 다운로드
        
        Args:
            image_url (str): 다운로드할 이미지 URL
            
        Returns:
            Optional[Path]: 다운로드된 이미지 경로 또는 실패 시 None
        """
        try:
            # 임시 디렉토리 확인 및 생성
            temp_dir = Path(settings.TEMP_DIR)
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # 임시 파일 경로 생성 (URL에서 파일명 추출 또는 임의 생성)
            file_name = os.path.basename(image_url.split("?")[0])  # URL 파라미터 제거
            if not file_name or len(file_name) < 5:  # 파일명이 너무 짧으면 임의 생성
                file_name = f"temp_image_{asyncio.current_task().get_name()}.jpg"
            
            temp_file_path = temp_dir / file_name
            
            # 이미지 다운로드
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to download image: HTTP {response.status}")
                        return None
                    
                    # 파일 크기 확인
                    content_length = response.content_length
                    if content_length and content_length > settings.MAX_IMAGE_SIZE:
                        logger.error(f"Image too large: {content_length} bytes (max: {settings.MAX_IMAGE_SIZE})")
                        return None
                    
                    # 이미지 저장
                    content = await response.read()
                    if len(content) > settings.MAX_IMAGE_SIZE:
                        logger.error(f"Downloaded image too large: {len(content)} bytes")
                        return None
                    
                    with open(temp_file_path, "wb") as f:
                        f.write(content)
                    
                    logger.info(f"Image downloaded successfully: {temp_file_path}")
                    return temp_file_path
                    
        except Exception as e:
            logger.error(f"Error downloading image: {e}")
            return None
    
    @staticmethod
    async def ensure_model_loaded():
        """
        모델이 로드되었는지 확인하고, 로드되지 않았다면 로드
        
        Returns:
            bool: 모델 로드 성공 여부
            
        Raises:
            RuntimeError: 모델 로드 실패 시
        """
        if not model_instance.model_loaded:
            try:
                logger.info("Model not loaded. Loading model...")
                if not model_instance.load_model():
                    raise RuntimeError("Failed to load AI model.")
                logger.info("Model loaded successfully.")
                return True
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise RuntimeError(f"Failed to load AI model: {e}")
        return True
    
    @staticmethod
    async def cleanup_temp_file(file_path: Optional[Path]):
        """
        임시 파일 정리
        
        Args:
            file_path (Optional[Path]): 정리할 파일 경로
        """
        if file_path and file_path.exists():
            try:
                os.remove(file_path)
                logger.info(f"Temporary file removed: {file_path}")
            except Exception as e:
                logger.error(f"Error removing temporary file: {e}")
    
    @staticmethod
    async def analyze_image(image_path: Path) -> Dict[str, Any]:
        """
        이미지 분석 수행
        
        Args:
            image_path (Path): 분석할 이미지 경로
            
        Returns:
            Dict[str, Any]: 분석 결과
            
        Raises:
            RuntimeError: 분석 오류 발생 시
        """
        try:
            # 모델 로드 상태 확인
            await AIService.ensure_model_loaded()
            
            # 이미지 분석
            result = await model_instance.analyze_image(image_path)
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise
    
    @staticmethod
    async def process_image_url(image_url: str) -> Dict[str, Any]:
        """
        이미지 URL 처리 및 분석 수행
        
        Args:
            image_url (str): 분석할 이미지 URL
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        temp_file_path = None
        try:
            # 이미지 다운로드
            temp_file_path = await AIService.download_image(image_url)
            if not temp_file_path:
                raise ValueError("Failed to download image")
            
            # 이미지 분석
            result = await AIService.analyze_image(temp_file_path)
            return result
            
        except Exception as e:
            logger.error(f"Error processing image URL: {e}")
            raise
            
        finally:
            # 임시 파일 정리
            if temp_file_path:
                await AIService.cleanup_temp_file(temp_file_path)


# 서비스 인스턴스
ai_service = AIService()
