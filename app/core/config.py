import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # 기본 정보
    PROJECT_NAME: str = "LLaVA Photo Evaluation Service"
    PROJECT_DESCRIPTION: str = "API for photo analysis and evaluation using LLaVA multi-modal model"
    API_VERSION: str = "v1"
    API_PREFIX: str = "/api/v1"
    
    # 서버 설정
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "False").lower() in ("true", "1", "t")
    WORKERS: int = int(os.getenv("WORKERS", 1))
    
    # GPU 사용 여부
    USE_GPU: bool = os.getenv("USE_GPU", "False").lower() in ("true", "1", "t")
    
    # CORS 설정
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",  # 프론트엔드 개발 서버
        "http://localhost:8000",  # 백엔드 개발 서버
        # 추가 도메인은 여기에 추가
    ]
    
    # LLaVA 모델 설정
    MODEL_NAME: str = os.getenv("MODEL_NAME", "llava-hf/llava-1.5-7b-hf")  # 기본 모델: LLaVA 1.5 7B
    MODEL_DEVICE: str = os.getenv("MODEL_DEVICE", "cuda" if os.getenv("USE_GPU", "False").lower() in ("true", "1", "t") else "cpu")
    
    # 이미지 처리 설정
    MAX_IMAGE_SIZE: int = int(os.getenv("MAX_IMAGE_SIZE", 1024))
    SUPPORTED_FORMATS: List[str] = ["jpg", "jpeg", "png", "webp"]
    
    # 평가 기준
    EVALUATION_CRITERIA: List[str] = [
        "composition",  # 구도
        "sharpness",    # 선명도
        "noise",        # 노이즈
        "exposure",     # 노출
        "color",        # 색감
        "aesthetics"    # 심미성
    ]
    
    # 로깅 설정
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # 추가 환경 변수 무시


# 설정 객체 생성
settings = Settings()