"""
환경 설정 및 구성 파일
"""
import os
from pathlib import Path
from typing import Optional, List, Dict
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션 설정"""
    # 프로젝트 루트 디렉토리
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent
    
    # 기본 설정
    APP_NAME: str = "AI-Image-Analysis-Server"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = Field(default=True)
    
    # S3 관련 설정
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_S3_BUCKET_NAME: Optional[str] = None
    AWS_REGION: str = "ap-northeast-2"
    
    # 번역 API 설정
    GOOGLE_TRANSLATE_KEY: Optional[str] = None
    
    # AI 모델 관련 설정
    MODEL_NAME: str = "llava-hf/llava-1.5-7b-hf"
    
    @property
    def MODEL_CACHE_DIR(self) -> Path:
        return self.ROOT_DIR / "models" / "downloads"
    
    USE_GPU: bool = Field(default=True)  # 기본값은 CPU 사용
    GPU_DEVICE: int = Field(default=0)    # GPU 장치 번호 (0부터 시작)
    
    # 임시 파일 디렉토리
    @property
    def TEMP_DIR(self) -> Path:
        return self.ROOT_DIR / "temp"
    
    # 최대 이미지 크기 제한 (바이트 단위, 기본 20MB)
    MAX_IMAGE_SIZE: int = 20 * 1024 * 1024
    
    # 평가 카테고리
    EVALUATION_CATEGORIES: List[str] = [
        "composition",      # 구도
        "sharpness",        # 선명도
        "subject",          # 주제
        "exposure",         # 노출
        "color_harmony",    # 색감
        "aesthetic_quality" # 미적감각
    ]
    
    # 평가 카테고리 한글명
    CATEGORY_KOREAN_NAMES: Dict[str, str] = {
        "composition": "구도",
        "sharpness": "선명도",
        "subject": "주제",
        "exposure": "노출",
        "color_harmony": "색감",
        "aesthetic_quality": "미적감각",
        "overall": "종합평가"
    }
    
    # 프롬프트 템플릿
    PROMPT_TEMPLATE: str = """
    Please evaluate this image according to the following criteria on a scale of 1 to 100 using precise individual integer scores (not rounded to 5 or 10):
    
    1. Composition: How well the elements in the photo are arranged
    2. Sharpness: How clearly the main subject of the photo is captured
    3. Subject: How clear and interesting the main subject of the photo is
    4. Exposure: How appropriate the brightness of the photo is
    5. Color Harmony: How well the colors work together
    6. Aesthetic Quality: The overall aesthetic value, artistic merit, and emotional impact of the photo
    
    Also, please suggest up to 4 relevant hashtags that describe the content, style, or subject of this image.
    
    Provide a score and a brief explanation for each item. Return the results in JSON format:
    {
        "composition": {"score": 0, "comment": ""},
        "sharpness": {"score": 0, "comment": ""},
        "subject": {"score": 0, "comment": ""},
        "exposure": {"score": 0, "comment": ""},
        "color_harmony": {"score": 0, "comment": ""},
        "aesthetic_quality": {"score": 0, "comment": ""},
        "overall": {"score": 0, "comment": ""},
        "hashtags": ["", "", "", ""]
    }
    
    Important: You MUST give exact integer scores between 1 and 100 for each criterion. DO NOT round scores to the nearest 5 or 10. Avoid common round numbers like 70, 75, 80, etc. Use more specific numbers like 73, 88, 91. 

    """
    
    # Pydantic v2 설정
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )


# 설정 인스턴스 생성
settings = Settings()
