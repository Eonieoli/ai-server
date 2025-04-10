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
    Please critically evaluate this image using the following criteria. You must assign a precise integer score between 1 and 100 for each criterion. Do NOT round to the nearest 5 or 10. Avoid using numbers like 70, 75, 80, etc. Instead, choose specific values like 73, 87, or 91. Be objective and avoid writing in an overly poetic or emotional tone.

    Evaluation Criteria:

    1. Composition: How well the elements in the photo are arranged.
    2. Sharpness: How clearly the main subject of the photo is captured.
    3. Subject: How clear and interesting the main subject is.
    4. Exposure: How well-balanced the brightness and contrast are.
    5. Color Harmony: How well the colors work together.
    6. Aesthetic Quality: The overall artistic value and visual appeal of the photo.

    In your response:
    - Return a JSON object with the following structure.
    - Each score must be a precise integer between 1 and 100 (not rounded).
    - Keep comments factual, concise, and avoid emotional or poetic language.
    - Include up to 4 relevant hashtags that describe the image's content, style, or subject.

    Format:
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

    Example output:
    {
        "composition": {"score": 87, "comment": "The elements are well-arranged with balanced spacing and alignment."},
        "sharpness": {"score": 91, "comment": "The subject is sharply focused with clear details."},
        "subject": {"score": 88, "comment": "The main subject is prominent and visually interesting."},
        "exposure": {"score": 84, "comment": "Lighting is well-balanced with good contrast and no overexposure."},
        "color_harmony": {"score": 89, "comment": "Colors are complementary and enhance the overall look."},
        "aesthetic_quality": {"score": 90, "comment": "The photo has strong visual appeal and professional quality."},
        "overall": {"score": 89, "comment": "A well-executed image with strong technical and visual components."},
        "hashtags": ["portrait", "cleancomposition", "sharpfocus", "professional"]
    }

    """
    
    # Pydantic v2 설정
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )


# 설정 인스턴스 생성
settings = Settings()
