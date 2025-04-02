"""
환경 설정 및 구성 파일
"""
import os
from pathlib import Path
from typing import Optional, List
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
    
    # AI 모델 관련 설정
    MODEL_NAME: str = "llava-hf/LLaVA-NeXT-Video-7B-hf"
    
    @property
    def MODEL_CACHE_DIR(self) -> Path:
        return self.ROOT_DIR / "models" / "downloads"
    
    USE_GPU: bool = Field(default=False)  # 기본값은 CPU 사용
    GPU_DEVICE: int = Field(default=0)    # GPU 장치 번호 (0부터 시작)
    
    # 임시 파일 디렉토리
    @property
    def TEMP_DIR(self) -> Path:
        return self.ROOT_DIR / "temp"
    
    # 최대 이미지 크기 제한 (바이트 단위, 기본 10MB)
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024
    
    # 평가 카테고리
    EVALUATION_CATEGORIES: List[str] = [
        "composition",     # 구도
        "sharpness",       # 선명도
        "noise",           # 노이즈
        "exposure",        # 노출
        "color_harmony",   # 색감
        "aesthetics"       # 심미성
    ]
    
    # 프롬프트 템플릿
    PROMPT_TEMPLATE: str = """
    이 사진을 다음 기준에 따라 1에서 10까지의 점수로 평가해주세요:
    
    1. 구도(Composition): 사진의 요소들이 얼마나 조화롭게 배치되어 있는지
    2. 선명도(Sharpness): 사진의 주요 주제가 얼마나 선명하게 촬영되었는지
    3. 노이즈(Noise): 사진에 노이즈가 얼마나 적은지 (10 = 노이즈 없음)
    4. 노출(Exposure): 사진의 밝기가 얼마나 적절한지
    5. 색감(Color Harmony): 색상의 조화가 얼마나 잘 이루어졌는지
    6. 심미성(Aesthetics): 전반적인 미적 품질
    
    각 항목에 대한 점수와 간략한 설명을 제공해주세요. 결과는 JSON 형식으로 반환해주세요:
    {
        "composition": {"score": 0, "comment": ""},
        "sharpness": {"score": 0, "comment": ""},
        "noise": {"score": 0, "comment": ""},
        "exposure": {"score": 0, "comment": ""},
        "color_harmony": {"score": 0, "comment": ""},
        "aesthetics": {"score": 0, "comment": ""},
        "overall_comment": ""
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
