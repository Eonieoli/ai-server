"""
환경 설정 및 구성 파일
"""
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EvaluationCategory:
    """평가 카테고리 클래스"""
    def __init__(self, key: str, korean_name: str, description_en: str):
        self.key = key
        self.korean_name = korean_name
        self.description_en = description_en


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
    MODEL_NAME: str = "llava-hf/LLaVA-NeXT-Video-7B-hf"
    
    @property
    def MODEL_CACHE_DIR(self) -> Path:
        return self.ROOT_DIR / "models" / "downloads"
    
    USE_GPU: bool = Field(default=True)  # GPU 사용 여부
    GPU_DEVICE: int = Field(default=0)    # GPU 장치 번호 (0부터 시작)
    
    # 임시 파일 디렉토리
    @property
    def TEMP_DIR(self) -> Path:
        return self.ROOT_DIR / "temp"
    
    # 최대 이미지 크기 제한 (바이트 단위, 기본 20MB)
    MAX_IMAGE_SIZE: int = 20 * 1024 * 1024
    
    # 평가 카테고리 정의
    _EVALUATION_CATEGORIES = [
        EvaluationCategory(
            key="composition",
            korean_name="구도",
            description_en="How well the elements in the photo are arranged"
        ),
        EvaluationCategory(
            key="sharpness",
            korean_name="선명도",
            description_en="How clearly the main subject of the photo is captured"
        ),
        EvaluationCategory(
            key="subject",
            korean_name="주제",
            description_en="How clear and interesting the main subject of the photo is"
        ),
        EvaluationCategory(
            key="exposure",
            korean_name="노출",
            description_en="How appropriate the brightness of the photo is"
        ),
        EvaluationCategory(
            key="color_harmony",
            korean_name="색감",
            description_en="How well the colors work together"
        ),
        EvaluationCategory(
            key="aesthetic_quality",
            korean_name="미적 감각",
            description_en="The overall aesthetic value, artistic merit, and emotional impact of the photo"
        )
    ]
    
    # 오버롤 카테고리 (별도 관리)
    _OVERALL_CATEGORY = EvaluationCategory(
        key="overall",
        korean_name="종합평가",
        description_en="Overall assessment of the photo"
    )
    
    # 평가 카테고리 키 목록
    @property
    def EVALUATION_CATEGORIES(self) -> List[str]:
        return [category.key for category in self._EVALUATION_CATEGORIES]
    
    # 카테고리 한글명 매핑
    @property
    def CATEGORY_KOREAN_NAMES(self) -> Dict[str, str]:
        mapping = {category.key: category.korean_name for category in self._EVALUATION_CATEGORIES}
        mapping[self._OVERALL_CATEGORY.key] = self._OVERALL_CATEGORY.korean_name
        return mapping
    
    # 기본 오류 메시지 (영어)
    DEFAULT_ERROR_MESSAGES_EN: Dict[str, str] = {
        "composition": "Could not analyze the composition.",
        "sharpness": "Could not evaluate the sharpness.",
        "subject": "Could not evaluate the subject.",
        "exposure": "Could not evaluate the exposure.",
        "color_harmony": "Could not evaluate the color harmony.",
        "aesthetic_quality": "Could not evaluate the aesthetic quality.",
        "overall": "An error occurred while analyzing the image."
    }
    
    # 기본 해시태그 (영어)
    DEFAULT_HASHTAGS_EN: List[str] = ["photo", "image", "analysis", "art"]
    
    # 프롬프트 템플릿 생성
    @property
    def PROMPT_TEMPLATE(self) -> str:
        # 카테고리 설명 부분 생성
        category_descriptions = []
        for i, category in enumerate(self._EVALUATION_CATEGORIES, 1):
            category_descriptions.append(f"{i}. {category.key.replace('_', ' ').title()}: {category.description_en}")
        
        # JSON 예시 부분 생성
        json_example_parts = []
        for category in self._EVALUATION_CATEGORIES:
            json_example_parts.append(f'        "{category.key}": {{"score": 0, "comment": ""}}')    
        json_example_parts.append(f'        "{self._OVERALL_CATEGORY.key}": {{"score": 0, "comment": ""}}')    
        json_example_parts.append('        "hashtags": ["", "", "", ""]')
        json_example = '{\\n' + ',\\n'.join(json_example_parts) + '\\n    }'
        
        # 전체 템플릿 생성
        return f"""
    Please evaluate this image according to the following criteria on a scale of 1 to 100:
    
    {"\\n    ".join(category_descriptions)}
    
    Also, please suggest up to 4 relevant hashtags that describe the content, style, or subject of this image.
    
    Provide a score and a brief explanation for each item. Return the results in JSON format:
    {json_example}
    
    Important: Scores should be between 1 and 100.
    """
    
    # Pydantic v2 설정
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )


# 설정 인스턴스 생성
settings = Settings()