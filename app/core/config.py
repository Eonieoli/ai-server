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
    Please evaluate the given image using the following six criteria. For each criterion, provide an integer score from 1 to 100, based on the detailed sub-criteria below. DO NOT USE ROUNDED SCORES ENDING IN 0 OR 5 (like 60, 65, 70, 75, 80). Instead, use precise values like 63, 72, 84, 97, etc.

    For each item, return a short feedback sentence along with the score, in the format:  
    "criterion_name": {"score": integer, "comment": "brief constructive suggestion"}

    Be analytical and constructive — avoid emotional or poetic language. Give practical feedback that could help improve the image if revised.

    ### Evaluation Criteria:

    1. **Composition** (Max 100 points)
    - Rule of thirds or centered framing clearly applied (25 pts)
    - Natural leading lines guide the viewer’s eyes (25 pts)
    - Use of framing elements (doorways, trees, etc.) to emphasize subject (25 pts)
    - Overall balance and visual flow (25 pts)

    2. **Sharpness** (Max 100 points)
    - Subject is clearly focused (40 pts)
    - Blur appears intentional (e.g., bokeh) (30 pts)
    - Textures and fine details are preserved (30 pts)

    3. **Subject** (Max 100 points)
    - Subject is clearly defined (50 pts)
    - The photo evokes emotion or conveys a message (30 pts)
    - Background complements the subject (20 pts)

    4. **Exposure** (Max 100 points)
    - No blown highlights (30 pts)
    - No loss of detail in shadows (30 pts)
    - Well-balanced contrast and brightness (40 pts)

    5. **Color Harmony** (Max 100 points)
    - Natural and undistorted color rendering (40 pts)
    - Appropriate saturation levels (30 pts)
    - Visually pleasing color combinations (30 pts)

    6. **Aesthetic Quality** (Max 100 points)
    - Overall visual beauty and impact (40 pts)
    - Emotional/storytelling depth (30 pts)
    - Creative or original visual approach (30 pts)

    ---

    **VERY IMPORTANT**
    - ALL SCORES MUST NOT END IN 0 OR 5. Never use scores like 70, 75, 80, 85, 90! Always use specific values like 73, 82, 91, etc.
    - The "hashtags" field MUST be an array of simple strings WITHOUT the '#' symbol. Example: ["nature", "portrait", "sunset"] NOT ["#nature", "#portrait", "#sunset"]
    - The output must be valid JSON, parseable using Python's `json.loads()`. No trailing commas, proper nesting of braces.
    - The "overall" score should be a weighted average of the other scores (not ending in 0 or 5).

    ---

    ### 📌 Example evaluation:
    Image description: *A woman walking in a park under bright sunlight, with trees and greenery in the background.*

    ```json
    {
    "composition": {
        "score": 82,
        "comment": "The subject is well-placed using the rule of thirds, but leading lines could be stronger."
    },
    "sharpness": {
        "score": 89,
        "comment": "The focus is sharp and textures on the subject’s clothing are clearly visible."
    },
    "subject": {
        "score": 84,
        "comment": "The subject is clearly the woman, and the scene conveys a peaceful mood."
    },
    "exposure": {
        "score": 78,
        "comment": "The image is slightly overexposed in some highlights; a lower ISO could help."
    },
    "color_harmony": {
        "score": 91,
        "comment": "The natural greens and skin tones work well together with no color distortion."
    },
    "aesthetic_quality": {
        "score": 87,
        "comment": "The image has a warm, pleasant feel and a good balance of emotion and beauty."
    },
    "overall": {
        "score": 86,
        "comment": "A well-composed and vibrant image that effectively captures a calm outdoor moment."
    },
    "hashtags": ["sunnyday", "parkportrait", "naturewalk", "outdoorvibes"]
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
