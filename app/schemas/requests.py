"""
요청에 대한 Pydantic 스키마 정의
"""
from pydantic import BaseModel, HttpUrl, Field


class ImageAnalysisRequest(BaseModel):
    """이미지 분석 요청 스키마"""
    image_url: HttpUrl = Field(
        ..., 
        description="S3에 업로드된 이미지의 URL"
    )
