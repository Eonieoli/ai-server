"""
응답에 대한 Pydantic 스키마 정의
"""
from typing import Dict, List
from pydantic import BaseModel, Field


class AnalysisText(BaseModel):
    """분석 텍스트"""
    구도: str
    선명도: str
    주제: str
    노출: str
    색감: str
    미적감각: str


class AnalysisChart(BaseModel):
    """분석 차트 데이터"""
    구도: int
    선명도: int
    주제: int
    노출: int
    색감: int
    미적감각: int


class ImageAnalysisResponse(BaseModel):
    """이미지 분석 응답 스키마"""
    score: int
    comment: str
    analysisText: AnalysisText
    analysisChart: AnalysisChart
    hashTag: List[str]
    version: int = 2


class HealthResponse(BaseModel):
    """헬스 체크 응답 스키마"""
    status: str = Field(..., description="시스템 상태")
    model_loaded: bool = Field(..., description="모델 로드 상태")
    gpu_available: bool = Field(default=False, description="GPU 사용 가능 여부")