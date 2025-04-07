"""
응답에 대한 Pydantic 스키마 정의
"""
from typing import Dict, Any, List
from pydantic import BaseModel, Field


class ScoreComment(BaseModel):
    """점수와 코멘트 스키마"""
    score: int
    comment: str


class AnalysisText(BaseModel):
    """분석 텍스트"""
    구도: str
    선명도: str
    노이즈: str
    노출: str
    색감: str
    심미성: str


class AnalysisChart(BaseModel):
    """분석 차트 데이터"""
    구도: int
    선명도: int
    노이즈: int
    노출: int
    색감: int
    심미성: int


class ImageAnalysisResponse(BaseModel):
    """이미지 분석 응답 스키마"""
    score: int
    comment: str
    analysisText: AnalysisText
    analysisChart: AnalysisChart
    hashTag: List[str]


class HealthResponse(BaseModel):
    """헬스 체크 응답 스키마"""
    status: str
    model_loaded: bool
    gpu_available: bool = Field(default=False)
