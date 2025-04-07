"""
응답에 대한 Pydantic 스키마 정의
"""
from typing import Dict, Any
from pydantic import BaseModel, Field


class ScoreComment(BaseModel):
    """점수와 코멘트 스키마"""
    score: int
    comment: str


class ImageAnalysisResponse(BaseModel):
    """이미지 분석 응답 스키마"""
    구도: ScoreComment
    선명도: ScoreComment
    노이즈: ScoreComment
    노출: ScoreComment
    색감: ScoreComment
    심미성: ScoreComment
    종합평가: ScoreComment


class HealthResponse(BaseModel):
    """헬스 체크 응답 스키마"""
    status: str
    model_loaded: bool
    gpu_available: bool = Field(default=False)
