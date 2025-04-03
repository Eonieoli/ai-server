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
    composition: ScoreComment
    sharpness: ScoreComment
    noise: ScoreComment
    exposure: ScoreComment
    color_harmony: ScoreComment
    aesthetics: ScoreComment
    overall: ScoreComment


class HealthResponse(BaseModel):
    """헬스 체크 응답 스키마"""
    status: str
    model_loaded: bool
    gpu_available: bool = Field(default=False)
