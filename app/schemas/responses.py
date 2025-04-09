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
    미적감각: str = Field(alias="미적 감각")


class AnalysisChart(BaseModel):
    """분석 차트 데이터"""
    구도: int
    선명도: int
    주제: int
    노출: int
    색감: int
    미적감각: int = Field(alias="미적 감각")


class ImageAnalysisResponse(BaseModel):
    """이미지 분석 응답 스키마"""
    score: int = Field(..., description="종합 점수")
    comment: str = Field(..., description="종합 평가 코멘트")
    analysisText: AnalysisText = Field(..., description="각 항목별 평가 텍스트")
    analysisChart: AnalysisChart = Field(..., description="각 항목별 평가 점수")
    hashTag: List[str] = Field(..., description="관련 해시태그")


class HealthResponse(BaseModel):
    """헬스 체크 응답 스키마"""
    status: str = Field(..., description="시스템 상태")
    model_loaded: bool = Field(..., description="모델 로드 상태")
    gpu_available: bool = Field(default=False, description="GPU 사용 가능 여부")