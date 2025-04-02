"""
응답에 대한 Pydantic 스키마 정의
"""
from pydantic import BaseModel, Field


class EvaluationDetail(BaseModel):
    """평가 세부 정보 스키마"""
    score: int = Field(..., ge=1, le=10, description="1부터 10까지의 점수")
    comment: str = Field(..., description="평가에 대한 간략한 설명")


class ImageAnalysisResponse(BaseModel):
    """이미지 분석 응답 스키마"""
    composition: EvaluationDetail
    sharpness: EvaluationDetail
    noise: EvaluationDetail
    exposure: EvaluationDetail
    color_harmony: EvaluationDetail
    aesthetics: EvaluationDetail
    overall_comment: str = Field(..., description="전체적인 평가 코멘트")


class HealthResponse(BaseModel):
    """헬스 체크 응답 스키마"""
    status: str
    model_loaded: bool
    gpu_available: bool = Field(default=False)
