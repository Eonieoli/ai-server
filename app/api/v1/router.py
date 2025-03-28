# app/api/v1/router.py
from fastapi import APIRouter

# 각 엔드포인트 라우터 임포트
from app.api.v1.endpoints.analysis import router as analysis_router
from app.api.v1.endpoints.health import router as health_router

# 메인 API 라우터 생성
api_router = APIRouter()

# 각 엔드포인트별 라우터 등록
api_router.include_router(analysis_router, prefix="/analysis", tags=["image-analysis"])
api_router.include_router(health_router, prefix="/health", tags=["health"])