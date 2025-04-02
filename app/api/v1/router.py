"""
API 라우터 설정
"""
from fastapi import APIRouter

from app.api.v1.endpoints import health, analysis

# API v1 라우터
api_router = APIRouter()

# 엔드포인트 라우터 등록
api_router.include_router(health.router, prefix="/system")
api_router.include_router(analysis.router, prefix="/image")
