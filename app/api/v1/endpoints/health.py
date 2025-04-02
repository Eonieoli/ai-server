"""
시스템 상태 확인 엔드포인트
"""
from fastapi import APIRouter, Depends

from app.models.ai_model import model_instance
from app.schemas.responses import HealthResponse
from app.utils.helper import get_system_info

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["시스템"])
async def health_check():
    """
    시스템 상태 확인 API
    
    Returns:
        HealthResponse: 시스템 상태 정보
    """
    system_info = get_system_info()
    
    return HealthResponse(
        status="ok",
        model_loaded=model_instance.model_loaded,
        gpu_available=system_info["gpu_available"]
    )
