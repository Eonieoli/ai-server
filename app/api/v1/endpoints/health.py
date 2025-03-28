# app/api/v1/endpoints/health.py
from fastapi import APIRouter
import time
import torch
from typing import Dict, Any

from app.core.config import settings

router = APIRouter()


@router.get("")
async def health_check() -> Dict[str, Any]:
    """
    기본 헬스 체크 엔드포인트
    """
    return {
        "status": "ok",
        "timestamp": time.time(),
        "api_version": settings.API_VERSION,
        "service": settings.PROJECT_NAME
    }


@router.get("/model")
async def model_health() -> Dict[str, Any]:
    """
    모델 상태 체크 엔드포인트
    """
    # 모델 정보
    model_info = {
        "name": settings.MODEL_NAME,
        "device": settings.MODEL_DEVICE,
    }
    
    # GPU 정보 (가능한 경우)
    if torch.cuda.is_available():
        model_info["cuda"] = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "N/A"
        }
    else:
        model_info["cuda"] = {"available": False}
    
    return {
        "status": "ok",
        "timestamp": time.time(),
        "model": model_info,
        "api_version": settings.API_VERSION
    }