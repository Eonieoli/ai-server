"""
이미지 분석 API 엔드포인트
"""
import logging
from fastapi import APIRouter, HTTPException, Depends

from app.schemas.requests import ImageAnalysisRequest
from app.schemas.responses import ImageAnalysisResponse
from app.services.ai_service import ai_service
from app.models.ai_model import model_instance

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analyze", response_model=ImageAnalysisResponse, tags=["분석"])
async def analyze_image(request: ImageAnalysisRequest):
    """
    이미지 분석 API
    
    Args:
        request (ImageAnalysisRequest): 이미지 URL이 포함된 요청
        
    Returns:
        ImageAnalysisResponse: 이미지 분석 결과
        
    Raises:
        HTTPException: 처리 중 오류 발생 시
    """
    # 모델 로드 확인
    if not model_instance.model_loaded:
        try:
            logger.info("Model not loaded. Loading model...")
            if not model_instance.load_model():
                raise HTTPException(
                    status_code=503,
                    detail="Failed to load AI model. Service unavailable."
                )
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise HTTPException(
                status_code=503,
                detail="Failed to load AI model. Service unavailable."
            )
    
    try:
        # 이미지 URL 처리 및 분석
        result = await ai_service.process_image_url(str(request.image_url))
        
        # 응답 생성
        return ImageAnalysisResponse(
            composition=result["composition"],
            sharpness=result["sharpness"],
            noise=result["noise"],
            exposure=result["exposure"],
            color_harmony=result["color_harmony"],
            aesthetics=result["aesthetics"],
            overall_comment=result["overall_comment"]
        )
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {str(e)}"
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the image"
        )
