"""
FastAPI 애플리케이션 메인 모듈
"""
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.api.v1.router import api_router
from app.utils.helper import setup_logging, cleanup_temp_files
from app.models.ai_model import model_instance

# 로깅 설정
setup_logging()
logger = logging.getLogger(__name__)

# FastAPI 애플리케이션 생성
app = FastAPI(
    title=settings.APP_NAME,
    description="이미지 분석 AI 서버 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    debug=settings.DEBUG
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 구체적인 출처 목록으로 제한해야 함
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 이벤트 핸들러"""
    logger.info("Starting AI Analysis Server...")
    
    # 임시 파일 정리
    cleanup_temp_files()
    
    # 모델 사전 로드 (선택적)
    # 실제 요청이 오기 전에 모델을 로드해 두면 첫 번째 요청의 지연을 줄일 수 있음
    # 하지만 리소스를 많이 사용하므로 선택적으로 사용
    try:
        logger.info("Preloading AI model...")
        if model_instance.load_model():
            logger.info("AI model loaded successfully")
        else:
            logger.warning("Failed to preload AI model. Will try again on first request.")
    except Exception as e:
        logger.error(f"Error preloading model: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 이벤트 핸들러"""
    logger.info("Shutting down AI Analysis Server...")
    
    # 모델 해제
    if model_instance.model_loaded:
        model_instance.unload_model()
    
    # 임시 파일 정리
    cleanup_temp_files(max_age_hours=0)  # 모든 임시 파일 정리


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """전역 예외 핸들러"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"}
    )


# 루트 엔드포인트
@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {
        "message": "Welcome to AI Image Analysis Server",
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/system/health"
    }


# 직접 실행 시
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
