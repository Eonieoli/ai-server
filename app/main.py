# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager

# 내부 모듈 임포트
from app.api.v1.router import api_router
from app.core.config import settings
from app.models.ai_model import load_model, cleanup_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 시작 시 AI 모델을 로드하고 종료 시 정리하는 lifespan 이벤트 핸들러
    """
    # 시작 시 모델 로드
    print("애플리케이션 시작: AI 모델 로딩 중...")
    load_model()
    print("AI 모델 로딩 완료!")
    
    yield  # 애플리케이션 실행
    
    # 종료 시 모델 정리
    print("애플리케이션 종료: 리소스 정리 중...")
    cleanup_model()
    print("정리 완료!")


def create_app() -> FastAPI:
    """
    FastAPI 애플리케이션 인스턴스를 생성하고 설정합니다.
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version=settings.API_VERSION,
        lifespan=lifespan,
    )
    
    # CORS 미들웨어 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # API 라우터 등록
    app.include_router(api_router, prefix=settings.API_PREFIX)
    
    return app


app = create_app()


if __name__ == "__main__":
    """
    직접 실행될 때 서버를 시작합니다.
    """
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG_MODE
    )