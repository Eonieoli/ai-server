"""
FastAPI 서버 실행 스크립트
"""
import os
import uvicorn

if __name__ == "__main__":
    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 3번 GPU 사용

    # 서버 설정
    host = "0.0.0.0"  # 모든 네트워크 인터페이스에서 접근 가능
    port = 8000        # 포트 번호

    print(f"Starting AI server on {host}:{port}")
    print(f"Using GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

    # 서버 실행
    uvicorn.run("app.main:app", host=host, port=port, workers=1)