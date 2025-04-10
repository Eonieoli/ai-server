"""
FastAPI 서버 실행 스크립트 (Docker 없이 직접 실행)
"""
import os
import sys
import platform
import uvicorn
import torch
from pathlib import Path

def check_environment():
    """실행 환경을 확인하고 출력합니다."""
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # PyTorch 정보
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU model: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # bitsandbytes 확인
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes version: {bnb.__version__}")
        print("4-bit quantization is available")
    except ImportError:
        print("WARNING: bitsandbytes not installed. 4-bit quantization will not be available.")
        print("Please install: pip install bitsandbytes>=0.41.0")

    # 디렉토리 확인
    root_dir = Path(__file__).resolve().parent
    print(f"Working directory: {root_dir}")
    
    # 필요한 디렉토리 생성
    cache_dir = root_dir / "models" / "downloads"
    temp_dir = root_dir / "temp"
    logs_dir = root_dir / "logs"
    
    for dir_path in [cache_dir, temp_dir, logs_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created/verified directory: {dir_path}")

if __name__ == "__main__":
    # 환경 확인
    check_environment()
    
    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 원하는 GPU 인덱스로 변경하세요

    # 서버 설정
    host = "0.0.0.0"  # 모든 네트워크 인터페이스에서 접근 가능
    port = 8000        # 포트 번호

    print(f"\nStarting AI server on {host}:{port}")
    print(f"Using GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print("Server will start with llava-1.5-7b-hf model")

    # 서버 실행
    uvicorn.run("app.main:app", host=host, port=port, workers=1)