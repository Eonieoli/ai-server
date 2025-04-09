"""
FastAPI 서버 실행 스크립트 (ngrok 연동)
"""
import os
import sys
import platform
import uvicorn
import torch
from pathlib import Path
from pyngrok import ngrok

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

def run_with_ngrok(port=8000):
    """ngrok 터널을 생성하고 서버를 실행합니다."""
    # ngrok 터널 생성
    public_url = ngrok.connect(port).public_url
    print(f"\nngrok 터널이 생성되었습니다: {public_url}")
    print(f"API 문서: {public_url}/docs")
    
    # 서버 실행
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, workers=1)

if __name__ == "__main__":
    # 환경 확인
    check_environment()
    
    # GPU 설정 (Colab에서는 하나만 있음)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 서버 설정
    port = 8000

    print(f"\nStarting AI server with ngrok tunnel")
    print(f"Using GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print("Server will start with 8-bit quantized LLaVA-NeXT-Video-7B model")

    # ngrok으로 실행
    run_with_ngrok(port)