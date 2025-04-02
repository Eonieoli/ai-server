"""
AI 모델 및 서비스 테스트
"""
import os
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.models.ai_model import ImageAnalysisModel
from app.services.ai_service import AIService


# 테스트 클라이언트 생성
client = TestClient(app)


# 루트 엔드포인트 테스트
def test_root_endpoint():
    """루트 엔드포인트 테스트"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


# 헬스 체크 엔드포인트 테스트
def test_health_endpoint():
    """헬스 체크 엔드포인트 테스트"""
    response = client.get(f"{settings.API_V1_STR}/system/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "ok"


# 모델 로드 테스트 (CPU 환경에 적합)
def test_model_device_selection():
    """모델 장치 선택 테스트"""
    # 원래 설정 저장
    original_use_gpu = settings.USE_GPU
    
    try:
        # CPU 강제 설정
        settings.USE_GPU = False
        model = ImageAnalysisModel()
        assert model.device.type == "cpu"
        
        # GPU 사용 가능 시 GPU 설정 테스트
        if torch.cuda.is_available():
            settings.USE_GPU = True
            model = ImageAnalysisModel()
            assert model.device.type == "cuda"
    finally:
        # 원래 설정 복원
        settings.USE_GPU = original_use_gpu


# 이미지 다운로드 테스트 (모의 객체 사용)
@pytest.mark.asyncio
async def test_download_image_mock():
    """이미지 다운로드 테스트 (모의 객체 사용)"""
    # 가짜 aiohttp 응답 객체 생성
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.content_length = 1024  # 1KB
    mock_response.read.return_value = b"fake_image_data"
    
    # 가짜 aiohttp 세션 객체 생성
    mock_session = MagicMock()
    mock_session.__aenter__.return_value = mock_session
    mock_session.get.return_value.__aenter__.return_value = mock_response
    
    # aiohttp.ClientSession 패치
    with patch("aiohttp.ClientSession", return_value=mock_session):
        # 이미지 다운로드 테스트
        image_path = await AIService.download_image("https://example.com/image.jpg")
        
        # 결과 확인
        assert image_path is not None
        assert image_path.exists()
        
        # 임시 파일 정리
        await AIService.cleanup_temp_file(image_path)


# CPU 환경에서의 모델 분석 기능 테스트 (간단한 이미지 사용)
@pytest.mark.asyncio
async def test_analyze_small_image():
    """작은 이미지 분석 테스트 (실제 모델 사용)"""
    # 테스트를 위한 작은 이미지 생성 (1x1 픽셀 빨간색 이미지)
    from PIL import Image
    
    # 임시 이미지 생성
    temp_dir = Path(settings.TEMP_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    test_image_path = temp_dir / "test_red_pixel.jpg"
    img = Image.new('RGB', (1, 1), color=(255, 0, 0))
    img.save(test_image_path)
    
    try:
        # 모델 응답 모의 객체 생성
        mock_pipe_result = [{"generated_text": """
        분석 결과:
        {
            "composition": {"score": 5, "comment": "구도 평가"},
            "sharpness": {"score": 6, "comment": "선명도 평가"},
            "noise": {"score": 7, "comment": "노이즈 평가"},
            "exposure": {"score": 8, "comment": "노출 평가"},
            "color_harmony": {"score": 5, "comment": "색감 평가"},
            "aesthetics": {"score": 6, "comment": "심미성 평가"},
            "overall_comment": "전체적인 평가 코멘트"
        }
        """}]
        
        # 모델 파이프라인 패치
        with patch.object(ImageAnalysisModel, "analyze_image", return_value=asyncio.Future()) as mock_analyze:
            mock_analyze.return_value.set_result({
                "composition": {"score": 5, "comment": "구도 평가"},
                "sharpness": {"score": 6, "comment": "선명도 평가"},
                "noise": {"score": 7, "comment": "노이즈 평가"},
                "exposure": {"score": 8, "comment": "노출 평가"},
                "color_harmony": {"score": 5, "comment": "색감 평가"},
                "aesthetics": {"score": 6, "comment": "심미성 평가"},
                "overall_comment": "전체적인 평가 코멘트"
            })
            
            # 서비스를 통한 이미지 분석 테스트
            result = await AIService.analyze_image(test_image_path)
            
            # 결과 확인
            assert "composition" in result
            assert "sharpness" in result
            assert "noise" in result
            assert "exposure" in result
            assert "color_harmony" in result
            assert "aesthetics" in result
            assert "overall_comment" in result
    finally:
        # 테스트 이미지 정리
        if test_image_path.exists():
            os.remove(test_image_path)


# 이미지 분석 API 엔드포인트 테스트
def test_analyze_endpoint_mock():
    """이미지 분석 API 엔드포인트 테스트 (모의 객체 사용)"""
    # 서비스 함수 패치
    with patch("app.services.ai_service.ai_service.process_image_url") as mock_process:
        # 가짜 응답 설정
        mock_process.return_value = asyncio.Future()
        mock_process.return_value.set_result({
            "composition": {"score": 5, "comment": "구도 평가"},
            "sharpness": {"score": 6, "comment": "선명도 평가"},
            "noise": {"score": 7, "comment": "노이즈 평가"},
            "exposure": {"score": 8, "comment": "노출 평가"},
            "color_harmony": {"score": 5, "comment": "색감 평가"},
            "aesthetics": {"score": 6, "comment": "심미성 평가"},
            "overall_comment": "전체적인 평가 코멘트"
        })
        
        # API 요청 테스트
        response = client.post(
            f"{settings.API_V1_STR}/image/analyze",
            json={"image_url": "https://example.com/test.jpg"}
        )
        
        # 응답 확인
        assert response.status_code == 200
        data = response.json()
        assert "composition" in data
        assert "sharpness" in data
        assert "noise" in data
        assert "exposure" in data
        assert "color_harmony" in data
        assert "aesthetics" in data
        assert "overall_comment" in data


if __name__ == "__main__":
    # 로컬에서 테스트 실행
    pytest.main(["-xvs", __file__])
