# app/services/ai_service.py
import asyncio
from typing import Dict, Any, List, Optional
import time
import os
import traceback

from app.models.ai_model import get_model
from app.utils.helper import read_image, preprocess_image


async def process_image(image_path: str, detailed: bool = False) -> Dict[str, Any]:
    """
    이미지를 AI 모델로 처리하여 분석 결과를 반환합니다.
    
    Args:
        image_path: 이미지 파일 경로
        detailed: 상세 결과 반환 여부
        
    Returns:
        분석 결과를 담은 딕셔너리
    """
    try:
        # 이미지 읽기
        image = read_image(image_path)
        if image is None:
            raise ValueError("이미지를 읽을 수 없습니다")
        
        # 이미지 전처리
        processed_image = preprocess_image(image)
        
        # 모델 가져오기
        model = get_model()
        
        # 비동기적으로 모델 추론 실행 (CPU/GPU 집약적 작업을 비동기로 처리)
        loop = asyncio.get_event_loop()
        prompt = "이 사진을 상세하게 분석해주세요. 구도, 색감, 선명도, 노이즈, 노출, 심미성 등에 대해 언급해주세요."
        prediction = await loop.run_in_executor(None, lambda: model.predict(processed_image, prompt))
        
        # 결과 포맷팅
        result = {
            "success": True,
            "timestamp": time.time(),
            "analysis": prediction
        }
        
        # 상세 분석 결과 추가 (옵션)
        if detailed:
            # 품질 평가 수행
            quality_scores = await loop.run_in_executor(None, lambda: model.evaluate_quality(processed_image))
            result["quality_scores"] = quality_scores
            
            # 평균 점수 계산
            if quality_scores:
                avg_score = sum(quality_scores.values()) / len(quality_scores)
                result["average_score"] = round(avg_score, 2)
            
            # 메타데이터 추가
            result["metadata"] = {
                "dimensions": f"{image.width}x{image.height}",
                "format": image.format if hasattr(image, 'format') else "Unknown",
                "file_size": os.path.getsize(image_path) // 1024  # KB 단위
            }
        
        return result
    
    except Exception as e:
        print(f"이미지 처리 오류: {str(e)}")
        traceback.print_exc()
        raise ValueError(f"이미지 처리 중 오류 발생: {str(e)}")


async def evaluate_image(
    image_path: str, 
    criteria: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    이미지의 품질을 평가합니다.
    
    Args:
        image_path: 이미지 파일 경로
        criteria: 평가 기준 목록 (None인 경우 모든 기준 사용)
        
    Returns:
        평가 결과를 담은 딕셔너리
    """
    try:
        # 이미지 읽기
        image = read_image(image_path)
        if image is None:
            raise ValueError("이미지를 읽을 수 없습니다")
        
        # 이미지 전처리
        processed_image = preprocess_image(image)
        
        # 모델 가져오기
        model = get_model()
        
        # 비동기적으로 모델 평가 실행
        loop = asyncio.get_event_loop()
        evaluation = await loop.run_in_executor(
            None, 
            lambda: model.evaluate_quality(processed_image, criteria=criteria)
        )
        
        # 종합 점수 계산 (평가 요소들의 평균)
        overall_score = sum(evaluation.values()) / len(evaluation) if evaluation else 0
        
        # 설명 생성을 위한 프롬프트
        feedback_prompt = f"""
        이 사진에 대한 품질 점수를 매겼습니다:
        {', '.join([f'{k}: {v}' for k, v in evaluation.items()])}
        
        이 점수를 바탕으로 사진의 장점과 개선점에 대해 상세한 피드백을 제공해주세요.
        각 평가 항목(구도, 선명도, 노이즈, 노출, 색감, 심미성)에 대해 1-2문장씩 설명해주세요.
        """
        
        # 피드백 생성
        feedback = await loop.run_in_executor(
            None,
            lambda: model.predict(processed_image, feedback_prompt)
        )
        
        # 결과 포맷팅
        result = {
            "success": True,
            "timestamp": time.time(),
            "overall_score": round(overall_score, 2),
            "criteria_scores": evaluation,
            "feedback": feedback,
            "evaluated_criteria": list(evaluation.keys()) if evaluation else []
        }
        
        return result
    
    except Exception as e:
        print(f"이미지 평가 오류: {str(e)}")
        traceback.print_exc()
        raise ValueError(f"이미지 평가 중 오류 발생: {str(e)}")