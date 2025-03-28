# app/models/ai_model.py
import os
import torch
from typing import Dict, List, Any, Optional
import numpy as np
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from app.core.config import settings
import json
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 변수로 모델과 프로세서 선언
_model = None
_processor = None


class LlavaModel:
    """LLaVA 모델을 관리하는 클래스"""
    
    def __init__(self, model_name_or_path: str = None):
        """
        LLaVA 모델 초기화
        
        Args:
            model_name_or_path: 모델 경로 또는 Hugging Face 모델 ID
        """
        self.model_name = model_name_or_path or settings.MODEL_NAME
        self.device = settings.MODEL_DEVICE
        self.model = None
        self.processor = None
        
    def load(self):
        """모델 및 프로세서 로드"""
        logger.info(f"Loading LLaVA model: {self.model_name}")
        try:
            # 모델 로드
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            # 프로세서 로드
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # GPU로 모델 이동 (가능한 경우)
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
                logger.info("Model loaded on CUDA")
            else:
                logger.info("Model loaded on CPU")
                
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load LLaVA model: {str(e)}")
    
    def predict(self, image, prompt=None):
        """
        이미지에 대한 예측 수행
        
        Args:
            image: PIL 이미지
            prompt: 선택적 텍스트 프롬프트
        
        Returns:
            생성된 텍스트
        """
        if prompt is None:
            prompt = "사진을 분석하고 품질을 평가해주세요."
            
        # 입력 처리
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        
        # GPU로 입력 이동 (필요한 경우)
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # 모델 추론
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
        
        # 결과 디코딩
        generated_text = self.processor.decode(output[0], skip_special_tokens=True)
        
        # 프롬프트 제거하여 실제 생성된 텍스트만 반환
        if prompt in generated_text:
            generated_text = generated_text.split(prompt)[1].strip()
        
        return generated_text
    
    def evaluate_quality(self, image, criteria=None):
        """
        이미지 품질 평가
        
        Args:
            image: PIL 이미지
            criteria: 평가 기준 리스트
            
        Returns:
            각 기준별 점수가 포함된 딕셔너리
        """
        if criteria is None:
            criteria = ["composition", "sharpness", "noise", "exposure", "color", "aesthetics"]
            
        criteria_str = ", ".join(criteria)
        
        prompt = f"""
        이 사진을 다음 기준에 따라 평가해주세요: {criteria_str}. 
        각 기준에 대해 1-10 사이의 점수를 매겨주세요. 
        JSON 형식으로 결과를 제공해주세요. 
        예: {{"composition": 8, "sharpness": 7, ...}}. 
        JSON 외에 다른 텍스트는 포함하지 마세요.
        """
        
        # 모델에서 텍스트 생성
        result_text = self.predict(image, prompt)
        
        try:
            # JSON 추출 시도
            # {...} 형태의 패턴 찾기
            import re
            json_pattern = r'\{.*\}'
            json_match = re.search(json_pattern, result_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                scores = json.loads(json_str)
                
                # 점수 정규화 (1-10 범위 확인)
                for key, value in scores.items():
                    if isinstance(value, (int, float)):
                        scores[key] = max(1, min(10, value))
                    else:
                        # 숫자가 아닌 경우 기본값 설정
                        scores[key] = 5
                        
                return scores
            else:
                # JSON이 추출되지 않았을 경우 기본값 설정
                logger.warning("Failed to extract JSON from model output")
                return {criterion: 5 for criterion in criteria}
                
        except Exception as e:
            logger.error(f"Error parsing evaluation result: {str(e)}")
            # 오류 시 기본값 반환
            return {criterion: 5 for criterion in criteria}

def load_model():
    """전역 모델 인스턴스 로드"""
    global _model, _processor
    
    model_instance = LlavaModel(settings.MODEL_NAME)
    model_instance.load()
    
    _model = model_instance.model
    _processor = model_instance.processor
    
    return model_instance


def get_model():
    """현재 로드된 모델 인스턴스 반환"""
    if _model is None or _processor is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    # 모델 래퍼 생성
    model_wrapper = LlavaModel(settings.MODEL_NAME)
    model_wrapper.model = _model
    model_wrapper.processor = _processor
    
    return model_wrapper


def cleanup_model():
    """전역 모델 인스턴스 정리"""
    global _model, _processor
    
    if _model is not None:
        # CUDA 메모리 정리
        if settings.MODEL_DEVICE == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        _model = None
        _processor = None
        
        logger.info("Global model resources cleaned up")