"""
AI 모델 정의 및 로딩 관련 코드
"""
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional

from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq, 
    AutoConfig,
    pipeline, 
    LlavaNextVideoProcessor
)

from app.core.config import settings

logger = logging.getLogger(__name__)


class ImageAnalysisModel:
    """이미지 분석 모델 클래스"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.pipe = None
        self.device = self._get_device()
        self.model_loaded = False
    
    def _get_device(self) -> torch.device:
        """
        사용할 장치(CPU/GPU) 결정
        
        Returns:
            torch.device: 사용할 장치
        """
        if settings.USE_GPU and torch.cuda.is_available():
            return torch.device(f"cuda:{settings.GPU_DEVICE}")
        return torch.device("cpu")
    
    def is_gpu_available(self) -> bool:
        """
        GPU 사용 가능 여부 확인
        
        Returns:
            bool: GPU 사용 가능 여부
        """
        return torch.cuda.is_available()
    
    def load_model(self):
        """모델 로드 함수"""
        try:
            logger.info(f"Loading model {settings.MODEL_NAME} on {self.device}")
            
            # 모델 캐시 디렉토리 생성
            Path(settings.MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
            
            # CPU에서 로드할 때는 더 적은 리소스 사용하도록 설정
            torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            
            # LLaVA-NeXT-Video 모델 설정 확인
            config = AutoConfig.from_pretrained(
                settings.MODEL_NAME,
                cache_dir=settings.MODEL_CACHE_DIR
            )
            
            # 모델 유형에 따라 적절한 클래스 선택
            try:
                # 멀티모달 비전-언어 모델 로드 시도
                self.model = AutoModelForVision2Seq.from_pretrained(
                    settings.MODEL_NAME,
                    torch_dtype=torch_dtype,
                    device_map=self.device if self.device.type == "cuda" else None,
                    cache_dir=settings.MODEL_CACHE_DIR,
                    low_cpu_mem_usage=True if self.device.type == "cpu" else False
                )
                logger.info("Model loaded using AutoModelForVision2Seq")
            except Exception as e:
                logger.warning(f"Failed to load with AutoModelForVision2Seq: {e}")
                # 일반 비전-언어 모델 로드 시도
                from transformers import LlavaNextVideoForConditionalGeneration
                self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                    settings.MODEL_NAME,
                    torch_dtype=torch_dtype,
                    device_map=self.device if self.device.type == "cuda" else None,
                    cache_dir=settings.MODEL_CACHE_DIR,
                    low_cpu_mem_usage=True if self.device.type == "cpu" else False
                )
                logger.info("Model loaded using LlavaNextVideoForConditionalGeneration")
            
            # 프로세서 로드
            try:
                self.processor = LlavaNextVideoProcessor.from_pretrained(
                    settings.MODEL_NAME,
                    cache_dir=settings.MODEL_CACHE_DIR
                )
                logger.info("Processor loaded using LlavaNextVideoProcessor")
            except Exception as e:
                logger.warning(f"Failed to load with LlavaNextVideoProcessor: {e}")
                self.processor = AutoProcessor.from_pretrained(
                    settings.MODEL_NAME,
                    cache_dir=settings.MODEL_CACHE_DIR
                )
                logger.info("Processor loaded using AutoProcessor")
            
            # 직접 추론 함수 정의
            self.pipe = self._create_custom_pipeline()
            
            self.model_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model_loaded = False
            return False
    
    def _create_custom_pipeline(self):
        """
        모델에 맞춘 커스텀 파이프라인 함수 생성
        
        Returns:
            callable: 이미지와 프롬프트를 처리하는 함수
        """
        def inference_fn(images, prompt, generate_kwargs=None):
            if generate_kwargs is None:
                generate_kwargs = {"max_new_tokens": 512, "temperature": 0.1}
            
            try:
                # 입력 준비
                inputs = self.processor(text=prompt, images=images, return_tensors="pt")
                
                # 디바이스로 이동
                inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
                
                # 생성
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **generate_kwargs
                    )
                
                # 디코딩
                generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                
                return [{"generated_text": generated_text}]
            except Exception as e:
                logger.error(f"Error in custom pipeline: {e}")
                return [{"generated_text": f"오류가 발생했습니다: {str(e)}"}]
        
        return inference_fn
    
    def unload_model(self):
        """모델 메모리에서 해제"""
        if self.model:
            del self.model
            del self.processor
            del self.pipe
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            self.model = None
            self.processor = None
            self.pipe = None
            self.model_loaded = False
            
            logger.info("Model unloaded from memory")
    
    async def analyze_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        이미지 분석 함수
        
        Args:
            image_path (Union[str, Path]): 분석할 이미지 경로
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        if not self.model_loaded:
            logger.error("Model not loaded. Cannot analyze image.")
            raise RuntimeError("Model not loaded. Cannot analyze image.")
        
        try:
            logger.info(f"Analyzing image: {image_path}")
            
            # 이미지 경로 확인
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # 모델에 프롬프트와 이미지 전달
            result = self.pipe(
                images=str(image_path),
                prompt=settings.PROMPT_TEMPLATE,
                generate_kwargs={"temperature": 0.1, "max_new_tokens": 512}
            )
            
            # 응답 텍스트 추출
            response_text = result[0]["generated_text"] if isinstance(result, list) and result else ""
            
            # JSON 부분 추출
            try:
                # JSON 부분만 추출 (중괄호 시작부터 끝까지)
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    analysis_result = json.loads(json_str)
                else:
                    # JSON을 찾을 수 없는 경우 기본 응답 생성
                    logger.warning("JSON not found in model response. Using default response.")
                    analysis_result = self._create_default_response()
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from model response")
                analysis_result = self._create_default_response()
            
            # 결과 유효성 검사 및 포맷 맞추기
            return self._validate_and_format_result(analysis_result)
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise
    
    def _create_default_response(self) -> Dict[str, Any]:
        """기본 응답 생성"""
        return {
            category: {"score": 5, "comment": "평가 실패"} 
            for category in settings.EVALUATION_CATEGORIES
        }
    
    def _validate_and_format_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        결과 유효성 검사 및 포맷 맞추기
        
        Args:
            result (Dict[str, Any]): 원본 결과
            
        Returns:
            Dict[str, Any]: 검증 및 포맷팅된 결과
        """
        # 결과가 모든 카테고리를 포함하는지 확인
        formatted_result = {}
        
        for category in settings.EVALUATION_CATEGORIES:
            if category in result and isinstance(result[category], dict):
                cat_result = result[category]
                
                # 점수 확인 및 조정
                score = cat_result.get("score", 5)
                if not isinstance(score, int) or score < 1 or score > 10:
                    score = max(1, min(10, int(score) if isinstance(score, (int, float)) else 5))
                
                # 코멘트 확인
                comment = cat_result.get("comment", "")
                if not isinstance(comment, str):
                    comment = str(comment)
                
                formatted_result[category] = {
                    "score": score,
                    "comment": comment
                }
            else:
                formatted_result[category] = {
                    "score": 5,
                    "comment": "평가 정보가 누락되었습니다."
                }
        
        # 전체 코멘트 확인
        overall_comment = result.get("overall_comment", "")
        if not isinstance(overall_comment, str):
            overall_comment = "이미지에 대한 전체적인 평가 정보를 제공하지 못했습니다."
        
        formatted_result["overall_comment"] = overall_comment
        
        return formatted_result


# 싱글턴 인스턴스
model_instance = ImageAnalysisModel()
