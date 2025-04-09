"""
AI 모델 정의 및 로딩 관련 코드
"""
import json
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, Optional, List, Tuple
from PIL import Image

from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration, BitsAndBytesConfig

from app.core.config import settings
from app.utils.translator import translate_dict_values, translate_batch
from app.utils.helper import scale_score

logger = logging.getLogger(__name__)


class ImageAnalysisModel:
    """이미지 분석 모델 클래스"""
    
    def __init__(self):
        self.model = None
        self.processor = None
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
    
    def load_model(self) -> bool:
        """
        모델 로드 함수 (8비트 양자화 적용)
        
        Returns:
            bool: 모델 로드 성공 여부
        """
        try:
            logger.info(f"Loading model {settings.MODEL_NAME} with 8-bit quantization on {self.device}")
            
            # 모델 캐시 디렉토리 생성
            Path(settings.MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
            
            # 8비트 양자화 설정
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,           # 8비트 양자화 적용
                llm_int8_threshold=6.0,      # 임계값 설정
                llm_int8_has_fp16_weight=False,  # 완전 int8 변환
                llm_int8_skip_modules=None,   # 양자화를 스킵할 모듈 없음
            )
            
            # 모델 로드 (8비트 양자화 적용)
            self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                settings.MODEL_NAME,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                cache_dir=settings.MODEL_CACHE_DIR,
                trust_remote_code=True,
            )
            
            logger.info("Model loaded with 8-bit quantization")
            
            # 프로세서 로드
            self.processor = LlavaNextVideoProcessor.from_pretrained(
                settings.MODEL_NAME,
                cache_dir=settings.MODEL_CACHE_DIR,
                trust_remote_code=True
            )
            logger.info("Processor loaded successfully")
            
            self.model_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model_loaded = False
            return False
    
    def unload_model(self) -> None:
        """모델 메모리에서 해제"""
        if self.model:
            del self.model
            del self.processor
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            self.model = None
            self.processor = None
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
            
            # 이미지 로드 및 검증
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # 이미지 처리
            image = Image.open(image_path).convert("RGB")
            generated_text = self._generate_model_response(image)
            
            # 응답 처리
            analysis_result = self._parse_model_response(generated_text)
            
            # 최종 결과 형식화
            return self._format_analysis_result(analysis_result)
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return self._create_default_response()
    
    def _generate_model_response(self, image: Image.Image) -> str:
        """
        모델에서 응답 생성
        
        Args:
            image (Image.Image): 처리할 이미지
            
        Returns:
            str: 모델이 생성한 텍스트
        """
        try:
            # Chat template 사용하여 프롬프트 생성
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": settings.PROMPT_TEMPLATE},
                        {"type": "image"},
                    ],
                },
            ]
            
            prompt = self.processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True
            )
            
            # 이미지를 numpy 배열로 변환
            image_np = np.array(image)
            
            # 입력 준비
            inputs = self.processor(
                text=prompt, 
                images=image_np, 
                padding=True, 
                return_tensors="pt"
            ).to(self.device)
            
            # 생성
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=512, 
                    do_sample=True,
                    temperature=0.1
                )
            
            # 응답 디코딩
            generated_text = self.processor.decode(
                output[0][2:], 
                skip_special_tokens=True
            )
            
            # 로그에 생성된 텍스트 기록 (긴 텍스트는 잘라서 기록)
            log_text = generated_text[:500] + "..." if len(generated_text) > 500 else generated_text
            logger.info(f"Generated text sample: {log_text}")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating model response: {e}")
            raise
    
    def _parse_model_response(self, text: str) -> Dict[str, Any]:
        """
        모델 응답에서 JSON 추출
        
        Args:
            text (str): 모델이 생성한 텍스트
            
        Returns:
            Dict[str, Any]: 파싱된 JSON 결과
        """
        try:
            # 마크다운 코드 블록에서 JSON 추출
            if "```json" in text and "```" in text.split("```json", 1)[1]:
                json_content = text.split("```json", 1)[1].split("```", 1)[0].strip()
                return json.loads(json_content)
            
            # 일반적인 JSON 추출 시도
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                return json.loads(json_str)
            
            # JSON을 찾을 수 없는 경우
            logger.warning("JSON not found in model response. Using default response.")
            return {}
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from model response: {e}")
            return {}
    
    def _create_default_response(self) -> Dict[str, Any]:
        """
        기본 응답 생성
        
        Returns:
            Dict[str, Any]: 기본 응답 데이터
        """
        import random
        
        # 카테고리 매핑 가져오기
        category_mapping = settings.CATEGORY_KOREAN_NAMES
        
        # 분석 텍스트와 차트 데이터 준비
        analysis_text = {}
        analysis_chart = {}
        
        # 각 카테고리별 기본 응답 생성
        for category in settings.EVALUATION_CATEGORIES:
            kor_category = category_mapping.get(category, category)
            score = random.randint(30, 70)
            error_message = settings.DEFAULT_ERROR_MESSAGES_EN.get(category, "Evaluation failed")
            analysis_text[kor_category] = error_message
            analysis_chart[kor_category] = score
        
        # 종합 점수와 코멘트
        overall_score = random.randint(30, 70)
        overall_comment = settings.DEFAULT_ERROR_MESSAGES_EN.get("overall", "An error occurred during analysis")
        
        # 기본 해시태그
        hashtags = settings.DEFAULT_HASHTAGS_EN.copy()
        
        # 번역 처리
        translated_text = translate_dict_values(analysis_text)
        translated_comment = translate_dict_values({"comment": overall_comment})["comment"]
        translated_hashtags = translate_batch(hashtags)
        
        # 최종 결과 조합
        return {
            "score": overall_score,
            "comment": translated_comment,
            "analysisText": translated_text,
            "analysisChart": analysis_chart,
            "hashTag": translated_hashtags
        }
    
    def _format_analysis_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        분석 결과 형식화
        
        Args:
            result (Dict[str, Any]): 파싱된 모델 응답
            
        Returns:
            Dict[str, Any]: 형식화된 결과
        """
        if not result:
            return self._create_default_response()
        
        # 분석 텍스트와 차트 데이터를 저장할 객체
        analysis_text = {}
        analysis_chart = {}
        
        # 한글 카테고리 매핑
        category_mapping = settings.CATEGORY_KOREAN_NAMES
        
        # 각 평가 카테고리 처리
        for category in settings.EVALUATION_CATEGORIES:
            if category in result and isinstance(result[category], dict):
                cat_result = result[category]
                
                # 점수 확인 및 조정
                score = scale_score(cat_result.get("score", 50))
                
                # 코멘트 확인
                comment = cat_result.get("comment", "")
                if not isinstance(comment, str):
                    comment = str(comment)
                
                # 한글 카테고리로 변환
                kor_category = category_mapping.get(category, category)
                
                # 분석 텍스트와 차트에 저장
                analysis_text[kor_category] = comment
                analysis_chart[kor_category] = score
            else:
                # 카테고리가 없으면 기본값 설정
                kor_category = category_mapping.get(category, category)
                error_message = settings.DEFAULT_ERROR_MESSAGES_EN.get(category, "Assessment information is missing")
                analysis_text[kor_category] = error_message
                analysis_chart[kor_category] = 50
        
        # 전체 평가 처리
        overall = result.get("overall", {})
        if not isinstance(overall, dict):
            overall = {}
        
        # 전체 점수 확인 및 조정
        overall_score = scale_score(overall.get("score", 50))
        
        # 전체 코멘트 가져오기
        overall_comment = overall.get("comment", "")
        if not isinstance(overall_comment, str):
            overall_comment = str(overall_comment)
        
        # 해시태그 처리
        hashtags = result.get("hashtags", [])
        if not isinstance(hashtags, list) or len(hashtags) == 0:
            # 기본 해시태그 사용
            hashtags = settings.DEFAULT_HASHTAGS_EN.copy()
        
        # 해시태그는 최대 4개로 제한
        hashtags = hashtags[:4]
        
        # 해시태그 수가 부족하면 기본 태그 추가
        while len(hashtags) < 3:
            if "photo" not in hashtags:
                hashtags.append("photo")
            elif "image" not in hashtags:
                hashtags.append("image")
            elif "art" not in hashtags:
                hashtags.append("art")
            else:
                break
        
        # 번역 처리
        translated_text = translate_dict_values(analysis_text)
        translated_comment = translate_dict_values({"comment": overall_comment})["comment"]
        translated_hashtags = translate_batch(hashtags)
        
        # 결과 형식화
        return {
            "score": overall_score,
            "comment": translated_comment,
            "analysisText": translated_text,
            "analysisChart": analysis_chart,
            "hashTag": translated_hashtags
        }


# 싱글턴 인스턴스
model_instance = ImageAnalysisModel()