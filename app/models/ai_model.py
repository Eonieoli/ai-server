"""
AI 모델 정의 및 로딩 관련 코드
"""
import json
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, Optional
from PIL import Image

from transformers import LlavaProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

from app.core.config import settings

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
    
    def load_model(self):
        """모델 로드 함수 (4비트 양자화 적용)"""
        try:
            logger.info(f"Loading model {settings.MODEL_NAME} with 4-bit quantization on {self.device}")
            
            # 모델 캐시 디렉토리 생성
            Path(settings.MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
            
            # 4비트 양자화 설정
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,           # 4비트 양자화 적용
                bnb_4bit_compute_dtype=torch.float16,  # 계산 시 float16 사용
                bnb_4bit_use_double_quant=True,  # 더블 양자화 적용 (메모리 사용량 절약)
                bnb_4bit_quant_type="nf4",  # NF4 양자화 타입 사용
            )
            
            # 모델 로드 (4뺄트 양자화 적용)
            # Hugging Face 문서에 따라 안정적인 커밋 아이디 사용
            self.model = LlavaForConditionalGeneration.from_pretrained(
                settings.MODEL_NAME,
                quantization_config=quantization_config,  # 양자화 설정 적용
                device_map="auto",                      # 자동 장치 매핑
                low_cpu_mem_usage=True,
                cache_dir=settings.MODEL_CACHE_DIR,      # 캐시 디렉토리 지정
                trust_remote_code=True,                 # 원격 코드 허용
                revision="02e0e67",                     # 안정적인 커밋 ID 지정
            )
            
            logger.info("Model loaded using LlavaForConditionalGeneration with 4-bit quantization")
            
            # 프로세서 로드 (고속 프로세서 사용)
            self.processor = LlavaProcessor.from_pretrained(
                settings.MODEL_NAME,
                cache_dir=settings.MODEL_CACHE_DIR,
                trust_remote_code=True,                   # 원격 코드 허용
                revision="02e0e67",                       # 모델과 동일한 커밋 ID 사용
            )
            logger.info("Processor loaded using LlavaProcessor")
            
            self.model_loaded = True
            logger.info(f"Model loaded successfully on {self.device} with 4-bit quantization")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model_loaded = False
            return False
    
    def unload_model(self):
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
            
            # 이미지 경로 확인
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # 이미지 로드
            image = Image.open(image_path).convert("RGB")
            
            try:
                # LLaVA-1.5 모델에 맞는 형식으로 입력 준비
                # Hugging Face 문서 및 GitHub 예제 참고
                # 프롬프트 생성
                prompt = settings.PROMPT_TEMPLATE

                # 이미지와 프롬프트 처리
                inputs = self.processor(
                    prompt,
                    image, 
                    return_tensors="pt"
                ).to(self.device)
                # 생성
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=512, 
                        do_sample=True,
                        temperature=0.2,       # 살짝 높은 온도로 변경 - 더 다양한 응답 생성
                        top_p=0.95,          # top-p 샘플링 설정
                        repetition_penalty=1.2  # 반복 패널티 설정
                    )
                
                # 응답 디코딩 - LLaVA-1.5 접근법
                # 입력 토큰 이후의 부분만 추출
                input_length = inputs.input_ids.shape[1]
                generated_ids = output[0][input_length:]
                generated_text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
                
                # 로그에 생성된 텍스트 기록
                logger.info(f"Generated text: {generated_text}")
                
                # JSON 부분 추출
                try:
                    # 마크다운 코드 블록에서 JSON 추출
                    if "```json" in generated_text and "```" in generated_text.split("```json", 1)[1]:
                        # 마크다운 JSON 코드 블록 파싱
                        json_content = generated_text.split("```json", 1)[1].split("```", 1)[0].strip()
                        analysis_result = json.loads(json_content)
                    else:
                        # 일반적인 JSON 추출 시도
                        json_start = generated_text.find("{")
                        json_end = generated_text.rfind("}") + 1
                        
                        if json_start >= 0 and json_end > json_start:
                            json_str = generated_text[json_start:json_end]
                            analysis_result = json.loads(json_str)
                        else:
                            # JSON을 찾을 수 없는 경우 기본 응답 생성
                            logger.warning("JSON not found in model response. Using default response.")
                            analysis_result = self._create_default_response()
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from model response: {e}")
                    analysis_result = self._create_default_response()
                
                # 결과 유효성 검사 및 포맷 맞추기
                return self._validate_and_format_result(analysis_result)
            
            except Exception as e:
                logger.error(f"Error in model inference: {e}")
                return self._create_default_response()
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return self._create_default_response()
    
    def _create_default_response(self) -> Dict[str, Any]:
        """기본 응답 생성"""
        import random
        
        # 카테고리 매핑 - settings에서 가져오기
        category_mapping = settings.CATEGORY_KOREAN_NAMES
        
        # 분석 텍스트와 차트 데이터 준비
        analysis_text = {}
        analysis_chart = {}
        
        # 기본 분석 메시지
        default_messages = {
            "구도": "Could not analyze the composition.",
            "선명도": "Could not evaluate the sharpness.",
            "주제": "Could not evaluate the subject.",
            "노출": "Could not evaluate the exposure.",
            "색감": "Could not evaluate the color harmony.",
            "미적감각": "Could not evaluate the aesthetic quality."
        }
        
        # 랜덤 점수로 각 카테고리 채우기
        for category in settings.EVALUATION_CATEGORIES:
            kor_category = category_mapping.get(category, category)
            score = random.randint(30, 70)
            analysis_text[kor_category] = default_messages.get(kor_category, "평가 실패")
            analysis_chart[kor_category] = score
        
        # 종합 점수와 코멘트
        overall_score = random.randint(30, 70)
        overall_comment = "An error occurred while analyzing the image."
        
        # 기본 해시태그
        hashtags = ["photo", "image", "analysis", "art"]
        
        # 최종 결과 조합
        return {
            "score": overall_score,
            "comment": overall_comment,
            "analysisText": analysis_text,
            "analysisChart": analysis_chart,
            "hashTag": hashtags
        }
    
    def _validate_and_format_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        결과 유효성 검사 및 포맷 맞추기
        
        Args:
            result (Dict[str, Any]): 원본 결과
            
        Returns:
            Dict[str, Any]: 새 형식으로 포맷팅된 결과
        """
        # 분석 텍스트와 차트 데이터를 저장할 객체
        analysis_text = {}
        analysis_chart = {}
        
        # 한글 카테고리 매핑 - settings에서 가져오기
        category_mapping = settings.CATEGORY_KOREAN_NAMES
        
        # 각 평가 카테고리 처리
        for category in settings.EVALUATION_CATEGORIES:
            if category in result and isinstance(result[category], dict):
                cat_result = result[category]
                
                # 점수 확인 및 조정 (유틸리티 함수 사용)
                from app.utils.helper import scale_score
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
                analysis_text[kor_category] = "평가 정보가 누락되었습니다."
                analysis_chart[kor_category] = 50
        
        # 전체 평가 처리
        overall = result.get("overall", "")
        if not isinstance(overall, dict):
            overall_score = 50
            overall_comment = "이미지에 대한 전체적인 평가 정보를 제공하지 못했습니다."
        else:
            # 전체 점수 확인 및 조정 (유틸리티 함수 사용)
            from app.utils.helper import scale_score
            overall_score = scale_score(overall.get("score", 50))
            
            # 전체 코멘트 가져오기
            overall_comment = overall.get("comment", "")
            if not isinstance(overall_comment, str):
                overall_comment = str(overall_comment)
        
        # 해시태그 처리
        hashtags = result.get("hashtags", [])
        if not isinstance(hashtags, list) or len(hashtags) == 0:
            # 모델이 해시태그를 제공하지 않은 경우, 기본 해시태그 생성
            hashtags = ["사진", "포토", "이미지"]
        
        # 해시태그는 최대 4개로 제한
        hashtags = hashtags[:4]
        
        # 해시태그 수가 부족하면 기본 태그 추가
        while len(hashtags) < 3:
            if "사진" not in hashtags:
                hashtags.append("사진")
            elif "포토" not in hashtags:
                hashtags.append("포토")
            elif "이미지" not in hashtags:
                hashtags.append("이미지")
            else:
                break
        
        # 일괄 번역 처리
        from app.utils.translator import translate_dict_values, translate_batch
        
        # 코멘트와 해시태그 번역
        analysis_text = translate_dict_values(analysis_text)
        overall_comment = translate_dict_values({"comment": overall_comment})["comment"]
        translated_hashtags = translate_batch(hashtags)
        
        # 결과 형식화
        formatted_result = {
            "score": overall_score,
            "comment": overall_comment,
            "analysisText": analysis_text,
            "analysisChart": analysis_chart,
            "hashTag": translated_hashtags
        }
        
        return formatted_result


# 싱글턴 인스턴스
model_instance = ImageAnalysisModel()