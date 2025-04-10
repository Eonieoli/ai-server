"""
AI 모델 정의 및 로딩 관련 코드
"""
import json
import torch
import logging
import numpy as np
import base64
import re  # 정규표현식 처리를 위해 추가
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Union, Optional
from PIL import Image

from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

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
            
            # 모델 로드 (4비트 양자화 적용)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                settings.MODEL_NAME,
                quantization_config=quantization_config,  # 양자화 설정 적용
                device_map="auto",                      # 자동 장치 매핑
                low_cpu_mem_usage=True,
                cache_dir=settings.MODEL_CACHE_DIR,      # 캐시 디렉토리 지정
                trust_remote_code=True                  # 원격 코드 허용
            )
            
            logger.info("Model loaded using LlavaForConditionalGeneration with 4-bit quantization")
            
            # 프로세서 로드
            self.processor = AutoProcessor.from_pretrained(
                settings.MODEL_NAME,
                cache_dir=settings.MODEL_CACHE_DIR,
                trust_remote_code=True                  # 원격 코드 허용
            )
            logger.info("Processor loaded using AutoProcessor")
            
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
    
    def _fix_json_hashtags(self, json_str: str) -> str:
        """
        JSON 문자열에서 해시태그 형식 문제 수정
        
        Args:
            json_str (str): 원본 JSON 문자열
            
        Returns:
            str: 수정된 JSON 문자열
        """
        # [#tag1 #tag2 #tag3] 형태를 ["tag1", "tag2", "tag3"] 형태로 변환
        hashtag_pattern = r'\[\s*#([^\[\]]+)\s*\]'
        
        def replace_hashtags(match):
            # 해시태그를 추출하여 적절한 형태로 변환
            hashtags = match.group(1).strip()
            tags = re.findall(r'#(\w+)', hashtags)
            
            # 정확한 JSON 형식의 배열로 변환
            json_array = '["' + '", "'.join(tags) + '"]'
            return json_array
        
        # 정규표현식 적용
        result = re.sub(hashtag_pattern, replace_hashtags, json_str)
        
        # 이중 따옴표 수정
        result = result.replace('\"', '"')
        
        # 엔티티 및 에스케이프 문자 처리
        result = result.replace('\\"', '"').replace('&quot;', '"')
        
        # color_harmony -> color_harmony 등의 키 이름 처리
        result = result.replace('color\\_harmony', 'color_harmony')
        result = result.replace('aesthetic\\_quality', 'aesthetic_quality')
        
        # “["#tag1", "#tag2"]” 형태의 해시태그에서 # 제거
        hashtags_with_hash_pattern = r'\[\s*"#([^"]+)"\s*(?:,\s*"#([^"]+)"\s*)*\]'
        
        def remove_hash_from_tags(match):
            full_match = match.group(0)
            # # 분자 제거
            return full_match.replace('"#', '"')
        
        # 정규표현식 적용
        result = re.sub(hashtags_with_hash_pattern, remove_hash_from_tags, result)
        
        return result
    
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
                # LLaVA 공식 예제 방식에 따른 처리
                # 대화 형식의 입력 구조
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": settings.PROMPT_TEMPLATE},
                            {"type": "image"}  # 이미지가 여기에 삽입됨
                        ],
                    },
                ]
                
                # 대화 템플릿 적용
                prompt = self.processor.apply_chat_template(
                    conversation, 
                    add_generation_prompt=True
                )
                
                # 이미지와 프롬프트로 입력 생성
                inputs = self.processor(
                    images=image, 
                    text=prompt, 
                    return_tensors="pt"
                )
                
                # 장치로 이동
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                
                # 생성 실행
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.2,
                        top_p=0.95,
                        repetition_penalty=1.2
                    )
                
                # 특수 토큰 제외하고 응답만 추출 (토큰 인덱스 2 이후)
                generated_text = self.processor.decode(output[0][2:], skip_special_tokens=True)
                
                # 로그에 생성된 텍스트 기록
                logger.info(f"Generated text: {generated_text}")
                
                # JSON 부분 추출
                try:
                    # 해시태그 형식 문제 해결
                    # ASSISTANT 부분 제거
                    if "ASSISTANT:" in generated_text:
                        generated_text = generated_text.split("ASSISTANT:", 1)[1].strip()
                    
                    # 마크다운 코드 블록에서 JSON 추출
                    if "```json" in generated_text and "```" in generated_text.split("```json", 1)[1]:
                        # 마크다운 JSON 코드 블록 파싱
                        json_content = generated_text.split("```json", 1)[1].split("```", 1)[0].strip()
                        
                        # 해시태그 형식 수정
                        json_content = self._fix_json_hashtags(json_content)
                        analysis_result = json.loads(json_content)
                    else:
                        # 일반적인 JSON 추출 시도
                        json_start = generated_text.find("{")
                        json_end = generated_text.rfind("}") + 1
                        
                        if json_start >= 0 and json_end > json_start:
                            json_str = generated_text[json_start:json_end]
                            
                            # 해시태그 형식 수정
                            json_str = self._fix_json_hashtags(json_str)
                            analysis_result = json.loads(json_str)
                        else:
                            # JSON을 찾을 수 없는 경우 기본 응답 생성
                            logger.warning("JSON not found in model response. Using default response.")
                            analysis_result = self._create_default_response()
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from model response: {e}")
                    logger.error(f"Raw JSON content: {generated_text}")
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
            "hashTag": hashtags,
            "version": 2
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
                
                # 점수 확인 - 모델이 정확한 점수를 사용하므로 scale_score 사용 안함
                score = cat_result.get("score", 50)
                
                # 점수가 1~100 범위에 있는지 확인
                if not isinstance(score, (int, float)):
                    score = 50  # 기본값
                score = max(1, min(100, score))  # 1~100 범위로 제한
                
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
            # 전체 점수 확인 및 제한
            overall_score = overall.get("score", 50)
            if not isinstance(overall_score, (int, float)):
                overall_score = 50
            overall_score = max(1, min(100, overall_score))  # 1~100 범위로 제한
            
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
            "hashTag": translated_hashtags,
            "version": 2
        }
        
        return formatted_result


# 싱글턴 인스턴스
model_instance = ImageAnalysisModel()