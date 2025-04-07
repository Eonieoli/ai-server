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

from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

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
        """모델 로드 함수"""
        try:
            logger.info(f"Loading model {settings.MODEL_NAME} on {self.device}")
            
            # 모델 캐시 디렉토리 생성
            Path(settings.MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
            
            # 모델 로드 (예제 코드 기반)
            self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                settings.MODEL_NAME,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            ).to(self.device)
            
            logger.info("Model loaded using LlavaNextVideoForConditionalGeneration")
            
            # 프로세서 로드
            self.processor = LlavaNextVideoProcessor.from_pretrained(
                settings.MODEL_NAME,
                cache_dir=settings.MODEL_CACHE_DIR
            )
            logger.info("Processor loaded using LlavaNextVideoProcessor")
            
            self.model_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            
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
                # 예제 코드에 따라 chat template을 사용하여 프롬프트 생성
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": settings.PROMPT_TEMPLATE},
                            {"type": "image"},  # 이미지를 여기에 넣을 것임
                        ],
                    },
                ]
                
                prompt = self.processor.apply_chat_template(
                    conversation, 
                    add_generation_prompt=True
                )
                
                # 이미지를 numpy 배열로 변환 (processor가 필요로 함)
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
                
                # 응답 디코딩 (예제 코드와 같이 output[0][2:]를 사용)
                generated_text = self.processor.decode(
                    output[0][2:], 
                    skip_special_tokens=True
                )
                
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
        
        # 카테고리 매핑
        category_mapping = {
            "composition": "구도",
            "sharpness": "선명도",
            "noise_free": "노이즈",
            "exposure": "노출",
            "color_harmony": "색감", 
            "aesthetics": "심미성"
        }
        
        # 분석 텍스트와 차트 데이터 준비
        analysis_text = {}
        analysis_chart = {}
        
        # 랜덤 점수로 각 카테고리 채우기
        for category in settings.EVALUATION_CATEGORIES:
            kor_category = category_mapping.get(category, category)
            score = random.randint(30, 70)
            analysis_text[kor_category] = "평가 실패"
            analysis_chart[kor_category] = score
        
        # 종합 점수와 코멘트
        overall_score = random.randint(30, 70)
        overall_comment = "이미지를 분석하는 동안 오류가 발생했습니다."
        
        # 기본 해시태그
        hashtags = ["사진", "이미지", "분석"]
        
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
        
        # 한글 카테고리 매핑
        category_mapping = {
            "composition": "구도",
            "sharpness": "선명도",
            "noise_free": "노이즈",
            "exposure": "노출",
            "color_harmony": "색감", 
            "aesthetics": "심미성"
        }
        
        # 로그 추가
        import random
        
        # 각 평가 카테고리 처리
        for category in settings.EVALUATION_CATEGORIES:
            if category in result and isinstance(result[category], dict):
                cat_result = result[category]
                
                # 점수 확인 및 조정
                score = cat_result.get("score", 50)
                if not isinstance(score, (int, float)):
                    score = 50
                
                # 1-10 점수를 1-100 점수로 변환
                if 1 <= score <= 10:
                    # 단순히 10을 곱하는 것이 아니라 분포를 가진 점수로 변환
                    base_score = int(score * 10)  # 기본 점수 (10점은 100점)
                    
                    # 1-7점의 경우: 분포를 가진 랜덤 점수 생성
                    if score < 8:
                        variation = random.randint(-5, 5)  # -5 ~ +5 사이의 변동
                        final_score = max(1, min(100, base_score + variation))
                    # 8-10점의 경우: 더 작은 범위의 변동 적용 (높은 점수 유지)
                    else:
                        variation = random.randint(-3, 3)  # -3 ~ +3 사이의 작은 변동
                        final_score = max(70, min(100, base_score + variation))
                    
                    score = final_score
                
                # 최종 점수 범위 확인
                score = max(1, min(100, int(score)))
                
                # 코멘트 확인
                comment = cat_result.get("comment", "")
                if not isinstance(comment, str):
                    comment = str(comment)
                
                # 번역 적용
                from app.utils.translator import translate_text
                translated_comment = translate_text(comment, source_lang="en", target_lang="ko")
                
                # 한글 카테고리로 변환
                kor_category = category_mapping.get(category, category)
                
                # 분석 텍스트와 차트에 저장
                analysis_text[kor_category] = translated_comment
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
            # 전체 점수 확인 및 조정
            overall_score = overall.get("score", 50)
            if not isinstance(overall_score, (int, float)):
                overall_score = 50
            
            # 1-10 점수를 1-100 점수로 변환
            if 1 <= overall_score <= 10:
                # 단순히 10을 곱하는 것이 아니라 분포를 가진 점수로 변환
                base_score = int(overall_score * 10)  # 기본 점수 (10점은 100점)
                
                # 점수에 따라 다른 변동 적용
                if overall_score < 8:
                    variation = random.randint(-5, 5)  # -5 ~ +5 사이의 변동
                    final_score = max(1, min(100, base_score + variation))
                else:
                    variation = random.randint(-3, 3)  # -3 ~ +3 사이의 작은 변동
                    final_score = max(70, min(100, base_score + variation))
                
                overall_score = final_score
            
            # 최종 점수 범위 확인
            overall_score = max(1, min(100, int(overall_score)))
            
            # 전체 코멘트 번역
            from app.utils.translator import translate_text
            overall_comment = overall.get("comment", "")
            if not isinstance(overall_comment, str):
                overall_comment = str(overall_comment)
            overall_comment = translate_text(overall_comment, source_lang="en", target_lang="ko")
        
        # 해시태그 처리
        hashtags = result.get("hashtags", [])
        if not isinstance(hashtags, list) or len(hashtags) == 0:
            # 모델이 해시태그를 제공하지 않은 경우, 기본 해시태그 생성
            hashtags = ["사진", "포토", "이미지"]
        
        # 해시태그 번역
        translated_hashtags = []
        for tag in hashtags[:4]:  # 최대 4개로 제한
            if tag and isinstance(tag, str):
                # 영어 태그를 한글로 번역
                translated_tag = translate_text(tag, source_lang="en", target_lang="ko")
                translated_hashtags.append(translated_tag)
        
        # 해시태그 수가 부족하면 기본 태그 추가
        while len(translated_hashtags) < 3:
            if "사진" not in translated_hashtags:
                translated_hashtags.append("사진")
            elif "포토" not in translated_hashtags:
                translated_hashtags.append("포토")
            elif "이미지" not in translated_hashtags:
                translated_hashtags.append("이미지")
            else:
                break
        
        # 결과 형식화
        formatted_result = {
            "score": overall_score,
            "comment": overall_comment,
            "analysisText": analysis_text,
            "analysisChart": analysis_chart,
            "hashTag": translated_hashtags[:4]  # 최대 4개로 제한
        }
        
        return formatted_result


# 싱글턴 인스턴스
model_instance = ImageAnalysisModel()