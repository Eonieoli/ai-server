"""
번역 관련 유틸리티 함수
"""
import logging
import requests
import json
from typing import Dict, Any, List, Optional, Union
from app.core.config import settings

logger = logging.getLogger(__name__)


def translate_text(text: str, source_lang: str = "en", target_lang: str = "ko") -> str:
    """
    텍스트 번역 함수 (Google Cloud Translation API v2 사용)
    
    Args:
        text (str): 번역할 텍스트
        source_lang (str, optional): 원본 언어 코드. 기본값은 "en".
        target_lang (str, optional): 목표 언어 코드. 기본값은 "ko".
        
    Returns:
        str: 번역된 텍스트
    """
    # 텍스트가 없거나 짧은 경우 번역하지 않음
    if not text or len(text) < 2:
        return text
    
    try:
        # API 키 가져오기
        api_key = settings.GOOGLE_TRANSLATE_KEY
        
        # API 키가 없으면 원본 텍스트 반환
        if not api_key:
            logger.warning("Google Translate API key not found. Text not translated.")
            return text
        
        # Google Cloud Translation API v2 호출
        url = "https://translation.googleapis.com/language/translate/v2"
        
        # 요청 파라미터와 본문
        params = {"key": api_key}
        data = {
            "q": text,
            "source": source_lang,
            "target": target_lang,
            "format": "text"
        }
        headers = {"Content-Type": "application/json"}
        
        # API 호출
        response = requests.post(
            url, 
            params=params, 
            data=json.dumps(data),
            headers=headers
        )
        
        # 응답 처리
        if response.status_code == 200:
            result = response.json()
            if "data" in result and "translations" in result["data"] and len(result["data"]["translations"]) > 0:
                translated_text = result["data"]["translations"][0]["translatedText"]
                logger.debug(f"Text translated successfully: {len(text)} chars")
                return translated_text
            
            logger.error(f"Unexpected response format: {result}")
        else:
            logger.error(f"Translation API error: {response.status_code} - {response.text}")
        
        return text
            
    except Exception as e:
        logger.error(f"Error in translation: {str(e)}")
        return text


def translate_batch(items: List[str], source_lang: str = "en", target_lang: str = "ko") -> List[str]:
    """
    여러 텍스트를 일괄적으로 번역
    
    Args:
        items (List[str]): 번역할 텍스트 리스트
        source_lang (str, optional): 원본 언어 코드. 기본값은 "en".
        target_lang (str, optional): 목표 언어 코드. 기본값은 "ko".
        
    Returns:
        List[str]: 번역된 텍스트 리스트
    """
    if not items:
        return []
    
    result = []
    for item in items:
        if isinstance(item, str):
            result.append(translate_text(item, source_lang, target_lang))
        else:
            # 문자열이 아닌 경우 그대로 반환
            result.append(item)
    
    return result


def translate_dict_values(
    data: Dict[str, Any], 
    keys_to_translate: Optional[List[str]] = None, 
    source_lang: str = "en", 
    target_lang: str = "ko"
) -> Dict[str, Any]:
    """
    사전의 특정 키에 해당하는 값들을 번역
    
    Args:
        data (Dict[str, Any]): 번역할 값을 포함한 사전
        keys_to_translate (Optional[List[str]]): 번역할 키 목록. None이면 모든 문자열 값 번역
        source_lang (str, optional): 원본 언어 코드. 기본값은 "en".
        target_lang (str, optional): 목표 언어 코드. 기본값은 "ko".
        
    Returns:
        Dict[str, Any]: 번역된 값을 포함한 사전
    """
    if not data:
        return {}
    
    result = {}
    
    for key, value in data.items():
        if keys_to_translate is None or key in keys_to_translate:
            if isinstance(value, str):
                result[key] = translate_text(value, source_lang, target_lang)
            elif isinstance(value, dict):
                # 중첩된 사전의 경우 재귀적으로 처리
                result[key] = translate_dict_values(
                    value, 
                    keys_to_translate, 
                    source_lang, 
                    target_lang
                )
            else:
                result[key] = value
        else:
            result[key] = value
    
    return result


def translate_dict(
    data: Dict[str, Any], 
    source_lang: str = "en", 
    target_lang: str = "ko"
) -> Dict[str, Any]:
    """
    사전의 모든 문자열 값을 번역 (간편 함수)
    
    Args:
        data (Dict[str, Any]): 번역할 사전
        source_lang (str, optional): 원본 언어 코드
        target_lang (str, optional): 목표 언어 코드
        
    Returns:
        Dict[str, Any]: 번역된 사전
    """
    return translate_dict_values(data, None, source_lang, target_lang)