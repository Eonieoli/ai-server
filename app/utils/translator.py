"""
번역 관련 유틸리티 함수
"""
import logging
import requests
import json
from typing import Optional
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
        
        # 요청 파라미터
        params = {
            "key": api_key
        }
        
        # 요청 본문
        data = {
            "q": text,  # 번역할 텍스트
            "source": source_lang,  # 원본 언어
            "target": target_lang,  # 목표 언어
            "format": "text"  # 텍스트 형식 (html 또는 text)
        }
        
        # 요청 헤더
        headers = {
            "Content-Type": "application/json"
        }
        
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
                logger.info(f"Text translated successfully: {len(text)} chars")
                return translated_text
            else:
                logger.error(f"Unexpected response format: {result}")
                return text
        else:
            logger.error(f"Translation API error: {response.status_code} - {response.text}")
            return text
            
    except Exception as e:
        logger.error(f"Error in translation: {str(e)}")
        return text
