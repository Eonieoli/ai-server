# app/api/v1/endpoints/analysis.py
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, HttpUrl

from app.services.ai_service import process_image, evaluate_image
from app.utils.helper import save_upload_file_temporarily, remove_file, download_image_from_url

router = APIRouter()


# URL 입력을 위한 Pydantic 모델
class ImageUrlInput(BaseModel):
    url: HttpUrl
    detailed: bool = False


@router.post("/process")
async def analyze_image(
    background_tasks: BackgroundTasks,
    image: Optional[UploadFile] = File(None),
    detailed: bool = False
):
    """
    이미지를 업로드하여 AI 모델로 분석합니다.
    
    - **image**: 분석할 이미지 파일
    - **detailed**: 상세한 분석 결과를 반환할지 여부
    """
    if not image:
        raise HTTPException(status_code=400, detail="이미지 파일이 제공되지 않았습니다")
        
    # 파일 형식 검증
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
    
    # 임시 파일로 저장
    temp_file_path = await save_upload_file_temporarily(image)
    
    try:
        # AI 모델로 이미지 처리
        result = await process_image(temp_file_path, detailed=detailed)
        # 백그라운드에서 임시 파일 삭제
        background_tasks.add_task(remove_file, temp_file_path)
        return result
    except Exception as e:
        # 오류 발생 시 임시 파일 즉시 삭제
        background_tasks.add_task(remove_file, temp_file_path)
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류 발생: {str(e)}")


@router.post("/process-url")
async def analyze_image_url(
    background_tasks: BackgroundTasks,
    image_input: ImageUrlInput
):
    """
    이미지 URL을 통해 AI 모델로 분석합니다.
    
    - **url**: 분석할 이미지의 URL (S3 등)
    - **detailed**: 상세한 분석 결과를 반환할지 여부
    """
    try:
        # URL에서 이미지 다운로드
        temp_file_path = await download_image_from_url(str(image_input.url))
        
        # AI 모델로 이미지 처리
        result = await process_image(temp_file_path, detailed=image_input.detailed)
        
        # 백그라운드에서 임시 파일 삭제
        background_tasks.add_task(remove_file, temp_file_path)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류 발생: {str(e)}")


@router.post("/evaluate")
async def evaluate_image_quality(
    background_tasks: BackgroundTasks,
    image: Optional[UploadFile] = File(None),
    criteria: Optional[List[str]] = None
):
    """
    이미지의 품질을 평가합니다.
    
    - **image**: 평가할 이미지 파일
    - **criteria**: 평가 기준 목록 (기본값: 모든 기준)
    """
    if not image:
        raise HTTPException(status_code=400, detail="이미지 파일이 제공되지 않았습니다")
        
    # 파일 형식 검증
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
    
    # 임시 파일로 저장
    temp_file_path = await save_upload_file_temporarily(image)
    
    try:
        # AI 모델로 이미지 평가
        result = await evaluate_image(temp_file_path, criteria=criteria)
        # 백그라운드에서 임시 파일 삭제
        background_tasks.add_task(remove_file, temp_file_path)
        return result
    except Exception as e:
        # 오류 발생 시 임시 파일 즉시 삭제
        background_tasks.add_task(remove_file, temp_file_path)
        raise HTTPException(status_code=500, detail=f"이미지 평가 중 오류 발생: {str(e)}")


# URL을 통한 평가를 위한 Pydantic 모델
class EvaluateUrlInput(BaseModel):
    url: HttpUrl
    criteria: Optional[List[str]] = None


@router.post("/evaluate-url")
async def evaluate_image_url(
    background_tasks: BackgroundTasks,
    eval_input: EvaluateUrlInput
):
    """
    이미지 URL을 통해 품질을 평가합니다.
    
    - **url**: 평가할 이미지의 URL (S3 등)
    - **criteria**: 평가 기준 목록 (기본값: 모든 기준)
    """
    try:
        # URL에서 이미지 다운로드
        temp_file_path = await download_image_from_url(str(eval_input.url))
        
        # AI 모델로 이미지 평가
        result = await evaluate_image(temp_file_path, criteria=eval_input.criteria)
        
        # 백그라운드에서 임시 파일 삭제
        background_tasks.add_task(remove_file, temp_file_path)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 평가 중 오류 발생: {str(e)}")