"""
Image Analysis API Routes
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request
from typing import Optional
from services.face_analyzer import face_analyzer
from dependencies import limiter
from api.auth import get_current_user
from api.file_validation import validate_image_bytes

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/image")
@limiter.limit("10/minute")
async def analyze_image(
    request: Request,
    image: UploadFile = File(...),
    include_xai: bool = Form(True),
    user_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    user: dict = Depends(get_current_user),
):
    """
    Analizira sliku lica i vraća emocije sa procentima.

    - **image**: Slika lica (JPEG, PNG, WebP)
    - **include_xai**: Da li uključiti XAI objašnjenja (default: true)

    Returns:
        ImageAnalysisResult sa emocijama i face box koordinatama
    """
    # Provjeri format - be flexible with content types
    allowed_formats = [
        "image/jpeg", "image/png", "image/webp", "image/jpg",
        "application/octet-stream",  # Sometimes sent by web browsers
    ]

    # Skip content type check if None or octet-stream (will validate by actually reading image)
    if image.content_type and image.content_type not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {image.content_type}. "
                   f"Supported formats: JPEG, PNG, WebP"
        )

    try:
        # Čitaj image bytes
        image_data = await image.read()

        # Provjeri veličinu (max 5MB)
        if len(image_data) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image file too large (max 5MB)")

        # Magic byte validation
        if not validate_image_bytes(image_data):
            raise HTTPException(status_code=400, detail="Invalid image file content")

        # Analiziraj
        result = face_analyzer.analyze_image(
            image_data=image_data,
            include_xai=include_xai
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Image analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal analysis error")


@router.get("/image/models")
@limiter.limit("60/minute")
async def get_image_models(request: Request, current_user: dict = Depends(get_current_user)):
    """
    Vraća informacije o modelima za image analizu.
    """
    return {
        "face_detection": {
            "library": "MTCNN",
            "description": "Multi-task Cascaded Convolutional Networks"
        },
        "emotion_detection": {
            "library": "FER",
            "model": "CNN-based emotion classifier",
            "emotions": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
            "based_on": "FER2013 dataset"
        },
        "xai_method": {
            "name": "Facial Action Coding System (FACS)",
            "description": "Objašnjenja bazirana na aktivaciji facijalnih mišića",
            "reference": "Ekman & Friesen (1978)"
        },
        "supported_formats": ["JPEG", "PNG", "WebP"],
        "max_file_size": "5MB"
    }
