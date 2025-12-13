"""
Image Analysis API Routes
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
from services.face_analyzer import face_analyzer

router = APIRouter()


@router.post("/image")
async def analyze_image(
    image: UploadFile = File(...),
    include_xai: bool = Form(True),
    user_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None)
):
    """
    Analizira sliku lica i vraća emocije sa procentima.

    - **image**: Slika lica (JPEG, PNG, WebP)
    - **include_xai**: Da li uključiti XAI objašnjenja (default: true)

    Returns:
        ImageAnalysisResult sa emocijama i face box koordinatama
    """
    # Provjeri format
    allowed_formats = ["image/jpeg", "image/png", "image/webp", "image/jpg"]

    if image.content_type not in allowed_formats:
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

        # Analiziraj
        result = face_analyzer.analyze_image(
            image_data=image_data,
            include_xai=include_xai
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")


@router.get("/image/models")
async def get_image_models():
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
