"""
Multimodal Analysis API Route
Accepts text + audio + image in a single request.
Returns fused emotion vector with incongruence detection.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
from services.text_analyzer import text_analyzer
from services.voice_analyzer import voice_analyzer
from services.face_analyzer import face_analyzer
from services.fusion_engine import fusion_engine

router = APIRouter()


@router.post("/multimodal")
async def analyze_multimodal(
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    include_xai: bool = Form(True),
    user_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    """
    Multimodal emotion analysis — combines text, voice, and face.
    Accepts any combination of modalities (at least one required).

    - **text**: Text input for NLP emotion detection
    - **audio**: Audio file for voice emotion detection (WAV, MP3, M4A, OGG, WebM)
    - **image**: Face image for facial expression detection (JPEG, PNG, WebP)
    - **include_xai**: Include XAI explanations (default: true)

    Returns:
        Fused emotion result with per-modality breakdown and incongruence detection
    """
    # Validate that at least one modality is provided
    has_text = text and text.strip()
    has_audio = audio is not None
    has_image = image is not None

    if not has_text and not has_audio and not has_image:
        raise HTTPException(
            status_code=400,
            detail="At least one modality (text, audio, or image) is required."
        )

    # Analyze each provided modality
    text_result = None
    voice_result = None
    face_result = None

    try:
        # Text analysis
        if has_text:
            text_result = text_analyzer.analyze(text, include_xai=include_xai)

        # Voice analysis
        if has_audio:
            audio_data = await audio.read()
            if len(audio_data) > 10 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="Audio file too large (max 10MB)")
            voice_result = voice_analyzer.analyze(audio_data, include_xai=include_xai)

        # Face/image analysis
        if has_image:
            image_data = await image.read()
            if len(image_data) > 5 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="Image file too large (max 5MB)")
            face_result = face_analyzer.analyze_image(
                image_data=image_data, include_xai=include_xai
            )

        # Fuse all results
        fused = fusion_engine.fuse(
            text_result=text_result,
            voice_result=voice_result,
            face_result=face_result,
        )

        return fused

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Multimodal analysis failed: {str(e)}"
        )


@router.get("/multimodal/info")
async def get_multimodal_info():
    """Returns information about the multimodal fusion system."""
    return {
        "description": "Multimodal Emotion Fusion with Incongruence Detection",
        "modalities": {
            "text": {
                "model": "j-hartmann/emotion-english-distilroberta-base",
                "type": "DistilRoBERTa",
                "default_weight": 0.40,
            },
            "voice": {
                "model": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                "type": "Wav2Vec2-Large-XLSR",
                "default_weight": 0.30,
            },
            "face": {
                "model": "FER (CNN on FER2013) / DeepFace (AffectNet)",
                "type": "CNN + MTCNN",
                "default_weight": 0.30,
            },
        },
        "fusion_method": "Confidence-weighted average",
        "incongruence_detection": {
            "method": "Pairwise cosine similarity between modality emotion vectors",
            "threshold": 0.70,
            "masking_threshold": 0.50,
        },
        "emotions": ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"],
    }
