"""
Voice Analysis API Routes
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request
from typing import Optional
from services.voice_analyzer import voice_analyzer
from dependencies import limiter
from api.auth import get_current_user
from api.file_validation import validate_audio_bytes

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/voice")
@limiter.limit("10/minute")
async def analyze_voice(
    request: Request,
    audio: UploadFile = File(...),
    include_xai: bool = Form(True),
    user_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    user: dict = Depends(get_current_user),
):
    """
    Analizira audio snimak i vraća emocije sa procentima.

    - **audio**: Audio fajl (WAV, MP3, M4A, OGG)
    - **include_xai**: Da li uključiti XAI objašnjenja (default: true)

    Returns:
        VoiceAnalysisResult sa emocijama i audio karakteristikama
    """
    # Provjeri format
    allowed_formats = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/m4a",
                       "audio/ogg", "audio/webm", "application/octet-stream"]

    if audio.content_type not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {audio.content_type}. "
                   f"Supported formats: WAV, MP3, M4A, OGG, WebM"
        )

    try:
        # Čitaj audio bytes
        audio_data = await audio.read()

        # Provjeri veličinu (max 10MB)
        if len(audio_data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Audio file too large (max 10MB)")

        # Magic byte validation
        if not validate_audio_bytes(audio_data):
            raise HTTPException(status_code=400, detail="Invalid audio file content")

        # Analiziraj
        result = voice_analyzer.analyze(
            audio_data=audio_data,
            include_xai=include_xai
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Voice analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal analysis error")


@router.get("/voice/models")
@limiter.limit("60/minute")
async def get_voice_models(request: Request, current_user: dict = Depends(get_current_user)):
    """
    Vraća informacije o modelima za voice analizu.
    """
    ml_available = voice_analyzer._is_ml_available()

    return {
        "feature_extraction": {
            "library": "librosa",
            "version": "0.10.1",
            "features": [
                "pitch (fundamental frequency)",
                "energy (RMS)",
                "tempo",
                "spectral centroid",
                "spectral rolloff",
                "zero crossing rate",
                "MFCC coefficients"
            ]
        },
        "emotion_detection": {
            "method": "wav2vec2_with_acoustic_features" if ml_available else "acoustic_feature_analysis",
            "model": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition" if ml_available else "heuristic",
            "model_type": "Wav2Vec2-Large-XLSR" if ml_available else "Rule-based",
            "training_data": "RAVDESS + TESS" if ml_available else "N/A",
            "emotions": ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"],
            "ml_available": ml_available,
        },
        "supported_formats": ["WAV", "MP3", "M4A", "OGG", "WebM"],
        "max_file_size": "10MB",
        "max_audio_duration": "30 seconds"
    }
