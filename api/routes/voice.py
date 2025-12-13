"""
Voice Analysis API Routes
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
from services.voice_analyzer import voice_analyzer

router = APIRouter()


@router.post("/voice")
async def analyze_voice(
    audio: UploadFile = File(...),
    include_xai: bool = Form(True),
    user_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None)
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

        # Analiziraj
        result = voice_analyzer.analyze(
            audio_data=audio_data,
            include_xai=include_xai
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice analysis failed: {str(e)}")


@router.get("/voice/models")
async def get_voice_models():
    """
    Vraća informacije o modelima za voice analizu.
    """
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
            "method": "acoustic_feature_analysis",
            "emotions": ["neutral", "happy", "sad", "angry", "fear"],
            "based_on": "Speech emotion research literature"
        },
        "supported_formats": ["WAV", "MP3", "M4A", "OGG", "WebM"],
        "max_file_size": "10MB"
    }
