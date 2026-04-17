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
    Analyzes an audio recording and returns emotions with percentages.

    - **audio**: Audio file (WAV, MP3, M4A, OGG)
    - **include_xai**: Whether to include XAI explanations (default: true)

    Returns:
        VoiceAnalysisResult with emotions and audio features
    """
    # Check format
    allowed_formats = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/m4a",
                       "audio/ogg", "audio/webm", "application/octet-stream"]

    if audio.content_type not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {audio.content_type}. "
                   f"Supported formats: WAV, MP3, M4A, OGG, WebM"
        )

    try:
        # Read audio bytes
        audio_data = await audio.read()

        # Check size (max 10MB)
        if len(audio_data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Audio file too large (max 10MB)")

        # Analyze
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
    Returns information about models used for voice analysis.
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
            "model": "superb/wav2vec2-base-superb-er" if ml_available else "heuristic",
            "model_type": "Wav2Vec2-base (superb/IEMOCAP)" if ml_available else "Rule-based",
            "training_data": "IEMOCAP" if ml_available else "N/A",
            "emotions": ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"],
            "ml_emotions_supported": ["anger", "joy", "sadness", "neutral"] if ml_available else None,
            "ml_available": ml_available,
        },
        "supported_formats": ["WAV", "MP3", "M4A", "OGG", "WebM"],
        "max_file_size": "10MB",
        "max_audio_duration": "30 seconds"
    }
