"""
Text Analysis API Routes
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from services.text_analyzer import text_analyzer

router = APIRouter()


class TextAnalysisRequest(BaseModel):
    """Request model za text analizu"""
    text: str
    include_xai: bool = True  # Za Group B korisnike
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class QuickAnalysisRequest(BaseModel):
    """Brza analiza bez XAI"""
    text: str


@router.post("/text")
async def analyze_text(request: TextAnalysisRequest):
    """
    Analizira tekst i vraća emocije sa procentima.

    - **text**: Tekst za analizu (max 1000 karaktera)
    - **include_xai**: Da li uključiti XAI objašnjenja (default: true)

    Returns:
        EmotionResult sa svim emocijama, sentimentom i XAI objašnjenjima
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    if len(request.text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long (max 5000 characters)")

    try:
        result = text_analyzer.analyze(
            text=request.text,
            include_xai=request.include_xai
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/text/quick")
async def quick_analyze_text(request: QuickAnalysisRequest):
    """
    Brza analiza teksta bez XAI objašnjenja.
    Koristi se za real-time typing feedback.
    """
    if not request.text or len(request.text) < 3:
        return {
            "success": True,
            "emotions": {"neutral": 100.0},
            "primary_emotion": "neutral",
            "confidence": 0.0
        }

    try:
        result = text_analyzer.analyze(
            text=request.text,
            include_xai=False
        )
        # Vraća samo osnovne podatke za brzinu
        return {
            "success": True,
            "emotions": result["emotions"],
            "primary_emotion": result["primary_emotion"],
            "confidence": result["confidence"]
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "emotions": {"neutral": 100.0},
            "primary_emotion": "neutral"
        }


@router.get("/text/models")
async def get_available_models():
    """
    Vraća informacije o dostupnim modelima za text analizu.
    Korisno za dokumentaciju i debugging.
    """
    return {
        "emotion_model": {
            "name": "j-hartmann/emotion-english-distilroberta-base",
            "type": "DistilRoBERTa",
            "emotions": ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"],
            "source": "HuggingFace"
        },
        "sentiment_model": {
            "name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "type": "RoBERTa",
            "labels": ["positive", "negative", "neutral"],
            "source": "HuggingFace"
        },
        "additional_analysis": {
            "name": "TextBlob",
            "metrics": ["polarity", "subjectivity"]
        }
    }
