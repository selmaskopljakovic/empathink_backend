"""
Text Analysis API Routes
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from services.text_analyzer import text_analyzer
from dependencies import limiter
from api.auth import get_current_user

logger = logging.getLogger(__name__)

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
@limiter.limit("20/minute")
async def analyze_text(request: Request, body: TextAnalysisRequest, user: dict = Depends(get_current_user)):
    """
    Analizira tekst i vraća emocije sa procentima.

    - **text**: Tekst za analizu (max 1000 karaktera)
    - **include_xai**: Da li uključiti XAI objašnjenja (default: true)

    Returns:
        EmotionResult sa svim emocijama, sentimentom i XAI objašnjenjima
    """
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    if len(body.text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long (max 5000 characters)")

    try:
        result = text_analyzer.analyze(
            text=body.text,
            include_xai=body.include_xai
        )
        return result

    except Exception as e:
        logger.error("Text analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal analysis error")


@router.post("/text/quick")
@limiter.limit("60/minute")
async def quick_analyze_text(request: Request, body: QuickAnalysisRequest, user: dict = Depends(get_current_user)):
    """
    Brza analiza teksta bez XAI objašnjenja.
    Koristi se za real-time typing feedback.
    """
    if not body.text or len(body.text) < 3:
        return {
            "success": True,
            "emotions": {"neutral": 100.0},
            "primary_emotion": "neutral",
            "confidence": 0.0
        }

    if len(body.text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long (max 5000 characters)")

    try:
        result = text_analyzer.analyze(
            text=body.text,
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
        logger.error(f"Quick text analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal analysis error")


@router.get("/text/models")
@limiter.limit("60/minute")
async def get_available_models(request: Request, current_user: dict = Depends(get_current_user)):
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
