"""
Text Analysis API Routes
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from services.text_analyzer import text_analyzer

router = APIRouter()


class TextAnalysisRequest(BaseModel):
    """Request model for text analysis"""
    text: str
    include_xai: bool = True  # For Group B users
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class QuickAnalysisRequest(BaseModel):
    """Quick analysis without XAI"""
    text: str


@router.post("/text")
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyzes text and returns emotions with percentages.

    - **text**: Text to analyze (max 1000 characters)
    - **include_xai**: Whether to include XAI explanations (default: true)

    Returns:
        EmotionResult with all emotions, sentiment and XAI explanations
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
    Quick text analysis without XAI explanations.
    Used for real-time typing feedback.
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
        # Return only basic data for speed
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
    Returns information about available models for text analysis.
    Useful for documentation and debugging.
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
