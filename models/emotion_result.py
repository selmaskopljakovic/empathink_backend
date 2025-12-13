"""
Emotion Result Models
Pydantic models for API responses
"""

from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime


class EmotionScores(BaseModel):
    """Emocije sa procentima (0-100)"""
    anger: float = 0.0
    disgust: float = 0.0
    fear: float = 0.0
    joy: float = 0.0
    sadness: float = 0.0
    surprise: float = 0.0
    neutral: float = 0.0


class SentimentResult(BaseModel):
    """Sentiment analiza rezultat"""
    label: str  # positive, negative, neutral
    score: float  # 0-100


class TextMetrics(BaseModel):
    """Dodatne metrike za tekst"""
    polarity: float  # -1 to 1
    subjectivity: float  # 0 to 1
    word_count: int


class XAIExplanation(BaseModel):
    """XAI objašnjenje za Group B korisnike"""
    method: str
    confidence: float
    reasoning: str
    key_indicators: Optional[List[str]] = None
    facial_action_units: Optional[List[str]] = None
    key_features: Optional[Dict[str, str]] = None


class TextAnalysisResult(BaseModel):
    """Rezultat text analize"""
    success: bool = True
    emotions: Dict[str, float]
    primary_emotion: str
    confidence: float
    sentiment: SentimentResult
    text_metrics: TextMetrics
    xai_explanation: XAIExplanation
    processing_time_ms: float
    timestamp: datetime


class AudioFeatures(BaseModel):
    """Akustične karakteristike za voice analizu"""
    duration_seconds: float
    energy: float
    pitch_mean: float
    pitch_std: float
    tempo: float
    spectral_centroid: float
    spectral_rolloff: float


class VoiceAnalysisResult(BaseModel):
    """Rezultat voice analize"""
    success: bool = True
    emotions: Dict[str, float]
    primary_emotion: str
    confidence: float
    audio_features: AudioFeatures
    xai_explanation: XAIExplanation
    processing_time_ms: float
    timestamp: datetime


class FaceBox(BaseModel):
    """Koordinate detektovanog lica"""
    x: int
    y: int
    width: int
    height: int


class ImageAnalysisResult(BaseModel):
    """Rezultat image analize"""
    success: bool = True
    face_detected: bool = True
    emotions: Dict[str, float]
    primary_emotion: str
    confidence: float
    face_box: Optional[FaceBox] = None
    xai_explanation: XAIExplanation
    processing_time_ms: float
    timestamp: datetime


class LiveFrameResult(BaseModel):
    """Rezultat jednog frame-a iz live kamere"""
    face_detected: bool
    emotions: Dict[str, float]
    primary_emotion: Optional[str] = None
    confidence: float = 0.0
    face_box: Optional[FaceBox] = None
    timestamp: float


class FusedEmotionResult(BaseModel):
    """Kombinovani rezultat iz svih modaliteta"""
    success: bool = True
    final_emotions: Dict[str, float]
    primary_emotion: str
    confidence: float
    modalities_used: List[str]
    weights: Dict[str, float]
    individual_results: Dict[str, Dict[str, float]]
    xai_explanation: XAIExplanation
    processing_time_ms: float
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    timestamp: datetime
