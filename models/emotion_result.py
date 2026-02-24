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


class ShapWordImportance(BaseModel):
    """Pojedinačni doprinos riječi prema SHAP analizi"""
    word: str
    contribution: float
    direction: str  # "positive" or "negative"
    rank: int


class ShapExplanation(BaseModel):
    """SHAP objašnjenje sa word-level važnostima"""
    method: str  # "shap_partition"
    model: str  # "distilroberta-emotion"
    target_emotion: str
    word_importance: List[ShapWordImportance]
    truncated: bool = False
    num_words_analyzed: int


class XAIExplanation(BaseModel):
    """XAI objašnjenje za Group B korisnike"""
    method: str
    confidence: float
    reasoning: str
    key_indicators: Optional[List[str]] = None
    facial_action_units: Optional[List[str]] = None
    key_features: Optional[Dict[str, str]] = None
    shap_explanation: Optional[ShapExplanation] = None


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


class MaskingSignal(BaseModel):
    """Pojedinacni signal maskiranja iz jednog sloja analize"""
    layer: str              # "distribution", "temporal", "landmarks"
    type: str               # "fake_smile", "suppressed_anger", etc.
    confidence: float       # 0-1
    detail: str             # Human-readable opis


class MaskingResult(BaseModel):
    """Rezultat detekcije maskiranih emocija (lazni osmijeh, potisnute emocije)"""
    detected: bool = False
    type: Optional[str] = None                      # "fake_smile", "suppressed_anger", etc.
    confidence: float = 0.0                          # 0-1
    surface_emotion: Optional[str] = None            # Prikazana emocija
    underlying_emotion: Optional[str] = None         # Moguca skrivena emocija
    layers_triggered: List[str] = []                 # Koji slojevi su detektovali
    num_signals: int = 0
    signals: List[MaskingSignal] = []
    explanation: Optional[Dict] = None               # XAI objasnjenje
    au6_score: Optional[float] = None                # Cheek raiser (Duchenne)
    au12_score: Optional[float] = None               # Lip corner puller
    is_duchenne: Optional[bool] = None               # Da li je Duchenne osmijeh


class LiveFrameResult(BaseModel):
    """Rezultat jednog frame-a iz live kamere"""
    face_detected: bool
    emotions: Dict[str, float]
    primary_emotion: Optional[str] = None
    confidence: float = 0.0
    face_box: Optional[FaceBox] = None
    masking: Optional[MaskingResult] = None
    timestamp: float


class IncongruenceResult(BaseModel):
    """Detekcija emocionalne nekongruencije između modaliteta"""
    is_incongruent: bool = False
    overall_score: float = 0.0           # 0-1, higher = more incongruent
    pairwise_similarities: Dict[str, float] = {}  # e.g. {"text_vs_face": 0.72}
    details: Optional[str] = None
    possible_masking: bool = False        # True if high incongruence suggests masking


class FusedEmotionResult(BaseModel):
    """Kombinovani rezultat iz svih modaliteta sa detekcijom nekongruencije"""
    success: bool = True
    final_emotions: Dict[str, float]
    primary_emotion: str
    confidence: float
    modalities_used: List[str]
    weights: Dict[str, float]
    individual_results: Dict[str, Dict[str, float]]
    incongruence: Optional[IncongruenceResult] = None
    xai_explanation: Optional[Dict] = None
    processing_time_ms: float
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    timestamp: datetime
