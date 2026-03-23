"""
Emotion Result Models
Pydantic models for API responses
"""

from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime


class EmotionScores(BaseModel):
    """Emotions with percentages (0-100)"""
    anger: float = 0.0
    disgust: float = 0.0
    fear: float = 0.0
    joy: float = 0.0
    sadness: float = 0.0
    surprise: float = 0.0
    neutral: float = 0.0


class SentimentResult(BaseModel):
    """Sentiment analysis result"""
    label: str  # positive, negative, neutral
    score: float  # 0-100


class TextMetrics(BaseModel):
    """Additional metrics for text"""
    polarity: float  # -1 to 1
    subjectivity: float  # 0 to 1
    word_count: int


class ShapWordImportance(BaseModel):
    """Individual word contribution according to SHAP analysis"""
    word: str
    contribution: float
    direction: str  # "positive" or "negative"
    rank: int


class ShapExplanation(BaseModel):
    """SHAP explanation with word-level importances"""
    method: str  # "shap_partition"
    model: str  # "distilroberta-emotion"
    target_emotion: str
    word_importance: List[ShapWordImportance]
    truncated: bool = False
    num_words_analyzed: int


class XAIExplanation(BaseModel):
    """XAI explanation for Group B users"""
    method: str
    confidence: float
    reasoning: str
    key_indicators: Optional[List[str]] = None
    facial_action_units: Optional[List[str]] = None
    key_features: Optional[Dict[str, str]] = None
    shap_explanation: Optional[ShapExplanation] = None


class TextAnalysisResult(BaseModel):
    """Text analysis result"""
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
    """Acoustic features for voice analysis"""
    duration_seconds: float
    energy: float
    pitch_mean: float
    pitch_std: float
    tempo: float
    spectral_centroid: float
    spectral_rolloff: float


class VoiceAnalysisResult(BaseModel):
    """Voice analysis result"""
    success: bool = True
    emotions: Dict[str, float]
    primary_emotion: str
    confidence: float
    audio_features: AudioFeatures
    xai_explanation: XAIExplanation
    processing_time_ms: float
    timestamp: datetime


class FaceBox(BaseModel):
    """Coordinates of the detected face"""
    x: int
    y: int
    width: int
    height: int


class ImageAnalysisResult(BaseModel):
    """Image analysis result"""
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
    """Individual masking signal from one analysis layer"""
    layer: str              # "distribution", "temporal", "landmarks"
    type: str               # "fake_smile", "suppressed_anger", etc.
    confidence: float       # 0-1
    detail: str             # Human-readable description


class MaskingResult(BaseModel):
    """Result of masked emotion detection (fake smile, suppressed emotions)"""
    detected: bool = False
    type: Optional[str] = None                      # "fake_smile", "suppressed_anger", etc.
    confidence: float = 0.0                          # 0-1
    surface_emotion: Optional[str] = None            # Displayed emotion
    underlying_emotion: Optional[str] = None         # Possible hidden emotion
    layers_triggered: List[str] = []                 # Which layers detected
    num_signals: int = 0
    signals: List[MaskingSignal] = []
    explanation: Optional[Dict] = None               # XAI explanation
    au6_score: Optional[float] = None                # Cheek raiser (Duchenne)
    au12_score: Optional[float] = None               # Lip corner puller
    is_duchenne: Optional[bool] = None               # Whether it is a Duchenne smile


class LiveFrameResult(BaseModel):
    """Result of a single frame from the live camera"""
    face_detected: bool
    emotions: Dict[str, float]
    primary_emotion: Optional[str] = None
    confidence: float = 0.0
    face_box: Optional[FaceBox] = None
    masking: Optional[MaskingResult] = None
    timestamp: float


class IncongruenceResult(BaseModel):
    """Detection of emotional incongruence between modalities"""
    is_incongruent: bool = False
    overall_score: float = 0.0           # 0-1, higher = more incongruent
    pairwise_similarities: Dict[str, float] = {}  # e.g. {"text_vs_face": 0.72}
    details: Optional[str] = None
    possible_masking: bool = False        # True if high incongruence suggests masking


class FusedEmotionResult(BaseModel):
    """Combined result from all modalities with incongruence detection"""
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
