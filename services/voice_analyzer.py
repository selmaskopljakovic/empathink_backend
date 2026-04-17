"""
Voice Emotion Analyzer Service
Uses Wav2Vec2 for ML-based speech emotion recognition
and librosa for acoustic feature extraction (XAI explanations).

Model: superb/wav2vec2-base-superb-er
  - Trained on IEMOCAP (naturalistic conversational speech)
  - 4 output labels: neu, hap, ang, sad  (mapped to 4 of 7 Ekman emotions)
  - Previous model ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
    had an uninitialized classifier head (HuggingFace warning: 'classifier.bias,
    classifier.weight, projector.bias, projector.weight... newly initialized'),
    which caused near-uniform outputs. See IZVJESTAJ_MULTIMODEL.md for analysis.

Three Ekman emotions (disgust, fear, surprise) are not produced by this model.
They are returned as 0.0% so the downstream schema stays stable. The heuristic
fallback in _predict_emotions_heuristic still emits all 7 when ML is unavailable.
"""

import time
import numpy as np
from typing import Dict, Optional
from datetime import datetime
import io
import logging

logger = logging.getLogger(__name__)

# Lazy-loaded Wav2Vec2 model
_ser_model = None
_ser_feature_extractor = None

# Max audio duration for ML inference (seconds)
_MAX_AUDIO_SECONDS = 30


def get_ser_model():
    """Lazy load Wav2Vec2 Speech Emotion Recognition model."""
    global _ser_model, _ser_feature_extractor

    if _ser_model is not None:
        return _ser_model, _ser_feature_extractor

    try:
        from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
        import torch

        model_name = "superb/wav2vec2-base-superb-er"

        _ser_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        _ser_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        _ser_model.eval()

        # Use float16 if memory is tight
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024 ** 3)
            if available_gb < 2.0:
                _ser_model = _ser_model.half()
                logger.info("Using float16 for Wav2Vec2 (low memory)")
        except ImportError:
            pass

        logger.info("Wav2Vec2 SER model loaded successfully")
        return _ser_model, _ser_feature_extractor

    except Exception as e:
        logger.error(f"Failed to load Wav2Vec2 SER model: {e}")
        raise


class VoiceEmotionAnalyzer:
    """
    Analyzes audio recordings and detects emotions using:
    - Wav2Vec2 (superb/wav2vec2-base-superb-er, IEMOCAP-trained)
      for ML-based emotion classification
    - librosa for acoustic feature extraction (XAI explanations)

    Emotions: anger, disgust, fear, joy, sadness, surprise, neutral (7 Ekman).
    Note: superb only outputs 4 classes (neu, hap, ang, sad). The other three
    Ekman emotions (disgust, fear, surprise) are returned as 0.0 from the ML
    path but populated by the heuristic fallback when ML is unavailable.
    """

    # Full 7-emotion set (Ekman model)
    EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

    # Mapping from superb/wav2vec2-base-superb-er labels to our Ekman labels.
    # Model output order (from its config): ['neu', 'hap', 'ang', 'sad']
    EMOTION_MAPPING = {
        "neu": "neutral",
        "hap": "joy",
        "ang": "anger",
        "sad": "sadness",
    }

    def __init__(self):
        self._ml_available = None

    def _is_ml_available(self) -> bool:
        """Check if Wav2Vec2 model can be loaded."""
        if self._ml_available is not None:
            return self._ml_available
        try:
            get_ser_model()
            self._ml_available = True
        except Exception:
            self._ml_available = False
        return self._ml_available

    def analyze(self, audio_data: bytes, include_xai: bool = True) -> Dict:
        """
        Analyzes an audio file and returns emotions with percentages.

        Args:
            audio_data: Audio data (bytes)
            include_xai: Whether to include XAI explanations

        Returns:
            Dict with emotions and audio features
        """
        start_time = time.time()

        try:
            import librosa
            import soundfile as sf

            # Load audio from bytes
            audio_buffer = io.BytesIO(audio_data)
            y, sr = sf.read(audio_buffer)

            # Convert to mono if stereo
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)

            # Resample to 16kHz if needed
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000

            # Truncate to max duration for ML model
            max_samples = _MAX_AUDIO_SECONDS * sr
            if len(y) > max_samples:
                y = y[:max_samples]

            # Extract audio features (for XAI)
            features = self._extract_features(y, sr)

            # Emotion prediction - ML model or heuristic fallback
            if self._is_ml_available():
                emotions = self._predict_emotions_ml(y, sr)
                method = "wav2vec2_with_acoustic_features"
            else:
                emotions = self._predict_emotions_heuristic(features)
                method = "acoustic_feature_analysis"

            # Find primary emotion
            primary_emotion = max(emotions, key=emotions.get)
            confidence = emotions[primary_emotion]

            # XAI explanation
            xai_explanation = None
            if include_xai:
                xai_explanation = self._generate_explanation(
                    features, primary_emotion, method
                )

            processing_time = (time.time() - start_time) * 1000

            return {
                "success": True,
                "emotions": emotions,
                "primary_emotion": primary_emotion,
                "confidence": confidence,
                "audio_features": features,
                "xai_explanation": xai_explanation,
                "processing_time_ms": round(processing_time, 2),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Voice analysis error: {e}")
            return {
                "success": False,
                "error": str(e),
                "emotions": {"neutral": 100.0},
                "primary_emotion": "neutral",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }

    def _extract_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extracts acoustic features from the audio signal"""
        import librosa

        # Duration
        duration = len(y) / sr

        # Energy (RMS)
        rms = librosa.feature.rms(y=y)
        energy = float(np.mean(rms))

        # Pitch (fundamental frequency) using librosa.pyin
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=50, fmax=500, sr=sr
            )
            f0_clean = f0[~np.isnan(f0)]
            pitch_mean = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 150.0
            pitch_std = float(np.std(f0_clean)) if len(f0_clean) > 0 else 30.0
        except:
            pitch_mean = 150.0
            pitch_std = 30.0

        # Tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo)
        except:
            tempo = 100.0

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)

        # MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = [float(x) for x in np.mean(mfccs, axis=1)]

        return {
            "duration_seconds": round(duration, 2),
            "energy": round(energy, 4),
            "pitch_mean": round(pitch_mean, 2),
            "pitch_std": round(pitch_std, 2),
            "tempo": round(tempo, 2),
            "spectral_centroid": round(float(np.mean(spectral_centroid)), 2),
            "spectral_rolloff": round(float(np.mean(spectral_rolloff)), 2),
            "zero_crossing_rate": round(float(np.mean(zcr)), 4),
            "mfcc_features": mfcc_means[:5]  # First 5 MFCC coefficients
        }

    def _predict_emotions_ml(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Predicts emotions using the Wav2Vec2 ML model.

        Model: superb/wav2vec2-base-superb-er
        Trained on: IEMOCAP (naturalistic conversational speech)
        Labels: neu, hap, ang, sad  (only 4 of 7 Ekman emotions)
        """
        import torch

        model, feature_extractor = get_ser_model()

        # Prepare input
        inputs = feature_extractor(
            y, sampling_rate=sr, return_tensors="pt", padding=True
        )

        with torch.no_grad():
            logits = model(**inputs).logits

        # Softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]

        # Read label ordering directly from the model config instead of hardcoding
        # so that any future checkpoint change stays correct.
        id2label = getattr(model.config, "id2label", None) or {
            0: "neu", 1: "hap", 2: "ang", 3: "sad",
        }
        model_labels = [id2label[i].lower() for i in range(len(id2label))]

        # Aggregate into our 7 Ekman emotions. Labels not in mapping (e.g., if a
        # future model adds classes we don't recognise) are silently skipped.
        emotion_scores = {e: 0.0 for e in self.EMOTIONS}
        for i, label in enumerate(model_labels):
            mapped = self.EMOTION_MAPPING.get(label)
            if mapped is not None:
                emotion_scores[mapped] += float(probs[i].item())

        # Convert to percentages and round
        total = sum(emotion_scores.values())
        if total > 0:
            emotions = {k: round((v / total) * 100, 1) for k, v in emotion_scores.items()}
        else:
            emotions = {e: 0.0 for e in self.EMOTIONS}
            emotions["neutral"] = 100.0

        return emotions

    def _predict_emotions_heuristic(self, features: Dict) -> Dict[str, float]:
        """
        Predicts emotions based on acoustic features (fallback).

        This is a heuristic approach based on research:
        - High energy + high pitch = happy/angry
        - Low energy + low pitch = sad
        - High variability = excited/fear
        - Low variability = neutral
        """
        energy = features["energy"]
        pitch_mean = features["pitch_mean"]
        pitch_std = features["pitch_std"]
        tempo = features["tempo"]

        # Normalize features
        energy_norm = min(energy * 10, 1.0)  # 0-1
        pitch_norm = (pitch_mean - 100) / 200  # Normalize around 200Hz
        pitch_var = pitch_std / 100  # Variability
        tempo_norm = (tempo - 80) / 80  # Normalize around 120bpm

        # Calculate score for each emotion (7 emotions)
        scores = {}

        # Joy: high energy, high pitch, faster tempo
        scores["joy"] = (
            0.3 * energy_norm +
            0.3 * max(0, pitch_norm) +
            0.2 * max(0, tempo_norm) +
            0.2 * pitch_var
        )

        # Sadness: low energy, low pitch, slower tempo
        scores["sadness"] = (
            0.3 * (1 - energy_norm) +
            0.3 * max(0, -pitch_norm) +
            0.2 * max(0, -tempo_norm) +
            0.2 * (1 - pitch_var)
        )

        # Anger: high energy, medium pitch, high variability
        scores["anger"] = (
            0.4 * energy_norm +
            0.2 * abs(pitch_norm) +
            0.2 * pitch_var +
            0.2 * max(0, tempo_norm)
        )

        # Fear: high variability, higher pitch
        scores["fear"] = (
            0.3 * pitch_var +
            0.3 * max(0, pitch_norm) +
            0.2 * (1 - energy_norm) +
            0.2 * max(0, tempo_norm)
        )

        # Surprise: high pitch, high energy, rapid changes
        scores["surprise"] = (
            0.3 * max(0, pitch_norm) +
            0.3 * pitch_var +
            0.2 * energy_norm +
            0.2 * max(0, tempo_norm)
        )

        # Disgust: low pitch, moderate energy
        scores["disgust"] = (
            0.3 * max(0, -pitch_norm) +
            0.3 * (1 - pitch_var) +
            0.2 * energy_norm +
            0.2 * (1 - abs(tempo_norm))
        )

        # Neutral: low values across all features
        scores["neutral"] = (
            0.3 * (1 - abs(pitch_norm)) +
            0.3 * (1 - pitch_var) +
            0.2 * (0.5 - abs(energy_norm - 0.5)) +
            0.2 * (1 - abs(tempo_norm))
        )

        # Normalize to 100%
        total = sum(scores.values())
        if total > 0:
            emotions = {k: round((v / total) * 100, 1) for k, v in scores.items()}
        else:
            emotions = {e: 0.0 for e in self.EMOTIONS}
            emotions["neutral"] = 100.0

        return emotions

    def _generate_explanation(
        self, features: Dict, primary_emotion: str, method: str = "acoustic_feature_analysis"
    ) -> Dict:
        """Generates XAI explanation for voice analysis"""

        explanations = {
            "joy": "High speech energy and elevated pitch indicate a positive mood.",
            "sadness": "Lower energy and slower speech tempo are characteristic of a sad mood.",
            "anger": "Intense energy and variable pitch suggest anger or frustration.",
            "fear": "Elevated pitch and faster tempo may indicate anxiety or fear.",
            "surprise": "A sudden rise in pitch and energy indicates surprise.",
            "disgust": "Low pitch with moderate energy suggests disgust or aversion.",
            "neutral": "Balanced speech parameters without extreme values.",
        }

        # Determine level for each feature
        energy_level = "high" if features["energy"] > 0.1 else "low"
        speech_rate = "fast" if features["tempo"] > 120 else "slow" if features["tempo"] < 80 else "normal"
        pitch_variation = "high" if features["pitch_std"] > 50 else "low"

        return {
            "method": method,
            "reasoning": explanations.get(primary_emotion, "Analysis of acoustic speech characteristics."),
            "key_features": {
                "energy_level": energy_level,
                "speech_rate": speech_rate,
                "pitch_variation": pitch_variation
            },
            "audio_metrics": {
                "duration": f"{features['duration_seconds']}s",
                "average_pitch": f"{features['pitch_mean']}Hz",
                "tempo": f"{features['tempo']}bpm"
            },
            "interpretation": f"Voice lasts {features['duration_seconds']} seconds with an average pitch of {features['pitch_mean']}Hz. "
                            f"Speech energy is {energy_level}, and tempo is {speech_rate}."
        }


# Singleton instance
voice_analyzer = VoiceEmotionAnalyzer()
