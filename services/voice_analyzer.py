"""
Voice Emotion Analyzer Service
Uses Wav2Vec2 for ML-based speech emotion recognition
and librosa for acoustic feature extraction (XAI explanations).
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

        model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

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
    Analizira audio snimke i detektuje emocije koristeći:
    - Wav2Vec2 (ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition)
      za ML-baziranu klasifikaciju emocija
    - librosa za ekstrakciju akustičnih karakteristika (XAI objašnjenja)

    Emocije: anger, disgust, fear, joy, sadness, surprise, neutral (7 Ekman)
    """

    # Full 7-emotion set (Ekman model)
    EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

    # Mapping from Wav2Vec2 model labels to our Ekman labels
    # ehcalabres model: angry, calm, disgust, fear, happy, neutral, sad, surprise
    EMOTION_MAPPING = {
        "angry": "anger",
        "calm": "neutral",
        "disgust": "disgust",
        "fear": "fear",
        "happy": "joy",
        "neutral": "neutral",
        "sad": "sadness",
        "surprise": "surprise",
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
        Analizira audio fajl i vraća emocije sa procentima.

        Args:
            audio_data: Audio podaci (bytes)
            include_xai: Da li uključiti XAI objašnjenja

        Returns:
            Dict sa emocijama i audio karakteristikama
        """
        start_time = time.time()

        try:
            import librosa
            import soundfile as sf

            # Učitaj audio iz bytes
            audio_buffer = io.BytesIO(audio_data)
            y, sr = sf.read(audio_buffer)

            # Konvertuj u mono ako je stereo
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)

            # Resample na 16kHz ako treba
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000

            # Truncate to max duration for ML model
            max_samples = _MAX_AUDIO_SECONDS * sr
            if len(y) > max_samples:
                y = y[:max_samples]

            # Ekstrahuj audio features (za XAI)
            features = self._extract_features(y, sr)

            # Predikcija emocija - ML model ili heuristički fallback
            if self._is_ml_available():
                emotions = self._predict_emotions_ml(y, sr)
                method = "wav2vec2_with_acoustic_features"
            else:
                emotions = self._predict_emotions_heuristic(features)
                method = "acoustic_feature_analysis"

            # Pronađi primarnu emociju
            primary_emotion = max(emotions, key=emotions.get)
            confidence = emotions[primary_emotion]

            # XAI objašnjenje
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
        """Ekstrahuje akustične karakteristike iz audio signala"""
        import librosa

        # Trajanje
        duration = len(y) / sr

        # Energija (RMS)
        rms = librosa.feature.rms(y=y)
        energy = float(np.mean(rms))

        # Pitch (fundamentalna frekvencija) koristeći librosa.pyin
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
            "mfcc_features": mfcc_means[:5]  # Prvih 5 MFCC koeficijenata
        }

    def _predict_emotions_ml(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Predviđa emocije koristeći Wav2Vec2 ML model.

        Model: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
        Trained on: RAVDESS + TESS datasets
        Labels: angry, calm, disgust, fear, happy, neutral, sad, surprise
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

        # Map model labels to our Ekman labels
        # Model label order: angry, calm, disgust, fear, happy, neutral, sad, surprise
        model_labels = ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

        # Aggregate into our 7 Ekman emotions
        emotion_scores = {e: 0.0 for e in self.EMOTIONS}

        for i, label in enumerate(model_labels):
            mapped = self.EMOTION_MAPPING.get(label, "neutral")
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
        Predviđa emocije na osnovu akustičnih karakteristika (fallback).

        Ovo je heuristički pristup baziran na istraživanjima:
        - Visoka energija + visok pitch = happy/angry
        - Niska energija + nizak pitch = sad
        - Visoka varijabilnost = excited/fear
        - Niska varijabilnost = neutral
        """
        energy = features["energy"]
        pitch_mean = features["pitch_mean"]
        pitch_std = features["pitch_std"]
        tempo = features["tempo"]

        # Normalizuj features
        energy_norm = min(energy * 10, 1.0)  # 0-1
        pitch_norm = (pitch_mean - 100) / 200  # Normalizuj oko 200Hz
        pitch_var = pitch_std / 100  # Varijabilnost
        tempo_norm = (tempo - 80) / 80  # Normalizuj oko 120bpm

        # Izračunaj score za svaku emociju (7 emocija)
        scores = {}

        # Joy: visoka energija, visok pitch, brži tempo
        scores["joy"] = (
            0.3 * energy_norm +
            0.3 * max(0, pitch_norm) +
            0.2 * max(0, tempo_norm) +
            0.2 * pitch_var
        )

        # Sadness: niska energija, nizak pitch, sporiji tempo
        scores["sadness"] = (
            0.3 * (1 - energy_norm) +
            0.3 * max(0, -pitch_norm) +
            0.2 * max(0, -tempo_norm) +
            0.2 * (1 - pitch_var)
        )

        # Anger: visoka energija, srednji pitch, visoka varijabilnost
        scores["anger"] = (
            0.4 * energy_norm +
            0.2 * abs(pitch_norm) +
            0.2 * pitch_var +
            0.2 * max(0, tempo_norm)
        )

        # Fear: visoka varijabilnost, viši pitch
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

        # Neutral: niske vrijednosti svega
        scores["neutral"] = (
            0.3 * (1 - abs(pitch_norm)) +
            0.3 * (1 - pitch_var) +
            0.2 * (0.5 - abs(energy_norm - 0.5)) +
            0.2 * (1 - abs(tempo_norm))
        )

        # Normalizuj na 100%
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
        """Generiše XAI objašnjenje za voice analizu"""

        explanations = {
            "joy": "Visoka energija govora i povišen pitch ukazuju na pozitivno raspoloženje.",
            "sadness": "Niža energija i sporiji tempo govora karakteristični su za tužno raspoloženje.",
            "anger": "Intenzivna energija i varijabilni pitch sugerišu ljutnju ili frustraciju.",
            "fear": "Povišen pitch i brži tempo mogu ukazivati na anksioznost ili strah.",
            "surprise": "Nagli porast pitcha i energije ukazuju na iznenađenje.",
            "disgust": "Nizak pitch sa umjerenom energijom sugerišu gađenje ili odbojnost.",
            "neutral": "Ujednačeni parametri govora bez ekstremnih vrijednosti.",
        }

        # Odredi level za svaki feature
        energy_level = "high" if features["energy"] > 0.1 else "low"
        speech_rate = "fast" if features["tempo"] > 120 else "slow" if features["tempo"] < 80 else "normal"
        pitch_variation = "high" if features["pitch_std"] > 50 else "low"

        return {
            "method": method,
            "reasoning": explanations.get(primary_emotion, "Analiza akustičnih karakteristika govora."),
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
            "interpretation": f"Glas traje {features['duration_seconds']} sekundi sa prosječnim pitchem od {features['pitch_mean']}Hz. "
                            f"Energija govora je {energy_level}, a tempo je {speech_rate}."
        }


# Singleton instance
voice_analyzer = VoiceEmotionAnalyzer()
