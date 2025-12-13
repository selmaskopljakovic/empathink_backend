"""
Voice Emotion Analyzer Service
Uses librosa for audio features and deep learning for emotion detection
"""

import time
import numpy as np
from typing import Dict, Optional
from datetime import datetime
import io


class VoiceEmotionAnalyzer:
    """
    Analizira audio snimke i detektuje emocije koristeći:
    - librosa za ekstrakciju akustičnih karakteristika
    - Deep learning model za klasifikaciju emocija
    """

    # Emocije koje detektujemo
    EMOTIONS = ["neutral", "happy", "sad", "angry", "fear"]

    def __init__(self):
        self._model = None
        self._is_initialized = False

    def _initialize(self):
        """Lazy initialization modela"""
        if self._is_initialized:
            return

        try:
            # Ovdje bi se učitao pre-trained model
            # Za sada koristimo feature-based pristup
            self._is_initialized = True
        except Exception as e:
            print(f"Voice model initialization error: {e}")

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

            # Ekstrahuj audio features
            features = self._extract_features(y, sr)

            # Predikcija emocija na osnovu features
            emotions = self._predict_emotions(features)

            # Pronađi primarnu emociju
            primary_emotion = max(emotions, key=emotions.get)
            confidence = emotions[primary_emotion]

            # XAI objašnjenje
            xai_explanation = None
            if include_xai:
                xai_explanation = self._generate_explanation(features, primary_emotion)

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

    def _predict_emotions(self, features: Dict) -> Dict[str, float]:
        """
        Predviđa emocije na osnovu akustičnih karakteristika.

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

        # Izračunaj score za svaku emociju
        scores = {}

        # Happy: visoka energija, visok pitch, brži tempo
        scores["happy"] = (
            0.3 * energy_norm +
            0.3 * max(0, pitch_norm) +
            0.2 * max(0, tempo_norm) +
            0.2 * pitch_var
        )

        # Sad: niska energija, nizak pitch, sporiji tempo
        scores["sad"] = (
            0.3 * (1 - energy_norm) +
            0.3 * max(0, -pitch_norm) +
            0.2 * max(0, -tempo_norm) +
            0.2 * (1 - pitch_var)
        )

        # Angry: visoka energija, srednji pitch, visoka varijabilnost
        scores["angry"] = (
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
            emotions = {"neutral": 100.0, "happy": 0.0, "sad": 0.0, "angry": 0.0, "fear": 0.0}

        return emotions

    def _generate_explanation(self, features: Dict, primary_emotion: str) -> Dict:
        """Generiše XAI objašnjenje za voice analizu"""

        explanations = {
            "happy": "Visoka energija govora i povišen pitch ukazuju na pozitivno raspoloženje.",
            "sad": "Niža energija i sporiji tempo govora karakteristični su za tužno raspoloženje.",
            "angry": "Intenzivna energija i varijabilni pitch sugerišu ljutnju ili frustraciju.",
            "fear": "Povišen pitch i brži tempo mogu ukazivati na anksioznost ili strah.",
            "neutral": "Ujednačeni parametri govora bez ekstremnih vrijednosti."
        }

        # Odredi level za svaki feature
        energy_level = "high" if features["energy"] > 0.1 else "low"
        speech_rate = "fast" if features["tempo"] > 120 else "slow" if features["tempo"] < 80 else "normal"
        pitch_variation = "high" if features["pitch_std"] > 50 else "low"

        return {
            "method": "acoustic_feature_analysis",
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
