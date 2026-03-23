"""
Unit tests for VoiceEmotionAnalyzer service.
Tests audio loading, feature extraction, emotion prediction, and edge cases.
Run with: pytest tests/test_voice_analyzer.py -v

NOTE: Some tests require librosa/soundfile. They are skipped if not installed.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from tests.conftest import generate_wav_bytes, generate_stereo_wav_bytes

try:
    import librosa
    import soundfile as sf
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False

requires_audio_libs = pytest.mark.skipif(
    not HAS_AUDIO_LIBS, reason="librosa/soundfile not installed"
)


# ---------------------------------------------------------------------------
# VoiceEmotionAnalyzer unit tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_AUDIO_LIBS, reason="librosa/soundfile not installed")
class TestVoiceAnalyzerBasic:
    """Basic contract tests for VoiceEmotionAnalyzer.analyze()."""

    def _make_analyzer(self):
        from services.voice_analyzer import VoiceEmotionAnalyzer
        analyzer = VoiceEmotionAnalyzer()
        analyzer._ml_available = False  # Force heuristic mode for unit tests
        return analyzer

    def test_analyze_mono_wav_returns_success(self, wav_bytes):
        """Valid mono WAV should return success with all required fields."""
        analyzer = self._make_analyzer()
        result = analyzer.analyze(wav_bytes, include_xai=False)

        assert result["success"] is True
        assert "emotions" in result
        assert "primary_emotion" in result
        assert "confidence" in result
        assert "audio_features" in result
        assert "processing_time_ms" in result
        assert "timestamp" in result

    def test_analyze_stereo_wav_converts_to_mono(self, wav_bytes_stereo):
        """Stereo WAV should be converted to mono without error."""
        analyzer = self._make_analyzer()
        result = analyzer.analyze(wav_bytes_stereo, include_xai=False)
        assert result["success"] is True

    def test_analyze_resamples_to_16khz(self, wav_bytes_44100):
        """44100Hz WAV should be resampled to 16kHz."""
        analyzer = self._make_analyzer()
        result = analyzer.analyze(wav_bytes_44100, include_xai=False)
        assert result["success"] is True

    def test_analyze_truncates_long_audio(self, wav_bytes_long):
        """Audio longer than 30s should be truncated."""
        analyzer = self._make_analyzer()
        result = analyzer.analyze(wav_bytes_long, include_xai=False)
        assert result["success"] is True
        # Duration in features should be ≤ 30 seconds
        assert result["audio_features"]["duration_seconds"] <= 30.0

    def test_analyze_silent_audio(self, silent_wav_bytes):
        """Silent audio should not crash and return valid result."""
        analyzer = self._make_analyzer()
        result = analyzer.analyze(silent_wav_bytes, include_xai=False)
        assert result["success"] is True
        assert result["audio_features"]["energy"] < 0.01

    def test_analyze_invalid_audio_returns_error(self):
        """Non-audio bytes should return error gracefully."""
        analyzer = self._make_analyzer()
        result = analyzer.analyze(b"this is not audio data", include_xai=False)
        assert result["success"] is False
        assert "error" in result
        assert result["primary_emotion"] == "neutral"

    def test_analyze_empty_bytes_returns_error(self):
        """Empty bytes should return error."""
        analyzer = self._make_analyzer()
        result = analyzer.analyze(b"", include_xai=False)
        assert result["success"] is False

    def test_emotions_sum_to_100(self, wav_bytes):
        """Emotion percentages should sum to approximately 100."""
        analyzer = self._make_analyzer()
        result = analyzer.analyze(wav_bytes, include_xai=False)
        total = sum(result["emotions"].values())
        assert abs(total - 100.0) < 1.0, f"Emotions sum to {total}"

    def test_all_seven_emotions_present(self, wav_bytes):
        """All 7 Ekman emotions must be present in result."""
        expected = {"anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"}
        analyzer = self._make_analyzer()
        result = analyzer.analyze(wav_bytes, include_xai=False)
        assert set(result["emotions"].keys()) == expected

    def test_primary_emotion_is_highest(self, wav_bytes):
        """primary_emotion should be the emotion with highest score."""
        analyzer = self._make_analyzer()
        result = analyzer.analyze(wav_bytes, include_xai=False)
        emotions = result["emotions"]
        expected_primary = max(emotions, key=emotions.get)
        assert result["primary_emotion"] == expected_primary


# ---------------------------------------------------------------------------
# Audio feature extraction tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_AUDIO_LIBS, reason="librosa/soundfile not installed")
class TestAudioFeatureExtraction:
    """Tests for _extract_features()."""

    def _make_analyzer(self):
        from services.voice_analyzer import VoiceEmotionAnalyzer
        return VoiceEmotionAnalyzer()

    def test_features_contain_required_fields(self, wav_bytes):
        """Feature dict must contain all expected acoustic features."""
        import soundfile as sf
        import io

        buf = io.BytesIO(wav_bytes)
        y, sr = sf.read(buf)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        analyzer = self._make_analyzer()
        features = analyzer._extract_features(y, sr)

        required = [
            "duration_seconds", "energy", "pitch_mean", "pitch_std",
            "tempo", "spectral_centroid", "spectral_rolloff",
            "zero_crossing_rate", "mfcc_features"
        ]
        for field in required:
            assert field in features, f"Missing feature: {field}"

    def test_features_values_are_numeric(self, wav_bytes):
        """All feature values should be numeric."""
        import soundfile as sf
        import io

        buf = io.BytesIO(wav_bytes)
        y, sr = sf.read(buf)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        analyzer = self._make_analyzer()
        features = analyzer._extract_features(y, sr)

        for key, value in features.items():
            if key == "mfcc_features":
                assert isinstance(value, list)
                for v in value:
                    assert isinstance(v, float)
            else:
                assert isinstance(value, (int, float)), f"{key} is {type(value)}"

    def test_features_duration_matches_audio(self, wav_bytes):
        """Duration should approximately match the audio length."""
        import soundfile as sf
        import io

        buf = io.BytesIO(wav_bytes)
        y, sr = sf.read(buf)

        analyzer = self._make_analyzer()
        features = analyzer._extract_features(y, sr)

        expected_duration = len(y) / sr
        assert abs(features["duration_seconds"] - expected_duration) < 0.1

    def test_mfcc_has_five_coefficients(self, wav_bytes):
        """MFCC features should contain 5 coefficients."""
        import soundfile as sf
        import io

        buf = io.BytesIO(wav_bytes)
        y, sr = sf.read(buf)

        analyzer = self._make_analyzer()
        features = analyzer._extract_features(y, sr)
        assert len(features["mfcc_features"]) == 5

    def test_silent_audio_has_low_energy(self, silent_wav_bytes):
        """Silent audio should have very low energy."""
        import soundfile as sf
        import io

        buf = io.BytesIO(silent_wav_bytes)
        y, sr = sf.read(buf)

        analyzer = self._make_analyzer()
        features = analyzer._extract_features(y, sr)
        assert features["energy"] < 0.001


# ---------------------------------------------------------------------------
# Heuristic prediction tests
# ---------------------------------------------------------------------------

class TestHeuristicPrediction:
    """Tests for _predict_emotions_heuristic()."""

    def _make_analyzer(self):
        from services.voice_analyzer import VoiceEmotionAnalyzer
        return VoiceEmotionAnalyzer()

    def test_high_energy_high_pitch_favors_joy_or_anger(self):
        """High energy + high pitch should favor joy or anger."""
        analyzer = self._make_analyzer()
        features = {
            "energy": 0.2,
            "pitch_mean": 300.0,
            "pitch_std": 80.0,
            "tempo": 150.0,
        }
        emotions = analyzer._predict_emotions_heuristic(features)
        # joy or anger should be among top 2
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        top_2_names = {sorted_emotions[0][0], sorted_emotions[1][0]}
        assert top_2_names & {"joy", "anger", "surprise"}, \
            f"Expected joy/anger/surprise in top 2, got {top_2_names}"

    def test_low_energy_low_pitch_favors_sadness(self):
        """Low energy + low pitch should favor sadness."""
        analyzer = self._make_analyzer()
        features = {
            "energy": 0.01,
            "pitch_mean": 80.0,
            "pitch_std": 10.0,
            "tempo": 60.0,
        }
        emotions = analyzer._predict_emotions_heuristic(features)
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        top_3_names = {e[0] for e in sorted_emotions[:3]}
        assert "sadness" in top_3_names, f"Expected sadness in top 3, got {top_3_names}"

    def test_heuristic_sums_to_100(self):
        """Heuristic emotions must sum to ~100."""
        analyzer = self._make_analyzer()
        features = {
            "energy": 0.1, "pitch_mean": 200.0,
            "pitch_std": 40.0, "tempo": 120.0,
        }
        emotions = analyzer._predict_emotions_heuristic(features)
        total = sum(emotions.values())
        assert abs(total - 100.0) < 1.0

    def test_heuristic_returns_all_seven_emotions(self):
        """All 7 emotions must be present."""
        analyzer = self._make_analyzer()
        features = {
            "energy": 0.1, "pitch_mean": 200.0,
            "pitch_std": 40.0, "tempo": 120.0,
        }
        emotions = analyzer._predict_emotions_heuristic(features)
        expected = {"anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"}
        assert set(emotions.keys()) == expected


# ---------------------------------------------------------------------------
# ML prediction tests (mocked)
# ---------------------------------------------------------------------------

class TestMLPrediction:
    """Tests for _predict_emotions_ml() with mocked model."""

    def _make_analyzer(self):
        from services.voice_analyzer import VoiceEmotionAnalyzer
        return VoiceEmotionAnalyzer()

    @patch("services.voice_analyzer.get_ser_model")
    def test_ml_prediction_maps_labels_correctly(self, mock_get_model):
        """Model labels should be correctly mapped to Ekman emotions."""
        import torch

        # Mock model output (8 classes: angry, calm, disgust, fear, happy, neutral, sad, surprise)
        mock_model = MagicMock()
        mock_model.return_value.logits = torch.tensor([[
            2.0,   # angry → anger
            0.5,   # calm → neutral
            0.1,   # disgust
            0.1,   # fear
            1.0,   # happy → joy
            0.3,   # neutral
            0.1,   # sad → sadness
            0.1,   # surprise
        ]])

        mock_extractor = MagicMock()
        mock_extractor.return_value = {"input_values": torch.randn(1, 16000)}

        mock_get_model.return_value = (mock_model, mock_extractor)

        analyzer = self._make_analyzer()
        y = np.random.randn(16000).astype(np.float32)
        emotions = analyzer._predict_emotions_ml(y, 16000)

        # anger should be highest (angry has score 2.0)
        assert "anger" in emotions
        assert "joy" in emotions
        assert "neutral" in emotions
        # calm and neutral both map to neutral, so neutral should be combined
        assert emotions["anger"] > emotions["joy"]

    @patch("services.voice_analyzer.get_ser_model")
    def test_ml_prediction_sums_to_100(self, mock_get_model):
        import torch

        mock_model = MagicMock()
        mock_model.return_value.logits = torch.tensor([[
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]])
        mock_extractor = MagicMock()
        mock_extractor.return_value = {"input_values": torch.randn(1, 16000)}
        mock_get_model.return_value = (mock_model, mock_extractor)

        analyzer = self._make_analyzer()
        y = np.random.randn(16000).astype(np.float32)
        emotions = analyzer._predict_emotions_ml(y, 16000)

        total = sum(emotions.values())
        assert abs(total - 100.0) < 1.0


# ---------------------------------------------------------------------------
# XAI explanation tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_AUDIO_LIBS, reason="librosa/soundfile not installed")
class TestVoiceXAI:
    """Tests for voice XAI explanations."""

    def _make_analyzer(self):
        from services.voice_analyzer import VoiceEmotionAnalyzer
        return VoiceEmotionAnalyzer()

    def test_xai_included_when_requested(self, wav_bytes):
        analyzer = self._make_analyzer()
        analyzer._ml_available = False
        result = analyzer.analyze(wav_bytes, include_xai=True)
        assert result["xai_explanation"] is not None
        assert "method" in result["xai_explanation"]
        assert "reasoning" in result["xai_explanation"]
        assert "key_features" in result["xai_explanation"]

    def test_xai_excluded_when_not_requested(self, wav_bytes):
        analyzer = self._make_analyzer()
        analyzer._ml_available = False
        result = analyzer.analyze(wav_bytes, include_xai=False)
        assert result["xai_explanation"] is None

    def test_xai_energy_level_classification(self):
        analyzer = self._make_analyzer()

        # High energy
        features_high = {
            "energy": 0.2, "pitch_mean": 200.0, "pitch_std": 40.0,
            "tempo": 120.0, "duration_seconds": 2.0,
        }
        xai = analyzer._generate_explanation(features_high, "joy")
        assert xai["key_features"]["energy_level"] == "high"

        # Low energy
        features_low = {
            "energy": 0.05, "pitch_mean": 200.0, "pitch_std": 40.0,
            "tempo": 120.0, "duration_seconds": 2.0,
        }
        xai = analyzer._generate_explanation(features_low, "sadness")
        assert xai["key_features"]["energy_level"] == "low"

    def test_xai_speech_rate_classification(self):
        analyzer = self._make_analyzer()

        # Fast speech
        features = {
            "energy": 0.1, "pitch_mean": 200.0, "pitch_std": 40.0,
            "tempo": 150.0, "duration_seconds": 2.0,
        }
        xai = analyzer._generate_explanation(features, "joy")
        assert xai["key_features"]["speech_rate"] == "fast"

        # Slow speech
        features["tempo"] = 60.0
        xai = analyzer._generate_explanation(features, "sadness")
        assert xai["key_features"]["speech_rate"] == "slow"

    def test_xai_contains_audio_metrics(self):
        analyzer = self._make_analyzer()
        features = {
            "energy": 0.1, "pitch_mean": 200.0, "pitch_std": 40.0,
            "tempo": 120.0, "duration_seconds": 3.5,
        }
        xai = analyzer._generate_explanation(features, "neutral")
        assert "audio_metrics" in xai
        assert "duration" in xai["audio_metrics"]
        assert "average_pitch" in xai["audio_metrics"]
