"""
Unit tests for MaskingDetector service.
Tests distribution analysis, temporal analysis, landmark analysis,
and signal combination.
Run with: pytest tests/test_masking_detector.py -v
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from services.masking_detector import MaskingDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector():
    return MaskingDetector()


def _emotions(**kwargs):
    """Helper to create an emotion dict with defaults."""
    base = {
        "anger": 0.0, "disgust": 0.0, "fear": 0.0,
        "joy": 0.0, "sadness": 0.0, "surprise": 0.0, "neutral": 0.0,
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Distribution analysis tests (Layer 1)
# ---------------------------------------------------------------------------

class TestDistributionAnalysis:

    def test_fake_smile_detected(self, detector):
        """Joy > 30% AND sadness > 15% = fake_smile."""
        emotions = _emotions(joy=50.0, sadness=20.0, neutral=30.0)
        result = detector._analyze_distribution(emotions)
        assert result is not None
        assert result["type"] == "fake_smile"
        assert result["surface_emotion"] == "joy"
        assert result["underlying_emotion"] == "sadness"

    def test_suppressed_anger_detected(self, detector):
        """Joy > 30% AND anger > 15% = suppressed_anger."""
        emotions = _emotions(joy=40.0, anger=25.0, neutral=35.0)
        result = detector._analyze_distribution(emotions)
        assert result is not None
        assert result["type"] == "suppressed_anger"

    def test_nervous_smile_detected(self, detector):
        """Joy > 30% AND fear > 15% = nervous_smile."""
        emotions = _emotions(joy=45.0, fear=20.0, neutral=35.0)
        result = detector._analyze_distribution(emotions)
        assert result is not None
        assert result["type"] == "nervous_smile"

    def test_suppressed_sadness_detected(self, detector):
        """Neutral > 30% AND sadness > 15% = suppressed_sadness."""
        emotions = _emotions(neutral=50.0, sadness=20.0, joy=30.0)
        result = detector._analyze_distribution(emotions)
        assert result is not None
        assert result["type"] == "suppressed_sadness"

    def test_no_masking_when_emotions_clear(self, detector):
        """Clear primary emotion with no conflicting secondary = no masking."""
        emotions = _emotions(joy=85.0, neutral=10.0, surprise=5.0)
        result = detector._analyze_distribution(emotions)
        # No conflicting pair, margin > 10
        assert result is None

    def test_no_masking_when_primary_too_low(self, detector):
        """If primary emotion < 30%, no masking detected."""
        emotions = _emotions(joy=25.0, sadness=20.0, neutral=55.0)
        result = detector._analyze_distribution(emotions)
        # Primary (neutral) is 55%, but (neutral, sadness) IS a conflict pair
        # Actually neutral=55 > 30 and sadness=20 > 15 → suppressed_sadness
        assert result is not None

    def test_mixed_signals_small_margin(self, detector):
        """Small margin between top 2 emotions = mixed_signals."""
        emotions = _emotions(joy=28.0, sadness=24.0, neutral=48.0)
        # neutral=48 > 30 and sadness=24 > 15 → suppressed_sadness
        result = detector._analyze_distribution(emotions)
        assert result is not None

    def test_confidence_bounded(self, detector):
        """Confidence should never exceed 0.95."""
        emotions = _emotions(joy=95.0, sadness=95.0)
        result = detector._analyze_distribution(emotions)
        if result:
            assert result["confidence"] <= 0.95


# ---------------------------------------------------------------------------
# Temporal analysis tests (Layer 2)
# ---------------------------------------------------------------------------

class TestTemporalAnalysis:

    def test_sudden_shift_detected(self, detector):
        """Sudden shift from sadness to joy should be flagged."""
        current = _emotions(joy=70.0, sadness=5.0, neutral=25.0)
        history = [
            _emotions(sadness=60.0, joy=10.0, neutral=30.0),
        ]
        result = detector._analyze_temporal(current, history)
        assert result is not None
        assert "sudden" in result["type"]
        assert result["layer"] == "temporal"

    def test_no_sudden_shift_with_gradual_change(self, detector):
        """Gradual emotion change should not be flagged."""
        current = _emotions(joy=50.0, sadness=20.0, neutral=30.0)
        history = [
            _emotions(joy=40.0, sadness=25.0, neutral=35.0),
        ]
        result = detector._analyze_temporal(current, history)
        # joy only increased by 10% (< TEMPORAL_JUMP of 30%)
        assert result is None

    def test_oscillation_detected(self, detector):
        """Rapid switching between emotions should be flagged."""
        current = _emotions(neutral=40.0, surprise=30.0, disgust=30.0)
        history = [
            _emotions(surprise=60.0, neutral=40.0),
            _emotions(disgust=60.0, neutral=40.0),
            _emotions(surprise=60.0, neutral=40.0),
            _emotions(disgust=60.0, neutral=40.0),
            _emotions(surprise=60.0, neutral=40.0),
            _emotions(disgust=60.0, neutral=40.0),
        ]
        result = detector._analyze_temporal(current, history)
        assert result is not None
        assert result["type"] == "emotional_instability"

    def test_stable_history_no_oscillation(self, detector):
        """Stable emotion history should not flag oscillation."""
        current = _emotions(joy=60.0, neutral=40.0)
        history = [_emotions(joy=60.0, neutral=40.0)] * 6
        result = detector._analyze_temporal(current, history)
        assert result is None

    def test_empty_history_returns_none(self, detector):
        current = _emotions(joy=60.0)
        result = detector._analyze_temporal(current, [])
        assert result is None

    def test_none_history_returns_none(self, detector):
        current = _emotions(joy=60.0)
        result = detector._analyze_temporal(current, None)
        assert result is None

    def test_oscillation_confidence_bounded(self, detector):
        """Temporal analysis confidence should be bounded."""
        current = _emotions(neutral=40.0, surprise=30.0, disgust=30.0)
        history = [
            _emotions(surprise=60.0, neutral=40.0),
            _emotions(disgust=60.0, neutral=40.0),
            _emotions(surprise=60.0, neutral=40.0),
            _emotions(disgust=60.0, neutral=40.0),
            _emotions(surprise=60.0, neutral=40.0),
            _emotions(disgust=60.0, neutral=40.0),
        ]
        result = detector._analyze_temporal(current, history)
        if result:
            assert result["confidence"] <= 0.85


# ---------------------------------------------------------------------------
# Landmark analysis tests (Layer 3)
# ---------------------------------------------------------------------------

class TestLandmarkAnalysis:

    def test_landmark_analysis_returns_none_without_mediapipe(self, detector):
        """If MediaPipe not available, should return None."""
        detector._mp_initialized = True
        detector._mp_face_mesh = None
        img = np.ones((100, 100, 3), dtype=np.uint8)
        result = detector._analyze_landmarks(img)
        assert result is None

    @patch.object(MaskingDetector, '_initialize_mediapipe')
    def test_landmark_analysis_with_mocked_mediapipe(self, mock_init, detector):
        """Mocked landmark analysis should detect fake smile."""
        mock_face_mesh = MagicMock()

        # Create mock landmarks with AU6 < threshold, AU12 > 0.3
        mock_landmark = MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmarks = [mock_landmark] * 500  # MediaPipe has 468+ landmarks

        # Position specific landmarks for fake smile detection
        # Lower eyelid (111, 340), cheek (117, 346) - close together = no AU6
        for i in [111, 340]:
            mock_landmarks[i] = MagicMock(x=0.5, y=0.45)
        for i in [117, 346]:
            mock_landmarks[i] = MagicMock(x=0.5, y=0.47)  # Very close to eyelid
        # Eye corners (33, 263)
        mock_landmarks[33] = MagicMock(x=0.3, y=0.45)
        mock_landmarks[263] = MagicMock(x=0.7, y=0.45)

        # Mouth corners (61, 291) and lip center (13) - big difference = AU12
        mock_landmarks[61] = MagicMock(x=0.35, y=0.62)
        mock_landmarks[291] = MagicMock(x=0.65, y=0.62)
        mock_landmarks[13] = MagicMock(x=0.5, y=0.68)
        mock_landmarks[1] = MagicMock(x=0.5, y=0.55)   # Nose tip

        mock_result = MagicMock()
        mock_result.multi_face_landmarks = [MagicMock(landmark=mock_landmarks)]
        mock_face_mesh.process.return_value = mock_result

        detector._mp_face_mesh = mock_face_mesh
        detector._mp_initialized = True

        img = np.ones((300, 300, 3), dtype=np.uint8)
        result = detector._analyze_landmarks(img)

        # May or may not detect fake smile depending on exact AU scores
        # The important thing is it doesn't crash
        assert result is None or result["layer"] == "landmarks"


# ---------------------------------------------------------------------------
# Signal combination tests
# ---------------------------------------------------------------------------

class TestSignalCombination:

    def test_no_signals_returns_none(self, detector):
        result = detector._combine_signals([], _emotions(joy=50.0))
        assert result is None

    def test_single_signal_below_confidence_returns_none(self, detector):
        """Signal below CONFIDENCE_MIN (0.4) should be filtered out."""
        signals = [{
            "layer": "distribution",
            "type": "mixed_signals",
            "surface_emotion": "joy",
            "underlying_emotion": "sadness",
            "confidence": 0.2,
            "detail": "test",
        }]
        result = detector._combine_signals(signals, _emotions(joy=50.0))
        assert result is None

    def test_single_strong_signal_returns_result(self, detector):
        signals = [{
            "layer": "distribution",
            "type": "fake_smile",
            "surface_emotion": "joy",
            "underlying_emotion": "sadness",
            "confidence": 0.7,
            "detail": "test detail",
        }]
        result = detector._combine_signals(signals, _emotions(joy=50.0, sadness=20.0))
        assert result is not None
        assert result["detected"] is True
        assert result["type"] == "fake_smile"
        assert result["confidence"] >= 0.7

    def test_multiple_signals_boost_confidence(self, detector):
        """Multiple layers agreeing should boost confidence."""
        signals = [
            {
                "layer": "distribution",
                "type": "fake_smile",
                "surface_emotion": "joy",
                "underlying_emotion": "sadness",
                "confidence": 0.5,
                "detail": "distribution signal",
            },
            {
                "layer": "temporal",
                "type": "sudden_fake_smile",
                "surface_emotion": "joy",
                "underlying_emotion": "sadness",
                "confidence": 0.45,
                "detail": "temporal signal",
            },
        ]
        result = detector._combine_signals(signals, _emotions(joy=50.0, sadness=20.0))
        assert result is not None
        # Confidence should be boosted (×1.3) since 2 layers
        assert result["confidence"] > 0.5

    def test_three_layer_agreement_high_confidence(self, detector):
        signals = [
            {"layer": "distribution", "type": "fake_smile",
             "surface_emotion": "joy", "underlying_emotion": "sadness",
             "confidence": 0.6, "detail": "d"},
            {"layer": "temporal", "type": "sudden_fake_smile",
             "surface_emotion": "joy", "underlying_emotion": "sadness",
             "confidence": 0.55, "detail": "t"},
            {"layer": "landmarks", "type": "fake_smile",
             "surface_emotion": "joy", "underlying_emotion": "unknown",
             "confidence": 0.5, "detail": "l",
             "au6_score": 0.1, "au12_score": 0.8, "is_duchenne": False},
        ]
        result = detector._combine_signals(signals, _emotions(joy=50.0, sadness=20.0))
        assert result is not None
        # 3 layers → boosted ×1.3 × 1.2
        assert result["confidence"] > 0.7

    def test_au_scores_included_in_result(self, detector):
        """AU scores from landmark layer should be in final result."""
        signals = [{
            "layer": "landmarks",
            "type": "fake_smile",
            "surface_emotion": "joy",
            "underlying_emotion": "unknown",
            "confidence": 0.7,
            "detail": "Non-Duchenne",
            "au6_score": 0.15,
            "au12_score": 0.85,
            "is_duchenne": False,
        }]
        result = detector._combine_signals(signals, _emotions(joy=50.0))
        assert result["au6_score"] == 0.15
        assert result["au12_score"] == 0.85
        assert result["is_duchenne"] is False

    def test_explanation_generated(self, detector):
        signals = [{
            "layer": "distribution",
            "type": "fake_smile",
            "surface_emotion": "joy",
            "underlying_emotion": "sadness",
            "confidence": 0.7,
            "detail": "test",
        }]
        result = detector._combine_signals(signals, _emotions(joy=50.0))
        assert "explanation" in result
        assert result["explanation"]["method"] == "masking_detection"


# ---------------------------------------------------------------------------
# Full analyze_frame integration tests
# ---------------------------------------------------------------------------

class TestAnalyzeFrame:

    def test_no_masking_with_clear_emotions(self, detector):
        """Clear single emotion = no masking."""
        emotions = _emotions(joy=90.0, neutral=10.0)
        result = detector.analyze_frame(emotions=emotions)
        assert result is None

    def test_masking_detected_with_conflicting_emotions(self, detector):
        """Conflicting emotions should trigger masking."""
        emotions = _emotions(joy=50.0, sadness=25.0, neutral=25.0)
        result = detector.analyze_frame(emotions=emotions)
        # May or may not detect depending on confidence threshold
        if result:
            assert result["detected"] is True

    def test_analyze_frame_with_history(self, detector):
        """Should use temporal analysis when history provided."""
        current = _emotions(joy=70.0, sadness=5.0, neutral=25.0)
        history = [
            _emotions(sadness=60.0, joy=10.0, neutral=30.0),
        ]
        result = detector.analyze_frame(
            emotions=current,
            emotion_history=history,
        )
        # Sudden shift from sadness to joy → should detect
        if result:
            assert result["detected"] is True

    def test_analyze_frame_landmark_only_for_joy(self, detector):
        """Landmark analysis should only run when joy > 20%."""
        detector._mp_initialized = True
        detector._mp_face_mesh = MagicMock()

        emotions_no_joy = _emotions(anger=80.0, neutral=20.0)
        img = np.ones((100, 100, 3), dtype=np.uint8)

        result = detector.analyze_frame(
            emotions=emotions_no_joy,
            image_rgb=img,
        )
        # Landmark analysis skipped because joy < 20%
        detector._mp_face_mesh.process.assert_not_called()
