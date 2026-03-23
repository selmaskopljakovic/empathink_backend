"""
Unit tests for FaceEmotionAnalyzer service.
Tests face detection, emotion classification, label normalization,
DeepFace/FER backends, and XAI explanations.
Run with: pytest tests/test_face_analyzer.py -v
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock


# ---------------------------------------------------------------------------
# FaceEmotionAnalyzer unit tests
# ---------------------------------------------------------------------------

class TestFaceAnalyzerBasic:
    """Basic contract tests for FaceEmotionAnalyzer."""

    def _make_analyzer(self):
        from services.face_analyzer import FaceEmotionAnalyzer
        return FaceEmotionAnalyzer()

    def test_label_normalization_happy_to_joy(self):
        """FER label 'happy' should be normalized to 'joy'."""
        analyzer = self._make_analyzer()
        emotions = {"happy": 80.0, "sad": 10.0, "angry": 5.0, "neutral": 5.0}
        normalized = analyzer._normalize_emotions(emotions)
        assert "joy" in normalized
        assert "happy" not in normalized
        assert normalized["joy"] == 80.0

    def test_label_normalization_sad_to_sadness(self):
        analyzer = self._make_analyzer()
        emotions = {"sad": 70.0, "happy": 30.0}
        normalized = analyzer._normalize_emotions(emotions)
        assert "sadness" in normalized
        assert normalized["sadness"] == 70.0

    def test_label_normalization_angry_to_anger(self):
        analyzer = self._make_analyzer()
        emotions = {"angry": 60.0, "neutral": 40.0}
        normalized = analyzer._normalize_emotions(emotions)
        assert "anger" in normalized
        assert normalized["anger"] == 60.0

    def test_label_normalization_preserves_unchanged(self):
        """Labels that don't need normalization should remain."""
        analyzer = self._make_analyzer()
        emotions = {"neutral": 50.0, "surprise": 30.0, "fear": 20.0}
        normalized = analyzer._normalize_emotions(emotions)
        assert normalized == emotions

    # --- analyze_image with mocked backend ---

    @patch("services.face_analyzer.masking_detector")
    def test_analyze_image_no_face_detected(self, mock_masking):
        """When no face is detected, should return neutral with face_detected=False."""
        import cv2

        analyzer = self._make_analyzer()
        analyzer._is_initialized = True
        analyzer._backend = "fer"
        analyzer._fer_detector = MagicMock()
        analyzer._fer_detector.detect_emotions.return_value = []

        # Create a valid image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        _, buf = cv2.imencode('.jpg', img)
        image_bytes = buf.tobytes()

        result = analyzer.analyze_image(image_bytes, include_xai=True)
        assert result["success"] is True
        assert result["face_detected"] is False
        assert result["primary_emotion"] == "neutral"

    @patch("services.face_analyzer.masking_detector")
    def test_analyze_image_with_face_detected(self, mock_masking):
        """When a face is detected, should return emotions and face box."""
        import cv2

        mock_masking.analyze_frame.return_value = None

        analyzer = self._make_analyzer()
        analyzer._is_initialized = True
        analyzer._backend = "fer"
        analyzer._fer_detector = MagicMock()
        analyzer._fer_detector.detect_emotions.return_value = [{
            "emotions": {
                "happy": 0.7, "sad": 0.1, "angry": 0.05,
                "neutral": 0.08, "surprise": 0.04, "fear": 0.02, "disgust": 0.01
            },
            "box": (50, 60, 200, 200)
        }]

        img = np.ones((300, 300, 3), dtype=np.uint8) * 128
        _, buf = cv2.imencode('.jpg', img)
        image_bytes = buf.tobytes()

        result = analyzer.analyze_image(image_bytes, include_xai=False)

        assert result["success"] is True
        assert result["face_detected"] is True
        assert result["primary_emotion"] == "joy"  # happy → joy
        assert "face_box" in result
        assert result["face_box"]["x"] == 50
        assert result["face_box"]["width"] == 200

    @patch("services.face_analyzer.masking_detector")
    def test_analyze_image_emotions_normalized(self, mock_masking):
        """Emotions should be normalized from FER labels to Ekman labels."""
        import cv2

        mock_masking.analyze_frame.return_value = None

        analyzer = self._make_analyzer()
        analyzer._is_initialized = True
        analyzer._backend = "fer"
        analyzer._fer_detector = MagicMock()
        analyzer._fer_detector.detect_emotions.return_value = [{
            "emotions": {
                "happy": 0.5, "sad": 0.2, "angry": 0.1,
                "neutral": 0.1, "surprise": 0.05, "fear": 0.03, "disgust": 0.02
            },
            "box": (0, 0, 100, 100)
        }]

        img = np.ones((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', img)

        result = analyzer.analyze_image(buf.tobytes(), include_xai=False)

        # Check normalized labels
        assert "joy" in result["emotions"]
        assert "sadness" in result["emotions"]
        assert "anger" in result["emotions"]
        assert "happy" not in result["emotions"]
        assert "sad" not in result["emotions"]
        assert "angry" not in result["emotions"]

    def test_analyze_image_invalid_bytes_returns_error(self):
        """Invalid image bytes should return error."""
        analyzer = self._make_analyzer()
        analyzer._is_initialized = True
        analyzer._backend = "fer"
        analyzer._fer_detector = MagicMock()

        result = analyzer.analyze_image(b"not an image", include_xai=False)
        assert result["success"] is False

    def test_analyze_image_empty_bytes_returns_error(self):
        analyzer = self._make_analyzer()
        result = analyzer.analyze_image(b"", include_xai=False)
        assert result["success"] is False

    # --- Masking integration ---

    @patch("services.face_analyzer.masking_detector")
    def test_masking_result_included_when_detected(self, mock_masking):
        """If masking is detected, result should include masking data."""
        import cv2

        mock_masking.analyze_frame.return_value = {
            "detected": True,
            "type": "fake_smile",
            "confidence": 0.75,
        }

        analyzer = self._make_analyzer()
        analyzer._is_initialized = True
        analyzer._backend = "fer"
        analyzer._fer_detector = MagicMock()
        analyzer._fer_detector.detect_emotions.return_value = [{
            "emotions": {"happy": 0.6, "sad": 0.2, "angry": 0.1,
                         "neutral": 0.05, "surprise": 0.03, "fear": 0.01, "disgust": 0.01},
            "box": (0, 0, 100, 100)
        }]

        img = np.ones((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', img)

        result = analyzer.analyze_image(buf.tobytes(), include_xai=False)
        assert "masking" in result
        assert result["masking"]["detected"] is True
        assert result["masking"]["type"] == "fake_smile"

    @patch("services.face_analyzer.masking_detector")
    def test_masking_error_does_not_crash(self, mock_masking):
        """Masking detection error should not crash image analysis."""
        import cv2

        mock_masking.analyze_frame.side_effect = RuntimeError("mediapipe crash")

        analyzer = self._make_analyzer()
        analyzer._is_initialized = True
        analyzer._backend = "fer"
        analyzer._fer_detector = MagicMock()
        analyzer._fer_detector.detect_emotions.return_value = [{
            "emotions": {"happy": 0.6, "sad": 0.1, "angry": 0.1,
                         "neutral": 0.1, "surprise": 0.05, "fear": 0.03, "disgust": 0.02},
            "box": (0, 0, 100, 100)
        }]

        img = np.ones((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', img)

        result = analyzer.analyze_image(buf.tobytes(), include_xai=False)
        assert result["success"] is True
        assert "masking" not in result


# ---------------------------------------------------------------------------
# DeepFace backend tests
# ---------------------------------------------------------------------------

class TestDeepFaceBackend:
    """Tests for the DeepFace analysis path."""

    def _make_analyzer(self):
        from services.face_analyzer import FaceEmotionAnalyzer
        analyzer = FaceEmotionAnalyzer()
        analyzer._is_initialized = True
        analyzer._backend = "deepface"
        return analyzer

    def test_deepface_result_conversion(self):
        """DeepFace results should be converted to FER-compatible format."""
        analyzer = self._make_analyzer()
        analyzer._deepface = MagicMock()
        analyzer._deepface.analyze.return_value = [{
            "emotion": {
                "happy": 70.0, "sad": 10.0, "angry": 5.0,
                "neutral": 8.0, "surprise": 4.0, "fear": 2.0, "disgust": 1.0
            },
            "region": {"x": 50, "y": 50, "w": 200, "h": 200}
        }]

        img_rgb = np.ones((300, 300, 3), dtype=np.uint8)
        result = analyzer._analyze_with_deepface(img_rgb)

        assert len(result) == 1
        assert "emotions" in result[0]
        assert "box" in result[0]
        # Emotions should be 0-1 (divided by 100)
        assert result[0]["emotions"]["happy"] == 0.7

    def test_deepface_empty_result(self):
        analyzer = self._make_analyzer()
        analyzer._deepface = MagicMock()
        analyzer._deepface.analyze.return_value = []

        img_rgb = np.ones((100, 100, 3), dtype=np.uint8)
        result = analyzer._analyze_with_deepface(img_rgb)
        assert result == []

    def test_deepface_fallback_to_fer_on_error(self):
        """If DeepFace crashes, should fallback to FER."""
        analyzer = self._make_analyzer()
        analyzer._deepface = MagicMock()
        analyzer._deepface.analyze.side_effect = RuntimeError("GPU error")
        analyzer._fer_detector = MagicMock()
        analyzer._fer_detector.detect_emotions.return_value = [{
            "emotions": {"neutral": 1.0},
            "box": (0, 0, 100, 100)
        }]

        img_rgb = np.ones((100, 100, 3), dtype=np.uint8)
        result = analyzer._analyze_with_deepface(img_rgb)
        assert len(result) == 1
        analyzer._fer_detector.detect_emotions.assert_called_once()


# ---------------------------------------------------------------------------
# XAI explanation tests
# ---------------------------------------------------------------------------

class TestFaceXAI:
    """Tests for face emotion XAI explanations."""

    def _make_analyzer(self):
        from services.face_analyzer import FaceEmotionAnalyzer
        return FaceEmotionAnalyzer()

    def test_explanation_contains_facs(self):
        """XAI explanation should reference Facial Action Coding System."""
        analyzer = self._make_analyzer()
        emotions = {"happy": 80.0, "sad": 10.0, "neutral": 10.0}
        xai = analyzer.generate_explanation(emotions, "happy")

        assert xai["method"] == "facial_action_coding_system"
        assert "facial_action_units" in xai
        assert len(xai["facial_action_units"]) > 0

    def test_explanation_for_each_emotion(self):
        """Each of the 7 emotions should produce a valid explanation."""
        analyzer = self._make_analyzer()
        emotions_list = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]

        for emotion in emotions_list:
            emotions = {e: 10.0 for e in emotions_list}
            emotions[emotion] = 50.0
            xai = analyzer.generate_explanation(emotions, emotion)
            assert xai["method"] == "facial_action_coding_system"
            assert "reasoning" in xai
            assert len(xai["reasoning"]) > 0

    def test_explanation_confidence_breakdown(self):
        """Explanation should contain top-4 emotion breakdown."""
        analyzer = self._make_analyzer()
        emotions = {
            "happy": 60.0, "sad": 15.0, "angry": 10.0,
            "neutral": 8.0, "surprise": 4.0, "fear": 2.0, "disgust": 1.0
        }
        xai = analyzer.generate_explanation(emotions, "happy")
        assert "confidence_breakdown" in xai
        assert len(xai["confidence_breakdown"]) <= 4

    def test_happy_explanation_mentions_au6_au12(self):
        """Happy explanation should mention AU6 and AU12."""
        analyzer = self._make_analyzer()
        emotions = {"happy": 90.0}
        xai = analyzer.generate_explanation(emotions, "happy")
        au_text = " ".join(xai["facial_action_units"])
        assert "AU6" in au_text
        assert "AU12" in au_text


# ---------------------------------------------------------------------------
# Fast frame analysis tests
# ---------------------------------------------------------------------------

class TestFastFrameAnalysis:
    """Tests for analyze_frame_fast() used in live camera."""

    def _make_analyzer(self):
        from services.face_analyzer import FaceEmotionAnalyzer
        return FaceEmotionAnalyzer()

    @patch("services.face_analyzer.masking_detector")
    def test_fast_analysis_invalid_base64_returns_no_face(self, mock_masking):
        """Invalid base64 should return face_detected=False."""
        analyzer = self._make_analyzer()
        result = analyzer.analyze_frame_fast("not-valid-base64")
        assert result["face_detected"] is False

    @patch("services.face_analyzer.masking_detector")
    def test_fast_analysis_valid_frame_with_mock(self, mock_masking):
        """Valid frame with mocked FER should return emotions."""
        import cv2
        import base64

        mock_masking.analyze_frame.return_value = None

        img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        _, buf = cv2.imencode('.jpg', img)
        frame_b64 = base64.b64encode(buf.tobytes()).decode()

        # Patch FER at the import location inside the method
        with patch.dict("sys.modules", {"fer": MagicMock()}) as _:
            import sys
            mock_fer_class = MagicMock()
            mock_fer_instance = MagicMock()
            mock_fer_instance.detect_emotions.return_value = [{
                "emotions": {"happy": 0.8, "sad": 0.05, "angry": 0.05,
                             "neutral": 0.05, "surprise": 0.03, "fear": 0.01, "disgust": 0.01},
                "box": (20, 20, 100, 100)
            }]
            mock_fer_class.return_value = mock_fer_instance
            sys.modules["fer"].FER = mock_fer_class

            analyzer = self._make_analyzer()
            result = analyzer.analyze_frame_fast(frame_b64)

            assert result["face_detected"] is True
            assert "joy" in result["emotions"]  # happy → joy
            assert result["face_box"] is not None
            # Box should be scaled back (divided by 0.5 scale factor)
            assert result["face_box"]["x"] == 40  # 20 / 0.5

    @patch("services.face_analyzer.masking_detector")
    def test_fast_analysis_no_face_in_frame(self, mock_masking):
        """Frame with no face should return face_detected=False."""
        import cv2
        import base64

        img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        _, buf = cv2.imencode('.jpg', img)
        frame_b64 = base64.b64encode(buf.tobytes()).decode()

        with patch.dict("sys.modules", {"fer": MagicMock()}) as _:
            import sys
            mock_fer_class = MagicMock()
            mock_fer_instance = MagicMock()
            mock_fer_instance.detect_emotions.return_value = []
            mock_fer_class.return_value = mock_fer_instance
            sys.modules["fer"].FER = mock_fer_class

            analyzer = self._make_analyzer()
            result = analyzer.analyze_frame_fast(frame_b64)
            assert result["face_detected"] is False
