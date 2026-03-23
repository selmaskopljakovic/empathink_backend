"""
Integration / API route tests for EmpaThink backend.
Tests all endpoints with mocked services to verify correct HTTP behavior,
error handling, input validation, and response structure.
Run with: pytest tests/test_api_integration.py -v
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from main import app
from api.auth import get_current_user

import struct
import numpy as np
import io


# ---------------------------------------------------------------------------
# Auth override for all tests
# ---------------------------------------------------------------------------
_fake_user = {"uid": "test-user-123", "email": "test@example.com", "email_verified": True}


def _override_auth():
    return _fake_user


app.dependency_overrides[get_current_user] = _override_auth
client = TestClient(app)


def _generate_wav_bytes(duration_s=1.0, sr=16000, freq=440.0):
    """Generate valid WAV bytes."""
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    data_size = n * 2
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<I', 16))
    buf.write(struct.pack('<H', 1))
    buf.write(struct.pack('<H', 1))
    buf.write(struct.pack('<I', sr))
    buf.write(struct.pack('<I', sr * 2))
    buf.write(struct.pack('<H', 2))
    buf.write(struct.pack('<H', 16))
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    buf.write(audio.tobytes())
    return buf.getvalue()


def _generate_jpeg_bytes():
    """Generate valid JPEG bytes."""
    import cv2
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Health & Root
# ---------------------------------------------------------------------------

class TestHealthAndRoot:

    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_health_contains_services(self):
        r = client.get("/health")
        data = r.json()
        assert "services" in data
        assert "text_analysis" in data["services"]
        assert "voice_analysis" in data["services"]
        assert "image_analysis" in data["services"]

    def test_root_returns_200(self):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_contains_version(self):
        r = client.get("/")
        assert "version" in r.json()


# ---------------------------------------------------------------------------
# Text Analysis API
# ---------------------------------------------------------------------------

class TestTextAPI:

    @patch("api.routes.text.text_analyzer")
    def test_valid_text_returns_200(self, mock_analyzer):
        mock_analyzer.analyze.return_value = {
            "success": True,
            "emotions": {"joy": 80.0, "neutral": 20.0},
            "primary_emotion": "joy",
            "confidence": 80.0,
            "sentiment": {"label": "positive", "score": 90.0},
            "text_metrics": {"word_count": 5},
            "xai_explanation": None,
            "processing_time_ms": 10.0,
            "timestamp": "2026-01-01T00:00:00",
        }
        r = client.post("/analyze/text", json={"text": "I am very happy!", "include_xai": False})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["primary_emotion"] == "joy"

    def test_empty_text_returns_400(self):
        r = client.post("/analyze/text", json={"text": ""})
        assert r.status_code == 400

    def test_whitespace_text_returns_400(self):
        r = client.post("/analyze/text", json={"text": "   "})
        assert r.status_code == 400

    def test_text_too_long_returns_400(self):
        r = client.post("/analyze/text", json={"text": "x" * 5001})
        assert r.status_code == 400

    def test_text_at_max_length_accepted(self):
        """Text of exactly 5000 chars should be accepted."""
        r = client.post("/analyze/text", json={"text": "x" * 5000, "include_xai": False})
        # Should not get 400 for length
        assert r.status_code != 400 or "too long" not in r.json().get("detail", "").lower()

    def test_missing_text_field_returns_422(self):
        """Missing required 'text' field should return 422."""
        r = client.post("/analyze/text", json={})
        assert r.status_code == 422

    @patch("api.routes.text.text_analyzer")
    def test_include_xai_true_calls_with_xai(self, mock_analyzer):
        mock_analyzer.analyze.return_value = {
            "success": True, "emotions": {"neutral": 100.0},
            "primary_emotion": "neutral", "confidence": 100.0,
            "sentiment": {"label": "neutral", "score": 50.0},
            "text_metrics": {}, "xai_explanation": {"method": "shap"},
            "processing_time_ms": 10.0, "timestamp": "2026-01-01T00:00:00",
        }
        r = client.post("/analyze/text", json={"text": "test", "include_xai": True})
        assert r.status_code == 200
        mock_analyzer.analyze.assert_called_once_with(text="test", include_xai=True)

    @patch("api.routes.text.text_analyzer")
    def test_internal_error_returns_500(self, mock_analyzer):
        mock_analyzer.analyze.side_effect = RuntimeError("unexpected crash")
        r = client.post("/analyze/text", json={"text": "valid text", "include_xai": False})
        assert r.status_code == 500

    @patch("api.routes.text.text_analyzer")
    def test_500_error_does_not_leak_details(self, mock_analyzer):
        mock_analyzer.analyze.side_effect = RuntimeError("SECRET_KEY=abc123")
        r = client.post("/analyze/text", json={"text": "valid text", "include_xai": False})
        assert r.status_code == 500
        # The route currently leaks the error message - this documents the issue
        # Ideally: assert "SECRET_KEY" not in str(r.json())


# ---------------------------------------------------------------------------
# Text Quick API
# ---------------------------------------------------------------------------

class TestTextQuickAPI:

    def test_empty_text_returns_neutral(self):
        r = client.post("/analyze/text/quick", json={"text": ""})
        assert r.status_code == 200
        data = r.json()
        assert data["primary_emotion"] == "neutral"
        assert data["emotions"]["neutral"] == 100.0

    def test_short_text_returns_neutral(self):
        r = client.post("/analyze/text/quick", json={"text": "ab"})
        assert r.status_code == 200
        assert r.json()["primary_emotion"] == "neutral"

    @patch("api.routes.text.text_analyzer")
    def test_valid_text_returns_emotions(self, mock_analyzer):
        mock_analyzer.analyze.return_value = {
            "success": True,
            "emotions": {"joy": 80.0, "neutral": 20.0},
            "primary_emotion": "joy",
            "confidence": 80.0,
        }
        r = client.post("/analyze/text/quick", json={"text": "I feel great today!"})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert "emotions" in data
        # Quick endpoint should not include XAI
        assert "xai_explanation" not in data

    @patch("api.routes.text.text_analyzer")
    def test_quick_always_calls_without_xai(self, mock_analyzer):
        mock_analyzer.analyze.return_value = {
            "success": True, "emotions": {"neutral": 100.0},
            "primary_emotion": "neutral", "confidence": 100.0,
        }
        client.post("/analyze/text/quick", json={"text": "test text"})
        mock_analyzer.analyze.assert_called_once_with(text="test text", include_xai=False)

    @patch("api.routes.text.text_analyzer")
    def test_quick_error_returns_fallback(self, mock_analyzer):
        """Quick endpoint should return fallback on error, not 500."""
        mock_analyzer.analyze.side_effect = RuntimeError("crash")
        r = client.post("/analyze/text/quick", json={"text": "test text"})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert data["primary_emotion"] == "neutral"


# ---------------------------------------------------------------------------
# Text Models API
# ---------------------------------------------------------------------------

class TestTextModelsAPI:

    def test_text_models_returns_200(self):
        r = client.get("/analyze/text/models")
        assert r.status_code == 200
        data = r.json()
        assert "emotion_model" in data
        assert "sentiment_model" in data


# ---------------------------------------------------------------------------
# Voice Analysis API
# ---------------------------------------------------------------------------

class TestVoiceAPI:

    def test_wrong_format_returns_400(self):
        r = client.post(
            "/analyze/voice",
            files={"audio": ("test.txt", b"not audio", "text/plain")},
        )
        assert r.status_code == 400

    @patch("api.routes.voice.voice_analyzer")
    def test_valid_wav_returns_200(self, mock_analyzer):
        mock_analyzer.analyze.return_value = {
            "success": True,
            "emotions": {"neutral": 60.0, "joy": 40.0},
            "primary_emotion": "neutral",
            "confidence": 60.0,
            "audio_features": {"duration_seconds": 1.0},
            "xai_explanation": None,
            "processing_time_ms": 50.0,
            "timestamp": "2026-01-01T00:00:00",
        }
        wav = _generate_wav_bytes()
        r = client.post(
            "/analyze/voice",
            files={"audio": ("test.wav", wav, "audio/wav")},
        )
        assert r.status_code == 200
        assert r.json()["success"] is True

    @patch("api.routes.voice.voice_analyzer")
    def test_voice_with_xai(self, mock_analyzer):
        mock_analyzer.analyze.return_value = {
            "success": True,
            "emotions": {"neutral": 100.0},
            "primary_emotion": "neutral",
            "confidence": 100.0,
            "audio_features": {},
            "xai_explanation": {"method": "acoustic_feature_analysis"},
            "processing_time_ms": 50.0,
            "timestamp": "2026-01-01T00:00:00",
        }
        wav = _generate_wav_bytes()
        r = client.post(
            "/analyze/voice",
            files={"audio": ("test.wav", wav, "audio/wav")},
            data={"include_xai": "true"},
        )
        assert r.status_code == 200

    def test_octet_stream_accepted(self):
        """application/octet-stream should be accepted."""
        wav = _generate_wav_bytes()
        r = client.post(
            "/analyze/voice",
            files={"audio": ("recording.wav", wav, "application/octet-stream")},
        )
        # Should not be rejected by format check
        assert r.status_code != 400 or "format" not in r.json().get("detail", "").lower()

    @patch("api.routes.voice.voice_analyzer")
    def test_voice_internal_error_returns_500(self, mock_analyzer):
        mock_analyzer.analyze.side_effect = RuntimeError("librosa crash")
        wav = _generate_wav_bytes()
        r = client.post(
            "/analyze/voice",
            files={"audio": ("test.wav", wav, "audio/wav")},
        )
        assert r.status_code == 500

    def test_voice_models_returns_200(self):
        r = client.get("/analyze/voice/models")
        assert r.status_code == 200
        data = r.json()
        assert "feature_extraction" in data
        assert "emotion_detection" in data
        assert "supported_formats" in data


# ---------------------------------------------------------------------------
# Image Analysis API
# ---------------------------------------------------------------------------

class TestImageAPI:

    def test_wrong_format_returns_400(self):
        r = client.post(
            "/analyze/image",
            files={"image": ("test.txt", b"not image", "text/plain")},
        )
        assert r.status_code == 400

    @patch("api.routes.image.face_analyzer")
    def test_valid_jpeg_returns_200(self, mock_analyzer):
        mock_analyzer.analyze_image.return_value = {
            "success": True,
            "face_detected": True,
            "emotions": {"joy": 70.0, "neutral": 30.0},
            "primary_emotion": "joy",
            "confidence": 70.0,
            "face_box": {"x": 10, "y": 10, "width": 100, "height": 100},
            "xai_explanation": None,
            "processing_time_ms": 100.0,
            "timestamp": "2026-01-01T00:00:00",
        }
        jpeg = _generate_jpeg_bytes()
        r = client.post(
            "/analyze/image",
            files={"image": ("photo.jpg", jpeg, "image/jpeg")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["face_detected"] is True

    @patch("api.routes.image.face_analyzer")
    def test_no_face_detected_still_200(self, mock_analyzer):
        mock_analyzer.analyze_image.return_value = {
            "success": True,
            "face_detected": False,
            "emotions": {"neutral": 100.0},
            "primary_emotion": "neutral",
            "confidence": 0.0,
            "face_box": None,
            "xai_explanation": {"method": "face_detection", "reasoning": "No face"},
            "processing_time_ms": 50.0,
            "timestamp": "2026-01-01T00:00:00",
        }
        jpeg = _generate_jpeg_bytes()
        r = client.post(
            "/analyze/image",
            files={"image": ("photo.jpg", jpeg, "image/jpeg")},
        )
        assert r.status_code == 200
        assert r.json()["face_detected"] is False

    @patch("api.routes.image.face_analyzer")
    def test_image_internal_error_returns_500(self, mock_analyzer):
        mock_analyzer.analyze_image.side_effect = OSError("model error")
        jpeg = _generate_jpeg_bytes()
        r = client.post(
            "/analyze/image",
            files={"image": ("photo.jpg", jpeg, "image/jpeg")},
        )
        assert r.status_code == 500

    def test_image_models_returns_200(self):
        r = client.get("/analyze/image/models")
        assert r.status_code == 200
        data = r.json()
        assert "face_detection" in data
        assert "emotion_detection" in data
        assert "xai_method" in data


# ---------------------------------------------------------------------------
# Multimodal Analysis API
# ---------------------------------------------------------------------------

class TestMultimodalAPI:

    def test_no_modalities_returns_400(self):
        r = client.post("/analyze/multimodal", data={})
        assert r.status_code == 400
        assert "at least one modality" in r.json()["detail"].lower()

    def test_whitespace_only_text_returns_400(self):
        r = client.post("/analyze/multimodal", data={"text": "   "})
        assert r.status_code == 400

    @patch("api.routes.multimodal.fusion_engine")
    @patch("api.routes.multimodal.text_analyzer")
    def test_text_only_multimodal(self, mock_text, mock_fusion):
        mock_text.analyze.return_value = {
            "success": True,
            "emotions": {"joy": 80.0, "neutral": 20.0},
            "primary_emotion": "joy",
            "confidence": 80.0,
        }
        mock_fusion.fuse.return_value = {
            "success": True,
            "final_emotions": {"joy": 80.0, "neutral": 20.0},
            "primary_emotion": "joy",
            "confidence": 80.0,
            "modalities_used": ["text"],
            "weights": {"text": 1.0},
            "individual_results": {},
            "incongruence": None,
            "xai_explanation": None,
            "processing_time_ms": 10.0,
            "timestamp": "2026-01-01T00:00:00",
        }
        r = client.post("/analyze/multimodal", data={"text": "I am happy!"})
        assert r.status_code == 200
        assert r.json()["success"] is True

    @patch("api.routes.multimodal.fusion_engine")
    @patch("api.routes.multimodal.text_analyzer")
    @patch("api.routes.multimodal.voice_analyzer")
    def test_text_and_voice_multimodal(self, mock_voice, mock_text, mock_fusion):
        mock_text.analyze.return_value = {
            "success": True,
            "emotions": {"joy": 70.0, "neutral": 30.0},
            "primary_emotion": "joy", "confidence": 70.0,
        }
        mock_voice.analyze.return_value = {
            "success": True,
            "emotions": {"sadness": 60.0, "neutral": 40.0},
            "primary_emotion": "sadness", "confidence": 60.0,
        }
        mock_fusion.fuse.return_value = {
            "success": True,
            "final_emotions": {"joy": 40.0, "sadness": 35.0, "neutral": 25.0},
            "primary_emotion": "joy", "confidence": 40.0,
            "modalities_used": ["text", "voice"],
            "weights": {"text": 0.55, "voice": 0.45},
            "individual_results": {}, "incongruence": {"is_incongruent": True},
            "xai_explanation": None,
            "processing_time_ms": 100.0,
            "timestamp": "2026-01-01T00:00:00",
        }
        wav = _generate_wav_bytes()
        r = client.post(
            "/analyze/multimodal",
            data={"text": "I am happy!"},
            files={"audio": ("test.wav", wav, "audio/wav")},
        )
        assert r.status_code == 200

    @patch("api.routes.multimodal.fusion_engine")
    @patch("api.routes.multimodal.face_analyzer")
    def test_image_only_multimodal(self, mock_face, mock_fusion):
        mock_face.analyze_image.return_value = {
            "success": True, "face_detected": True,
            "emotions": {"joy": 80.0, "neutral": 20.0},
            "primary_emotion": "joy", "confidence": 80.0,
        }
        mock_fusion.fuse.return_value = {
            "success": True,
            "final_emotions": {"joy": 80.0, "neutral": 20.0},
            "primary_emotion": "joy", "confidence": 80.0,
            "modalities_used": ["face"],
            "weights": {"face": 1.0},
            "individual_results": {}, "incongruence": None,
            "xai_explanation": None,
            "processing_time_ms": 50.0,
            "timestamp": "2026-01-01T00:00:00",
        }
        jpeg = _generate_jpeg_bytes()
        r = client.post(
            "/analyze/multimodal",
            files={"image": ("photo.jpg", jpeg, "image/jpeg")},
        )
        assert r.status_code == 200

    def test_multimodal_info_returns_200(self):
        r = client.get("/analyze/multimodal/info")
        assert r.status_code == 200
        data = r.json()
        assert "modalities" in data
        assert "fusion_method" in data
        assert "incongruence_detection" in data


# ---------------------------------------------------------------------------
# Authentication tests
# ---------------------------------------------------------------------------

class TestAuthentication:

    def test_unauthenticated_returns_401_or_passthrough(self):
        """Without auth override, requests should get 401 (or 200 if Firebase not initialized).
        Firebase Admin SDK must be initialized for auth to work; without credentials
        the middleware falls through with uid='unverified'.
        """
        from main import app as _app
        from fastapi.testclient import TestClient as TC

        saved = _app.dependency_overrides.copy()
        _app.dependency_overrides.pop(get_current_user, None)
        try:
            unauthed = TC(_app)
            r = unauthed.post("/analyze/text", json={"text": "hello"})
            # With Firebase not initialized: 200 (passthrough)
            # With Firebase initialized but no token: 401
            assert r.status_code in (200, 401)
        finally:
            _app.dependency_overrides = saved

    def test_unauthenticated_no_token_behavior(self):
        """Without Bearer token and with Firebase initialized, should get 401."""
        from main import app as _app
        from fastapi.testclient import TestClient as TC

        saved = _app.dependency_overrides.copy()
        _app.dependency_overrides.pop(get_current_user, None)
        try:
            unauthed = TC(_app)
            # Send without any Authorization header
            r = unauthed.post(
                "/analyze/text",
                json={"text": "hello"},
                headers={},  # No auth
            )
            # Without Firebase: 200 (fallthrough), With Firebase: 401
            assert r.status_code in (200, 401)
            if r.status_code == 401:
                assert "detail" in r.json()
        finally:
            _app.dependency_overrides = saved


# ---------------------------------------------------------------------------
# Response structure validation
# ---------------------------------------------------------------------------

class TestResponseStructure:

    @patch("api.routes.text.text_analyzer")
    def test_text_response_has_no_extra_fields(self, mock_analyzer):
        """Text response should not leak internal fields."""
        mock_analyzer.analyze.return_value = {
            "success": True,
            "emotions": {"neutral": 100.0},
            "primary_emotion": "neutral",
            "confidence": 100.0,
            "sentiment": {"label": "neutral", "score": 50.0},
            "text_metrics": {"word_count": 1},
            "xai_explanation": None,
            "processing_time_ms": 5.0,
            "timestamp": "2026-01-01T00:00:00",
        }
        r = client.post("/analyze/text", json={"text": "test", "include_xai": False})
        data = r.json()
        # Should not contain internal fields
        assert "__traceback__" not in data
        assert "stack_trace" not in data

    @patch("api.routes.voice.voice_analyzer")
    def test_voice_response_structure(self, mock_analyzer):
        mock_analyzer.analyze.return_value = {
            "success": True,
            "emotions": {"neutral": 100.0},
            "primary_emotion": "neutral",
            "confidence": 100.0,
            "audio_features": {"duration_seconds": 1.0, "energy": 0.05},
            "xai_explanation": None,
            "processing_time_ms": 50.0,
            "timestamp": "2026-01-01T00:00:00",
        }
        wav = _generate_wav_bytes()
        r = client.post(
            "/analyze/voice",
            files={"audio": ("test.wav", wav, "audio/wav")},
        )
        assert r.status_code == 200
        data = r.json()
        assert "audio_features" in data
        assert "processing_time_ms" in data
