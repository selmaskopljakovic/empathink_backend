"""
Shared fixtures for EmpaThink backend tests.
"""

import pytest
import numpy as np
import io
import struct
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Auth override
# ---------------------------------------------------------------------------
FAKE_USER = {
    "uid": "test-user-123",
    "email": "test@example.com",
    "email_verified": True,
}


@pytest.fixture
def fake_user():
    return FAKE_USER.copy()


# ---------------------------------------------------------------------------
# Audio fixtures
# ---------------------------------------------------------------------------

def generate_wav_bytes(duration_s=1.0, sample_rate=16000, frequency=440.0):
    """Generate a valid WAV file as bytes with a sine wave tone."""
    num_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    audio = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    buf = io.BytesIO()
    # WAV header
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align

    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<I', 16))  # Subchunk1Size
    buf.write(struct.pack('<H', 1))   # AudioFormat (PCM)
    buf.write(struct.pack('<H', num_channels))
    buf.write(struct.pack('<I', sample_rate))
    buf.write(struct.pack('<I', byte_rate))
    buf.write(struct.pack('<H', block_align))
    buf.write(struct.pack('<H', bits_per_sample))
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    buf.write(audio.tobytes())

    return buf.getvalue()


def generate_stereo_wav_bytes(duration_s=1.0, sample_rate=16000):
    """Generate a valid stereo WAV file as bytes."""
    num_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    left = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    right = (np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)

    # Interleave channels
    stereo = np.empty(num_samples * 2, dtype=np.int16)
    stereo[0::2] = left
    stereo[1::2] = right

    buf = io.BytesIO()
    num_channels = 2
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align

    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<I', 16))
    buf.write(struct.pack('<H', 1))
    buf.write(struct.pack('<H', num_channels))
    buf.write(struct.pack('<I', sample_rate))
    buf.write(struct.pack('<I', byte_rate))
    buf.write(struct.pack('<H', block_align))
    buf.write(struct.pack('<H', bits_per_sample))
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    buf.write(stereo.tobytes())

    return buf.getvalue()


@pytest.fixture
def wav_bytes():
    """1-second mono WAV at 16kHz, 440Hz sine."""
    return generate_wav_bytes(duration_s=1.0, sample_rate=16000, frequency=440.0)


@pytest.fixture
def wav_bytes_long():
    """35-second mono WAV (exceeds 30s truncation limit)."""
    return generate_wav_bytes(duration_s=35.0, sample_rate=16000, frequency=300.0)


@pytest.fixture
def wav_bytes_stereo():
    """1-second stereo WAV."""
    return generate_stereo_wav_bytes(duration_s=1.0, sample_rate=16000)


@pytest.fixture
def wav_bytes_44100():
    """1-second mono WAV at 44100Hz (needs resampling to 16kHz)."""
    return generate_wav_bytes(duration_s=1.0, sample_rate=44100, frequency=440.0)


@pytest.fixture
def silent_wav_bytes():
    """1-second silent WAV (all zeros)."""
    sample_rate = 16000
    num_samples = sample_rate
    audio = np.zeros(num_samples, dtype=np.int16)

    buf = io.BytesIO()
    data_size = num_samples * 2
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<I', 16))
    buf.write(struct.pack('<H', 1))
    buf.write(struct.pack('<H', 1))
    buf.write(struct.pack('<I', sample_rate))
    buf.write(struct.pack('<I', sample_rate * 2))
    buf.write(struct.pack('<H', 2))
    buf.write(struct.pack('<H', 16))
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    buf.write(audio.tobytes())

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Image fixtures
# ---------------------------------------------------------------------------

def generate_minimal_jpeg():
    """Generate minimal valid JPEG bytes (1x1 pixel)."""
    # Minimal JPEG: SOI + APP0 + ... This is a real 1x1 white JPEG
    import cv2
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


def generate_minimal_png():
    """Generate minimal valid PNG bytes."""
    import cv2
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    _, buf = cv2.imencode('.png', img)
    return buf.tobytes()


@pytest.fixture
def jpeg_bytes():
    return generate_minimal_jpeg()


@pytest.fixture
def png_bytes():
    return generate_minimal_png()


@pytest.fixture
def face_image_bytes():
    """Generate a synthetic image with a simple 'face-like' pattern.
    Real face detection may or may not detect this - use mocking for reliable tests."""
    import cv2
    img = np.ones((300, 300, 3), dtype=np.uint8) * 200
    # Draw a simple face
    cv2.circle(img, (150, 150), 80, (180, 150, 130), -1)  # Face
    cv2.circle(img, (125, 130), 10, (50, 50, 50), -1)      # Left eye
    cv2.circle(img, (175, 130), 10, (50, 50, 50), -1)      # Right eye
    cv2.ellipse(img, (150, 175), (25, 10), 0, 0, 180, (50, 50, 50), 2)  # Mouth
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Emotion result fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_text_result():
    """A plausible text analysis result."""
    return {
        "success": True,
        "emotions": {
            "joy": 72.5, "surprise": 10.0, "neutral": 8.0,
            "sadness": 4.0, "anger": 2.5, "fear": 2.0, "disgust": 1.0,
        },
        "primary_emotion": "joy",
        "confidence": 72.5,
        "sentiment": {"label": "positive", "score": 92.0},
        "text_metrics": {"polarity": 0.5, "subjectivity": 0.6, "word_count": 7},
        "xai_explanation": None,
        "processing_time_ms": 42.0,
        "timestamp": "2026-01-01T00:00:00",
    }


@pytest.fixture
def sample_voice_result():
    """A plausible voice analysis result."""
    return {
        "success": True,
        "emotions": {
            "sadness": 45.0, "neutral": 25.0, "anger": 10.0,
            "fear": 8.0, "joy": 5.0, "surprise": 4.0, "disgust": 3.0,
        },
        "primary_emotion": "sadness",
        "confidence": 45.0,
        "audio_features": {
            "duration_seconds": 2.5,
            "energy": 0.05,
            "pitch_mean": 120.0,
            "pitch_std": 20.0,
            "tempo": 80.0,
        },
        "xai_explanation": None,
        "processing_time_ms": 150.0,
        "timestamp": "2026-01-01T00:00:00",
    }


@pytest.fixture
def sample_face_result():
    """A plausible face analysis result."""
    return {
        "success": True,
        "face_detected": True,
        "emotions": {
            "joy": 60.0, "neutral": 20.0, "surprise": 10.0,
            "sadness": 5.0, "anger": 2.0, "fear": 2.0, "disgust": 1.0,
        },
        "primary_emotion": "joy",
        "confidence": 60.0,
        "face_box": {"x": 50, "y": 50, "width": 200, "height": 200},
        "xai_explanation": None,
        "processing_time_ms": 200.0,
        "timestamp": "2026-01-01T00:00:00",
    }


@pytest.fixture
def sample_face_no_face_result():
    """Face result when no face detected."""
    return {
        "success": True,
        "face_detected": False,
        "emotions": {"neutral": 100.0},
        "primary_emotion": "neutral",
        "confidence": 0.0,
        "face_box": None,
    }
