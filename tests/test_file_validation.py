"""
Unit tests for file_validation module.
Tests magic byte validation for audio and image formats.
Run with: pytest tests/test_file_validation.py -v
"""

import pytest
from api.file_validation import validate_audio_bytes, validate_image_bytes


# ---------------------------------------------------------------------------
# Audio validation tests
# ---------------------------------------------------------------------------

class TestAudioValidation:

    def test_valid_wav(self):
        """RIFF header = valid WAV."""
        data = b'RIFF' + b'\x00' * 100
        assert validate_audio_bytes(data) is True

    def test_valid_mp3_sync_word_fb(self):
        data = b'\xff\xfb' + b'\x00' * 100
        assert validate_audio_bytes(data) is True

    def test_valid_mp3_sync_word_f3(self):
        data = b'\xff\xf3' + b'\x00' * 100
        assert validate_audio_bytes(data) is True

    def test_valid_mp3_sync_word_f2(self):
        data = b'\xff\xf2' + b'\x00' * 100
        assert validate_audio_bytes(data) is True

    def test_valid_mp3_with_id3(self):
        data = b'ID3' + b'\x00' * 100
        assert validate_audio_bytes(data) is True

    def test_valid_ogg(self):
        data = b'OggS' + b'\x00' * 100
        assert validate_audio_bytes(data) is True

    def test_valid_webm(self):
        data = b'\x1aE\xdf\xa3' + b'\x00' * 100
        assert validate_audio_bytes(data) is True

    def test_valid_m4a(self):
        """ftyp at offset 4 = M4A."""
        data = b'\x00\x00\x00\x00ftyp' + b'\x00' * 100
        assert validate_audio_bytes(data) is True

    def test_invalid_random_bytes(self):
        data = b'\x01\x02\x03\x04' + b'\x00' * 100
        assert validate_audio_bytes(data) is False

    def test_invalid_text_file(self):
        data = b'Hello World this is text' + b'\x00' * 100
        assert validate_audio_bytes(data) is False

    def test_too_short_data(self):
        """Data shorter than 12 bytes should fail."""
        assert validate_audio_bytes(b'RIFF') is False
        assert validate_audio_bytes(b'') is False
        assert validate_audio_bytes(b'\xff') is False

    def test_image_bytes_rejected(self):
        """Image magic bytes should not validate as audio."""
        jpeg = b'\xff\xd8\xff\xe0' + b'\x00' * 100
        assert validate_audio_bytes(jpeg) is False

        png = b'\x89PNG' + b'\x00' * 100
        assert validate_audio_bytes(png) is False


# ---------------------------------------------------------------------------
# Image validation tests
# ---------------------------------------------------------------------------

class TestImageValidation:

    def test_valid_jpeg(self):
        data = b'\xff\xd8\xff\xe0' + b'\x00' * 100
        assert validate_image_bytes(data) is True

    def test_valid_jpeg_exif(self):
        """JPEG with EXIF marker."""
        data = b'\xff\xd8\xff\xe1' + b'\x00' * 100
        assert validate_image_bytes(data) is True

    def test_valid_png(self):
        data = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
        assert validate_image_bytes(data) is True

    def test_valid_webp(self):
        """RIFF + WEBP at offset 8."""
        data = b'RIFF\x00\x00\x00\x00WEBP' + b'\x00' * 100
        assert validate_image_bytes(data) is True

    def test_invalid_random_bytes(self):
        data = b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c'
        assert validate_image_bytes(data) is False

    def test_invalid_text_file(self):
        data = b'This is not an image file' + b'\x00' * 100
        assert validate_image_bytes(data) is False

    def test_too_short_data(self):
        assert validate_image_bytes(b'\xff\xd8') is False
        assert validate_image_bytes(b'') is False

    def test_audio_bytes_rejected(self):
        """Audio magic bytes should not validate as image."""
        wav = b'RIFF\x00\x00\x00\x00WAVE' + b'\x00' * 100
        # WAV starts with RIFF but offset 8 is WAVE not WEBP
        assert validate_image_bytes(wav) is False

        mp3 = b'\xff\xfb\x00\x00' + b'\x00' * 100
        assert validate_image_bytes(mp3) is False

    def test_riff_without_webp_rejected(self):
        """RIFF container that isn't WebP should fail."""
        data = b'RIFF\x00\x00\x00\x00AVI ' + b'\x00' * 100
        assert validate_image_bytes(data) is False


# ---------------------------------------------------------------------------
# Cross-validation tests
# ---------------------------------------------------------------------------

class TestCrossValidation:
    """Ensure audio and image validators don't overlap incorrectly."""

    def test_wav_is_audio_not_image(self):
        """WAV should pass audio validation but fail image validation."""
        wav = b'RIFF\x00\x00\x00\x00WAVE' + b'\x00' * 100
        assert validate_audio_bytes(wav) is True
        assert validate_image_bytes(wav) is False

    def test_jpeg_is_image_not_audio(self):
        jpeg = b'\xff\xd8\xff\xe0' + b'\x00' * 100
        assert validate_image_bytes(jpeg) is True
        assert validate_audio_bytes(jpeg) is False

    def test_png_is_image_not_audio(self):
        png = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
        assert validate_image_bytes(png) is True
        assert validate_audio_bytes(png) is False

    def test_webp_is_image_not_just_audio(self):
        """WebP uses RIFF container, should be image but also passes audio RIFF check."""
        webp = b'RIFF\x00\x00\x00\x00WEBP' + b'\x00' * 100
        assert validate_image_bytes(webp) is True
        # Note: RIFF prefix will match audio too - this is a known ambiguity
        assert validate_audio_bytes(webp) is True  # Both match RIFF


# ---------------------------------------------------------------------------
# Boundary & security tests
# ---------------------------------------------------------------------------

class TestBoundaryAndSecurity:

    def test_exactly_12_bytes_audio(self):
        """Minimum valid length for audio validation."""
        data = b'RIFF' + b'\x00' * 8
        assert validate_audio_bytes(data) is True

    def test_exactly_12_bytes_image(self):
        data = b'\xff\xd8\xff' + b'\x00' * 9
        assert validate_image_bytes(data) is True

    def test_11_bytes_too_short(self):
        """11 bytes should fail both validators."""
        data = b'RIFF' + b'\x00' * 7
        assert validate_audio_bytes(data) is False
        assert validate_image_bytes(data) is False

    def test_polyglot_prevention(self):
        """Ensure a file can't masquerade as both audio and image (except RIFF edge case)."""
        # A file starting with MP3 magic followed by JPEG magic
        data = b'\xff\xfb\xff\xd8\xff\xe0' + b'\x00' * 100
        # Should be audio (MP3) but not image
        assert validate_audio_bytes(data) is True
        assert validate_image_bytes(data) is False

    def test_null_bytes_rejected(self):
        data = b'\x00' * 100
        assert validate_audio_bytes(data) is False
        assert validate_image_bytes(data) is False
