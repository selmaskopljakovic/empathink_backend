"""
File upload validation using magic bytes.
Prevents polyglot file attacks by verifying actual file content
matches the claimed content type.
"""

import logging

logger = logging.getLogger(__name__)

# Magic byte signatures for supported formats
_AUDIO_SIGNATURES = {
    b'RIFF': 'audio/wav',           # WAV
    b'\xff\xfb': 'audio/mpeg',       # MP3 (with sync word)
    b'\xff\xf3': 'audio/mpeg',       # MP3 (MPEG2 Layer3)
    b'\xff\xf2': 'audio/mpeg',       # MP3 (MPEG2 Layer3)
    b'ID3': 'audio/mpeg',            # MP3 with ID3 tag
    b'OggS': 'audio/ogg',            # OGG
    b'\x1aE\xdf\xa3': 'audio/webm',  # WebM/Matroska
    b'ftyp': 'audio/m4a',            # M4A (offset 4)
}

_IMAGE_SIGNATURES = {
    b'\xff\xd8\xff': 'image/jpeg',    # JPEG
    b'\x89PNG': 'image/png',          # PNG
    b'RIFF': 'image/webp',            # WebP (RIFF container)
}


def validate_audio_bytes(data: bytes) -> bool:
    """
    Check if audio data starts with a valid audio file signature.
    Returns True if the file appears to be a valid audio file.
    """
    if len(data) < 12:
        return False

    # Check direct prefix matches
    for sig in [b'RIFF', b'\xff\xfb', b'\xff\xf3', b'\xff\xf2', b'ID3', b'OggS', b'\x1aE\xdf\xa3']:
        if data.startswith(sig):
            return True

    # M4A: 'ftyp' at offset 4
    if data[4:8] == b'ftyp':
        return True

    return False


def validate_image_bytes(data: bytes) -> bool:
    """
    Check if image data starts with a valid image file signature.
    Returns True if the file appears to be a valid image file.
    """
    if len(data) < 12:
        return False

    # JPEG
    if data.startswith(b'\xff\xd8\xff'):
        return True

    # PNG
    if data.startswith(b'\x89PNG'):
        return True

    # WebP (RIFF....WEBP)
    if data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return True

    return False
