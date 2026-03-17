"""
HeadGestureDetector - Detects head nod/shake gestures from video frame sequences.

Uses MediaPipe FaceMesh to track nose tip (landmark 1) movement across frames.
Nod = vertical oscillation, Shake = horizontal oscillation.
Applies detrending and zero-crossing analysis for robust detection.
"""

import base64
import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class HeadGestureDetector:
    """Detects head nod and shake gestures from a sequence of frames."""

    # Detection thresholds
    MIN_FRAMES = 8
    NOD_AMPLITUDE_THRESHOLD = 0.012  # Minimum Y oscillation amplitude
    SHAKE_AMPLITUDE_THRESHOLD = 0.015  # Minimum X oscillation amplitude
    MIN_ZERO_CROSSINGS = 2  # Minimum direction changes
    CONFIDENCE_BASE = 0.6

    def __init__(self):
        self._mp_face_mesh = None
        self._mp_initialized = False

    def _initialize_mediapipe(self):
        """Lazy initialization of MediaPipe FaceMesh."""
        if self._mp_initialized:
            return

        try:
            import mediapipe as mp

            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
            )
            self._mp_initialized = True
        except ImportError:
            logger.warning("MediaPipe not available - gesture detection disabled")
            self._mp_initialized = True
        except Exception as e:
            logger.error("MediaPipe initialization error: %s", e)
            self._mp_initialized = True

    def _decode_frame(self, frame_base64: str) -> Optional[np.ndarray]:
        """Decode base64 frame to numpy RGB array."""
        try:
            import cv2

            img_bytes = base64.b64decode(frame_base64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

    def _extract_nose_position(self, frame_rgb: np.ndarray) -> Optional[tuple]:
        """Extract nose tip (landmark 1) normalized position from frame."""
        if self._mp_face_mesh is None:
            return None

        try:
            results = self._mp_face_mesh.process(frame_rgb)
            if results.multi_face_landmarks:
                nose = results.multi_face_landmarks[0].landmark[1]
                return (nose.x, nose.y)
        except Exception:
            pass

        return None

    def _detrend(self, signal: np.ndarray) -> np.ndarray:
        """Remove linear trend from signal."""
        n = len(signal)
        if n < 2:
            return signal
        x = np.arange(n)
        coeffs = np.polyfit(x, signal, 1)
        trend = np.polyval(coeffs, x)
        return signal - trend

    def _count_zero_crossings(self, signal: np.ndarray) -> int:
        """Count number of zero crossings in signal."""
        if len(signal) < 2:
            return 0
        signs = np.sign(signal)
        signs[signs == 0] = 1  # Treat zero as positive
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        return int(crossings)

    def detect_gesture(self, frames_base64: List[str]) -> Dict:
        """
        Detect head gesture from a sequence of base64-encoded frames.

        Args:
            frames_base64: List of 10-15 base64-encoded image frames

        Returns:
            {"gesture": "nod"|"shake"|"none", "confidence": float}
        """
        self._initialize_mediapipe()

        if self._mp_face_mesh is None:
            return {"gesture": "none", "confidence": 0.0}

        if len(frames_base64) < self.MIN_FRAMES:
            return {"gesture": "none", "confidence": 0.0}

        # Extract nose positions from all frames
        positions = []
        for frame_b64 in frames_base64:
            frame = self._decode_frame(frame_b64)
            if frame is None:
                continue

            pos = self._extract_nose_position(frame)
            if pos is not None:
                positions.append(pos)

        if len(positions) < self.MIN_FRAMES:
            return {"gesture": "none", "confidence": 0.0}

        # Separate X and Y signals
        x_signal = np.array([p[0] for p in positions])
        y_signal = np.array([p[1] for p in positions])

        # Detrend to remove slow head drift
        x_detrended = self._detrend(x_signal)
        y_detrended = self._detrend(y_signal)

        # Calculate amplitude (peak-to-peak)
        x_amplitude = np.ptp(x_detrended)
        y_amplitude = np.ptp(y_detrended)

        # Count zero crossings (direction changes)
        x_crossings = self._count_zero_crossings(x_detrended)
        y_crossings = self._count_zero_crossings(y_detrended)

        # Detect nod (vertical oscillation)
        is_nod = (
            y_amplitude >= self.NOD_AMPLITUDE_THRESHOLD
            and y_crossings >= self.MIN_ZERO_CROSSINGS
            and y_amplitude > x_amplitude * 1.3  # Y must dominate
        )

        # Detect shake (horizontal oscillation)
        is_shake = (
            x_amplitude >= self.SHAKE_AMPLITUDE_THRESHOLD
            and x_crossings >= self.MIN_ZERO_CROSSINGS
            and x_amplitude > y_amplitude * 1.3  # X must dominate
        )

        if is_nod and not is_shake:
            confidence = min(
                1.0,
                self.CONFIDENCE_BASE
                + (y_crossings - self.MIN_ZERO_CROSSINGS) * 0.1
                + (y_amplitude - self.NOD_AMPLITUDE_THRESHOLD) * 10,
            )
            return {"gesture": "nod", "confidence": round(confidence, 2)}

        elif is_shake and not is_nod:
            confidence = min(
                1.0,
                self.CONFIDENCE_BASE
                + (x_crossings - self.MIN_ZERO_CROSSINGS) * 0.1
                + (x_amplitude - self.SHAKE_AMPLITUDE_THRESHOLD) * 10,
            )
            return {"gesture": "shake", "confidence": round(confidence, 2)}

        return {"gesture": "none", "confidence": 0.0}


# Singleton instance
head_gesture_detector = HeadGestureDetector()
