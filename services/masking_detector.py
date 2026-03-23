"""
Masking Detector Service
Detects fake smiles and masked emotions using 3 layers:
1. Emotion distribution analysis (conflicting emotion pairs)
2. Temporal analysis (sudden changes, oscillations)
3. MediaPipe facial landmarks (Duchenne smile via AU6/AU12)
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple


class MaskingDetector:
    """
    Detects fake smiles and masked emotions through three layers of analysis.
    Duchenne smile = AU6 (cheek raiser) + AU12 (lip corner puller) -> genuine
    Fake smile = only AU12 (mouth smiles but eyes don't) -> non-genuine
    """

    # Conflicting emotion pairs: surface → possible underlying
    CONFLICT_PAIRS = {
        ("joy", "sadness"): "fake_smile",
        ("joy", "anger"): "suppressed_anger",
        ("joy", "fear"): "nervous_smile",
        ("neutral", "sadness"): "suppressed_sadness",
        ("neutral", "anger"): "suppressed_anger",
    }

    # Thresholds
    PRIMARY_MIN = 30.0       # Primary emotion must be at least 30%
    SECONDARY_MIN = 15.0     # Secondary conflicting emotion must be at least 15%
    AU6_THRESHOLD = 0.3      # Minimum cheek raise for genuine smile
    CONFIDENCE_MIN = 0.4     # Minimum overall confidence to report
    TEMPORAL_JUMP = 30.0     # Minimum jump in emotion % to flag as sudden change
    OSCILLATION_WINDOW = 6   # Number of frames to check for oscillation

    def __init__(self):
        self._mp_face_mesh = None
        self._mp_initialized = False

    def _initialize_mediapipe(self):
        """Lazy initialization of MediaPipe FaceMesh"""
        if self._mp_initialized:
            return

        try:
            import mediapipe as mp
            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
            self._mp_initialized = True
        except ImportError:
            print("MediaPipe not available - landmark analysis disabled")
            self._mp_initialized = True  # Don't retry
        except Exception as e:
            print(f"MediaPipe initialization error: {e}")
            self._mp_initialized = True

    def analyze_frame(
        self,
        emotions: Dict[str, float],
        image_rgb: Optional[np.ndarray] = None,
        emotion_history: Optional[List[Dict[str, float]]] = None,
    ) -> Optional[Dict]:
        """
        Main entry point: analyzes a frame for possible emotional masking.

        Args:
            emotions: Current emotion scores (0-100)
            image_rgb: RGB image for landmark analysis (optional)
            emotion_history: List of previous emotion dicts for temporal analysis

        Returns:
            Dict with masking result or None if no masking detected
        """
        signals = []

        # Layer 1: Distribution analysis
        dist_result = self._analyze_distribution(emotions)
        if dist_result:
            signals.append(dist_result)

        # Layer 2: Temporal analysis
        if emotion_history and len(emotion_history) >= 2:
            temp_result = self._analyze_temporal(emotions, emotion_history)
            if temp_result:
                signals.append(temp_result)

        # Layer 3: Landmark analysis (Duchenne smile)
        if image_rgb is not None and emotions.get("joy", 0) > 20:
            landmark_result = self._analyze_landmarks(image_rgb)
            if landmark_result:
                signals.append(landmark_result)

        # Combine signals
        if signals:
            return self._combine_signals(signals, emotions)

        return None

    def _analyze_distribution(self, emotions: Dict[str, float]) -> Optional[Dict]:
        """
        Layer 1: Detects conflicting emotion pairs in current frame.
        e.g., joy > 30% AND sadness > 15% → possible masking
        """
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_emotions) < 2:
            return None

        primary_name, primary_score = sorted_emotions[0]
        if primary_score < self.PRIMARY_MIN:
            return None

        for secondary_name, secondary_score in sorted_emotions[1:]:
            if secondary_score < self.SECONDARY_MIN:
                break

            pair = (primary_name, secondary_name)
            if pair in self.CONFLICT_PAIRS:
                masking_type = self.CONFLICT_PAIRS[pair]
                # Confidence based on how strong both signals are
                confidence = min(primary_score, secondary_score * 2) / 100.0
                confidence = min(confidence, 0.95)

                return {
                    "layer": "distribution",
                    "type": masking_type,
                    "surface_emotion": primary_name,
                    "underlying_emotion": secondary_name,
                    "confidence": round(confidence, 2),
                    "detail": f"{primary_name} at {primary_score:.0f}% with "
                              f"conflicting {secondary_name} at {secondary_score:.0f}%",
                }

        # Check for mixed signals: small margin between top emotions
        margin = sorted_emotions[0][1] - sorted_emotions[1][1]
        if margin < 10 and sorted_emotions[0][1] > 20:
            return {
                "layer": "distribution",
                "type": "mixed_signals",
                "surface_emotion": sorted_emotions[0][0],
                "underlying_emotion": sorted_emotions[1][0],
                "confidence": round(0.3 * (1 - margin / 10), 2),
                "detail": f"Small margin ({margin:.0f}%) between "
                          f"{sorted_emotions[0][0]} and {sorted_emotions[1][0]}",
            }

        return None

    def _analyze_temporal(
        self,
        current: Dict[str, float],
        history: List[Dict[str, float]],
    ) -> Optional[Dict]:
        """
        Layer 2: Detects sudden emotion changes and oscillations.
        - Sudden jump from sadness to joy = suspicious
        - Oscillation between emotions = instability
        """
        if not history:
            return None

        prev = history[-1]

        # Check for sudden jumps between conflicting emotions
        for (surface, underlying), masking_type in self.CONFLICT_PAIRS.items():
            prev_underlying = prev.get(underlying, 0)
            curr_surface = current.get(surface, 0)
            prev_surface = prev.get(surface, 0)

            # Sudden shift: underlying was high, now surface jumped up
            if (prev_underlying > 25 and
                curr_surface > self.PRIMARY_MIN and
                curr_surface - prev_surface > self.TEMPORAL_JUMP):
                confidence = min(
                    (curr_surface - prev_surface) / 100.0 * 1.5,
                    0.85
                )
                return {
                    "layer": "temporal",
                    "type": f"sudden_{masking_type}",
                    "surface_emotion": surface,
                    "underlying_emotion": underlying,
                    "confidence": round(confidence, 2),
                    "detail": f"Sudden shift: {underlying} ({prev_underlying:.0f}%) "
                              f"→ {surface} ({curr_surface:.0f}%)",
                }

        # Check for oscillation in recent history
        if len(history) >= self.OSCILLATION_WINDOW:
            recent = history[-self.OSCILLATION_WINDOW:]
            primary_changes = 0
            primaries = []
            for h in recent:
                if h:
                    p = max(h, key=h.get) if h else "neutral"
                    primaries.append(p)

            for i in range(1, len(primaries)):
                if primaries[i] != primaries[i - 1]:
                    primary_changes += 1

            # If emotion flips more than 3 times in the window → unstable
            if primary_changes >= 3:
                return {
                    "layer": "temporal",
                    "type": "emotional_instability",
                    "surface_emotion": primaries[-1] if primaries else "unknown",
                    "underlying_emotion": "mixed",
                    "confidence": round(min(primary_changes / 6, 0.75), 2),
                    "detail": f"{primary_changes} emotion changes in last "
                              f"{self.OSCILLATION_WINDOW} frames",
                }

        return None

    def _analyze_landmarks(self, image_rgb: np.ndarray) -> Optional[Dict]:
        """
        Layer 3: Uses MediaPipe facial landmarks to detect Duchenne vs fake smile.

        Duchenne smile: AU6 (cheek raiser) + AU12 (lip corner puller)
        Fake smile: Only AU12 (mouth smiles but eyes don't)

        AU6 approximation: Distance between lower eyelid and cheek landmarks
        AU12 approximation: Lip corner elevation relative to lip center
        """
        self._initialize_mediapipe()

        if self._mp_face_mesh is None:
            return None

        try:
            results = self._mp_face_mesh.process(image_rgb)

            if not results.multi_face_landmarks:
                return None

            landmarks = results.multi_face_landmarks[0].landmark
            h, w = image_rgb.shape[:2]

            # Calculate AU6 (cheek raiser) - eye-cheek distance
            au6_score = self._calculate_au6(landmarks, h, w)

            # Calculate AU12 (lip corner puller) - mouth corner elevation
            au12_score = self._calculate_au12(landmarks, h, w)

            # Determine smile type
            if au12_score > 0.3:  # Mouth is smiling
                if au6_score < self.AU6_THRESHOLD:
                    # Fake smile: mouth smiles but cheeks/eyes don't engage
                    confidence = min((0.3 + au12_score - au6_score) * 0.8, 0.9)
                    return {
                        "layer": "landmarks",
                        "type": "fake_smile",
                        "surface_emotion": "joy",
                        "underlying_emotion": "unknown",
                        "confidence": round(confidence, 2),
                        "detail": f"Non-Duchenne smile detected: "
                                  f"AU6={au6_score:.2f} (low), AU12={au12_score:.2f} (high)",
                        "au6_score": round(au6_score, 3),
                        "au12_score": round(au12_score, 3),
                        "is_duchenne": False,
                    }

            return None

        except Exception as e:
            print(f"Landmark analysis error: {e}")
            return None

    def _calculate_au6(self, landmarks, h: int, w: int) -> float:
        """
        Approximate AU6 (Cheek Raiser) using MediaPipe landmarks.
        Measures the elevation of the cheek area relative to the lower eyelid.
        Higher value = more cheek engagement (genuine smile).

        Key landmarks:
        - Lower eyelid: 111 (left), 340 (right)
        - Cheek: 117 (left), 346 (right)
        - Eye outer corner: 33 (left), 263 (right)
        """
        # Left eye lower lid to cheek distance
        left_lower_lid = landmarks[111]
        left_cheek = landmarks[117]
        left_eye_corner = landmarks[33]

        # Right eye
        right_lower_lid = landmarks[340]
        right_cheek = landmarks[346]
        right_eye_corner = landmarks[263]

        # Normalize by inter-eye distance
        inter_eye_dist = self._landmark_distance(
            landmarks[33], landmarks[263], h, w
        )
        if inter_eye_dist < 1:
            return 0.0

        # Cheek rise: how much the cheek pushes up toward the eye
        left_cheek_rise = (left_cheek.y - left_lower_lid.y) * h / inter_eye_dist
        right_cheek_rise = (right_cheek.y - right_lower_lid.y) * h / inter_eye_dist

        # Lower values mean cheek is closer to eye (more AU6 activation)
        # Invert and normalize to 0-1 range
        au6_left = max(0, 1.0 - left_cheek_rise * 2.5)
        au6_right = max(0, 1.0 - right_cheek_rise * 2.5)

        return (au6_left + au6_right) / 2.0

    def _calculate_au12(self, landmarks, h: int, w: int) -> float:
        """
        Approximate AU12 (Lip Corner Puller) using MediaPipe landmarks.
        Measures how much the lip corners are pulled up relative to lip center.

        Key landmarks:
        - Left mouth corner: 61
        - Right mouth corner: 291
        - Upper lip center: 13
        - Lower lip center: 14
        - Nose tip: 1 (reference point)
        """
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        upper_lip = landmarks[13]
        nose_tip = landmarks[1]

        # Normalize by nose-to-lip distance
        nose_lip_dist = abs(nose_tip.y - upper_lip.y) * h
        if nose_lip_dist < 1:
            return 0.0

        # Lip corners relative to lip center
        lip_center_y = upper_lip.y
        left_elevation = (lip_center_y - left_corner.y) * h / nose_lip_dist
        right_elevation = (lip_center_y - right_corner.y) * h / nose_lip_dist

        # Positive = corners above center (smile)
        au12 = (left_elevation + right_elevation) / 2.0

        # Normalize to 0-1
        return max(0, min(au12 * 1.5, 1.0))

    def _landmark_distance(self, lm1, lm2, h: int, w: int) -> float:
        """Euclidean distance between two landmarks in pixel space"""
        dx = (lm1.x - lm2.x) * w
        dy = (lm1.y - lm2.y) * h
        return (dx * dx + dy * dy) ** 0.5

    def _combine_signals(
        self,
        signals: List[Dict],
        emotions: Dict[str, float],
    ) -> Optional[Dict]:
        """
        Combines signals from all layers into a final masking result.
        Requires minimum confidence and provides XAI explanation.
        """
        if not signals:
            return None

        # Sort by confidence
        signals.sort(key=lambda s: s["confidence"], reverse=True)
        best = signals[0]

        # Boost confidence if multiple layers agree
        num_layers = len(set(s["layer"] for s in signals))
        boosted_confidence = best["confidence"]
        if num_layers >= 2:
            boosted_confidence = min(boosted_confidence * 1.3, 0.95)
        if num_layers >= 3:
            boosted_confidence = min(boosted_confidence * 1.2, 0.98)

        # Check minimum confidence
        if boosted_confidence < self.CONFIDENCE_MIN:
            return None

        # Determine masking type (prefer non-mixed types)
        masking_type = best["type"]
        for s in signals:
            if s["type"] != "mixed_signals" and s["type"] != "emotional_instability":
                masking_type = s["type"]
                break

        # Build explanation
        layer_details = [s["detail"] for s in signals]
        layers_used = list(set(s["layer"] for s in signals))

        explanation = self._build_explanation(
            masking_type, best, signals, emotions
        )

        return {
            "detected": True,
            "type": masking_type,
            "confidence": round(boosted_confidence, 2),
            "surface_emotion": best["surface_emotion"],
            "underlying_emotion": best.get("underlying_emotion", "unknown"),
            "layers_triggered": layers_used,
            "num_signals": len(signals),
            "signals": [
                {
                    "layer": s["layer"],
                    "type": s["type"],
                    "confidence": s["confidence"],
                    "detail": s["detail"],
                }
                for s in signals
            ],
            "explanation": explanation,
            # Include AU scores if available
            "au6_score": next(
                (s.get("au6_score") for s in signals if "au6_score" in s), None
            ),
            "au12_score": next(
                (s.get("au12_score") for s in signals if "au12_score" in s), None
            ),
            "is_duchenne": next(
                (s.get("is_duchenne") for s in signals if "is_duchenne" in s), None
            ),
        }

    def _build_explanation(
        self,
        masking_type: str,
        best_signal: Dict,
        all_signals: List[Dict],
        emotions: Dict[str, float],
    ) -> Dict:
        """Builds XAI explanation for the masking detection"""
        type_explanations = {
            "fake_smile": (
                "Possible fake smile detected. Analysis shows signs of a smile "
                "that does not include typical muscle activations of genuine happiness "
                "(Duchenne smile)."
            ),
            "suppressed_anger": (
                "Possible suppressed anger. The surface expression shows calmness, "
                "but signs of anger are present in the background."
            ),
            "nervous_smile": (
                "Possible nervous smile. A smile is present but with signs "
                "of fear or anxiety."
            ),
            "suppressed_sadness": (
                "Possible suppressed sadness. The facial expression appears neutral but "
                "indicators of sadness are present."
            ),
            "mixed_signals": (
                "Mixed emotional signals. Multiple emotions are present with similar "
                "intensity, which may indicate a complex emotional state."
            ),
            "emotional_instability": (
                "Emotional instability detected. Frequent changes in the dominant "
                "emotion may indicate an internal conflict."
            ),
        }

        # Handle sudden_ prefixed types
        base_type = masking_type.replace("sudden_", "")
        reasoning = type_explanations.get(
            base_type,
            "Possible emotion masking detected based on analysis of facial "
            "expressions and behavior."
        )
        if masking_type.startswith("sudden_"):
            reasoning = "Sudden emotion change. " + reasoning

        layers_used = list(set(s["layer"] for s in all_signals))
        methods = []
        if "distribution" in layers_used:
            methods.append("emotion distribution analysis")
        if "temporal" in layers_used:
            methods.append("temporal analysis")
        if "landmarks" in layers_used:
            methods.append("facial landmark analysis (AU6/AU12)")

        return {
            "method": "masking_detection",
            "reasoning": reasoning,
            "methods_used": methods,
            "layers_triggered": len(layers_used),
            "note": "This is an indication, not a diagnosis. Results are based on "
                    "statistical analysis of facial expressions.",
        }


# Singleton instance
masking_detector = MaskingDetector()
