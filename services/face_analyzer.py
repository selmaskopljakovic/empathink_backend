"""
Face Emotion Analyzer Service
Uses FER/DeepFace for facial expression recognition with MTCNN face detection.
Supports backend switching between FER (FER2013) and DeepFace (AffectNet).
"""

import time
import os
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
import io
import base64

from services.masking_detector import masking_detector

# Backend selection: "deepface" (better accuracy, AffectNet) or "fer" (lighter, FER2013)
FACE_BACKEND = os.environ.get("FACE_BACKEND", "deepface")


class FaceEmotionAnalyzer:
    """
    Analyzes facial images and detects emotions using:
    - DeepFace (default): AffectNet dataset, better accuracy
    - FER (fallback): FER2013 dataset, lighter model
    - MTCNN for face detection
    - OpenCV for image processing
    """

    # FER emotion labels
    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    # Normalization of FER labels to Ekman standard (same as text_analyzer and voice_analyzer)
    LABEL_NORMALIZATION = {
        "happy": "joy",
        "sad": "sadness",
        "angry": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "surprise": "surprise",
        "neutral": "neutral",
    }

    @staticmethod
    def _normalize_emotions(emotions: Dict[str, float]) -> Dict[str, float]:
        """Normalizes FER labels (happy/sad/angry) to standard (joy/sadness/anger)."""
        return {
            FaceEmotionAnalyzer.LABEL_NORMALIZATION.get(k, k): v
            for k, v in emotions.items()
        }

    def __init__(self):
        self._fer_detector = None
        self._deepface = None
        self._backend = FACE_BACKEND
        self._is_initialized = False

    def _initialize(self):
        """Lazy initialization of face emotion detector"""
        if self._is_initialized:
            return

        if self._backend == "deepface":
            try:
                from deepface import DeepFace
                self._deepface = DeepFace
                self._is_initialized = True
                print("DeepFace initialized (AffectNet backend)")
            except ImportError:
                print("DeepFace not available, falling back to FER")
                self._backend = "fer"
                self._initialize_fer()
        else:
            self._initialize_fer()

    def _initialize_fer(self):
        """Initialize FER detector as fallback"""
        try:
            from fer import FER
            self._fer_detector = FER(mtcnn=True)
            self._is_initialized = True
            self._backend = "fer"
            print("FER initialized (FER2013 backend)")
        except Exception as e:
            print(f"FER initialization error: {e}")
            self._is_initialized = False

    def analyze_image(self, image_data: bytes, include_xai: bool = True) -> Dict:
        """
        Analyzes an image and returns emotions with percentages.

        Args:
            image_data: Image as bytes
            include_xai: Whether to include XAI explanations

        Returns:
            Dict with emotions and face box coordinates
        """
        start_time = time.time()

        try:
            import cv2

            # Lazy initialize
            self._initialize()

            # Load image from bytes
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return self._error_result("Could not decode image")

            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect emotions — DeepFace or FER
            if self._backend == "deepface" and self._deepface is not None:
                result = self._analyze_with_deepface(img_rgb)
            else:
                if self._fer_detector is None:
                    from fer import FER
                    self._fer_detector = FER(mtcnn=True)
                result = self._fer_detector.detect_emotions(img_rgb)

            if not result:
                return {
                    "success": True,
                    "face_detected": False,
                    "emotions": {"neutral": 100.0},
                    "primary_emotion": "neutral",
                    "confidence": 0.0,
                    "face_box": None,
                    "xai_explanation": {
                        "method": "face_detection",
                        "reasoning": "No face detected in the image. Please try with better lighting."
                    },
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                    "timestamp": datetime.now().isoformat()
                }

            # Take the first detected face
            face = result[0]
            emotions = face["emotions"]
            box = face["box"]

            # Convert to percentages
            emotions_percent = {k: round(v * 100, 1) for k, v in emotions.items()}

            # Find primary emotion (FER labels for XAI)
            primary_emotion_fer = max(emotions, key=emotions.get)
            confidence = round(emotions[primary_emotion_fer] * 100, 1)

            # Normalize labels: happy->joy, sad->sadness, angry->anger
            emotions_normalized = self._normalize_emotions(emotions_percent)
            primary_emotion = self.LABEL_NORMALIZATION.get(primary_emotion_fer, primary_emotion_fer)

            # Face box
            face_box = {
                "x": int(box[0]),
                "y": int(box[1]),
                "width": int(box[2]),
                "height": int(box[3])
            }

            # XAI explanation (uses FER labels because FACS mapping uses originals)
            xai_explanation = None
            if include_xai:
                xai_explanation = self.generate_explanation(emotions_percent, primary_emotion_fer)

            # Masking detection
            masking_result = None
            try:
                masking_result = masking_detector.analyze_frame(
                    emotions=emotions_normalized,
                    image_rgb=img_rgb,
                )
            except Exception as e:
                print(f"Masking detection error: {e}")

            processing_time = (time.time() - start_time) * 1000

            result = {
                "success": True,
                "face_detected": True,
                "emotions": emotions_normalized,
                "primary_emotion": primary_emotion,
                "confidence": confidence,
                "face_box": face_box,
                "xai_explanation": xai_explanation,
                "processing_time_ms": round(processing_time, 2),
                "timestamp": datetime.now().isoformat()
            }

            if masking_result:
                result["masking"] = masking_result

            return result

        except Exception as e:
            print(f"Face analysis error: {e}")
            return self._error_result(str(e))

    def analyze_frame_fast(
        self,
        frame_base64: str,
        emotion_history: Optional[List[Dict[str, float]]] = None,
    ) -> Dict:
        """
        Fast analysis of a single frame for live camera.
        Does not use MTCNN for greater speed.

        Args:
            frame_base64: Base64 encoded frame
            emotion_history: List of previous emotions for temporal masking analysis

        Returns:
            Dict with emotions for real-time display
        """
        try:
            import cv2
            from fer import FER

            # Create fast detector (without MTCNN)
            fast_detector = FER(mtcnn=False)

            # Decode base64
            img_bytes = base64.b64decode(frame_base64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return {
                    "face_detected": False,
                    "emotions": {},
                    "primary_emotion": None,
                    "confidence": 0.0,
                    "timestamp": time.time()
                }

            # Reduce resolution for speed
            scale = 0.5
            small_frame = cv2.resize(frame, None, fx=scale, fy=scale)

            # Detect emotions
            result = fast_detector.detect_emotions(small_frame)

            if not result:
                return {
                    "face_detected": False,
                    "emotions": {},
                    "primary_emotion": None,
                    "confidence": 0.0,
                    "timestamp": time.time()
                }

            face = result[0]
            emotions_raw = {k: round(v * 100, 1) for k, v in face["emotions"].items()}

            # Normalize labels: happy->joy, sad->sadness, angry->anger
            emotions = self._normalize_emotions(emotions_raw)
            primary_fer = max(emotions_raw, key=emotions_raw.get)
            primary = self.LABEL_NORMALIZATION.get(primary_fer, primary_fer)

            # Scale box back
            box = face["box"]
            face_box = {
                "x": int(box[0] / scale),
                "y": int(box[1] / scale),
                "width": int(box[2] / scale),
                "height": int(box[3] / scale)
            }

            # Masking detection (uses original frame for landmarks)
            masking_result = None
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                masking_result = masking_detector.analyze_frame(
                    emotions=emotions,
                    image_rgb=frame_rgb,
                    emotion_history=emotion_history,
                )
            except Exception as e:
                print(f"Masking detection error (fast): {e}")

            # Generate XAI explanation
            xai_explanation = self.generate_explanation(emotions_raw, primary_fer)

            response = {
                "face_detected": True,
                "emotions": emotions,
                "primary_emotion": primary,
                "confidence": emotions[primary],
                "face_box": face_box,
                "xai_explanation": xai_explanation,
                "timestamp": time.time()
            }

            if masking_result:
                response["masking"] = masking_result

            return response

        except Exception as e:
            print(f"Fast frame analysis error: {e}")
            return {
                "face_detected": False,
                "emotions": {},
                "error": str(e),
                "timestamp": time.time()
            }

    def _analyze_with_deepface(self, img_rgb: np.ndarray) -> List[Dict]:
        """
        Analyze face using DeepFace (AffectNet backend).
        Returns result in same format as FER for compatibility.
        """
        try:
            results = self._deepface.analyze(
                img_path=img_rgb,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',  # Faster than mtcnn for DeepFace
                silent=True,
            )

            if not results:
                return []

            # DeepFace returns list of dicts or single dict
            if isinstance(results, dict):
                results = [results]

            converted = []
            for face in results:
                emotions = face.get("emotion", {})
                # DeepFace returns 0-100 scores, normalize to 0-1 for FER compat
                emotions_normalized = {k.lower(): v / 100.0 for k, v in emotions.items()}

                region = face.get("region", {})
                box = (
                    region.get("x", 0),
                    region.get("y", 0),
                    region.get("w", 0),
                    region.get("h", 0),
                )

                converted.append({
                    "emotions": emotions_normalized,
                    "box": box,
                })

            return converted

        except Exception as e:
            print(f"DeepFace analysis error: {e}, falling back to FER")
            # Fallback to FER
            if self._fer_detector is None:
                from fer import FER
                self._fer_detector = FER(mtcnn=True)
            return self._fer_detector.detect_emotions(img_rgb)

    def generate_explanation(self, emotions: Dict[str, float], primary_emotion: str) -> Dict:
        """Generates XAI explanation for face analysis"""

        # Facial Action Units (FACS) for each emotion
        facial_action_units = {
            "happy": [
                "AU6 (Cheek Raiser) - raised cheeks",
                "AU12 (Lip Corner Puller) - smile"
            ],
            "sad": [
                "AU1 (Inner Brow Raiser) - raised inner brow",
                "AU4 (Brow Lowerer) - lowered brows",
                "AU15 (Lip Corner Depressor) - lowered lip corners"
            ],
            "angry": [
                "AU4 (Brow Lowerer) - furrowed brows",
                "AU5 (Upper Lid Raiser) - wide open eyes",
                "AU7 (Lid Tightener) - tightened eyelids"
            ],
            "fear": [
                "AU1+2 (Brow Raiser) - raised eyebrows",
                "AU5 (Upper Lid Raiser) - wide open eyes",
                "AU20 (Lip Stretcher) - stretched lips"
            ],
            "surprise": [
                "AU1+2 (Brow Raiser) - raised eyebrows",
                "AU5 (Upper Lid Raiser) - wide open eyes",
                "AU26 (Jaw Drop) - open mouth"
            ],
            "disgust": [
                "AU9 (Nose Wrinkler) - wrinkled nose",
                "AU15 (Lip Corner Depressor) - lowered lips",
                "AU16 (Lower Lip Depressor) - lowered lower lip"
            ],
            "neutral": [
                "No significant facial muscle activations"
            ]
        }

        explanations = {
            "happy": "Raised cheeks and a smile indicate happiness.",
            "sad": "Lowered brows and lip corners are characteristic of sadness.",
            "angry": "Furrowed brows and a tense facial expression indicate anger.",
            "fear": "Raised eyebrows and wide open eyes suggest fear.",
            "surprise": "Raised eyebrows and an open mouth indicate surprise.",
            "disgust": "A wrinkled nose and lowered lips are characteristic of disgust.",
            "neutral": "The face is relaxed without pronounced emotions."
        }

        # Sort emotions for breakdown
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

        return {
            "method": "facial_action_coding_system",
            "reasoning": explanations.get(primary_emotion, "Facial expression analysis."),
            "facial_action_units": facial_action_units.get(primary_emotion, []),
            "confidence_breakdown": {e[0]: e[1] for e in sorted_emotions[:4]},
            "interpretation": f"The model analyzed facial muscles and detected "
                            f"'{primary_emotion}' as the dominant expression with "
                            f"{emotions[primary_emotion]}% confidence."
        }

    def _error_result(self, error_message: str) -> Dict:
        """Returns a standard error response"""
        return {
            "success": False,
            "face_detected": False,
            "error": error_message,
            "emotions": {},
            "primary_emotion": None,
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat()
        }


# Singleton instance
face_analyzer = FaceEmotionAnalyzer()
