"""
Face Emotion Analyzer Service
Primary: HSEmotion (AffectNet, state-of-the-art, 1st place ABAW 2025)
Fallback: FER (FER2013, lighter)
"""

import time
import os
import logging
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
import io
import base64

from services.masking_detector import masking_detector

logger = logging.getLogger(__name__)

# Lazy-loaded HSEmotion model
_hsemotion_model = None


def get_hsemotion_model():
    """Lazy load HSEmotion (EfficientNet-B0 trained on AffectNet)."""
    global _hsemotion_model
    if _hsemotion_model is not None:
        return _hsemotion_model

    try:
        from hsemotion.facial_emotions import HSEmotionRecognizer
        _hsemotion_model = HSEmotionRecognizer(
            model_name='enet_b2_8',
            device='cpu'
        )
        logger.info("HSEmotion model loaded (enet_b2_8, AffectNet, ~66.3% accuracy)")
        return _hsemotion_model
    except Exception as e:
        logger.error("Failed to load HSEmotion: %s", e)
        return None


class FaceEmotionAnalyzer:
    """
    Analyzes face images and detects emotions using:
    - HSEmotion (primary): AffectNet dataset, state-of-the-art accuracy, 8 emotions
    - FER (fallback): FER2013 dataset, lighter model
    - OpenCV for image processing
    """

    # HSEmotion labels (8 emotions from AffectNet)
    HSEMOTION_LABELS = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

    # HSEmotion → Ekman 7 mapping
    LABEL_NORMALIZATION = {
        "happiness": "joy",
        "sadness": "sadness",
        "anger": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "surprise": "surprise",
        "neutral": "neutral",
        "contempt": "anger",  # Map contempt to anger family
        # FER compat labels
        "happy": "joy",
        "sad": "sadness",
        "angry": "anger",
    }

    @staticmethod
    def _normalize_emotions(emotions: Dict[str, float]) -> Dict[str, float]:
        """Normalizes emotion labels to Ekman standard (joy/sadness/anger/...)."""
        ekman = {
            "joy": 0.0, "sadness": 0.0, "anger": 0.0,
            "disgust": 0.0, "fear": 0.0, "surprise": 0.0, "neutral": 0.0,
        }
        for k, v in emotions.items():
            mapped = FaceEmotionAnalyzer.LABEL_NORMALIZATION.get(k, k)
            if mapped in ekman:
                ekman[mapped] += v
        return ekman

    def __init__(self):
        self._fer_detector = None
        self._fast_detector = None
        self._hsemotion = None
        self._backend = "hsemotion"
        self._is_initialized = False

    def _initialize(self):
        """Lazy initialization — tries HSEmotion first, then FER fallback."""
        if self._is_initialized:
            return

        self._hsemotion = get_hsemotion_model()
        if self._hsemotion is not None:
            self._backend = "hsemotion"
            self._is_initialized = True
            return

        logger.warning("HSEmotion not available, falling back to FER")
        self._initialize_fer()

    def _initialize_fer(self):
        """Initialize FER detector as fallback"""
        try:
            from fer import FER
            self._fer_detector = FER(mtcnn=True)
            self._is_initialized = True
            self._backend = "fer"
            logger.info("FER initialized (FER2013 fallback)")
        except Exception as e:
            logger.error("FER initialization error: %s", e)
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

            # Detect emotions — HSEmotion (primary) or FER (fallback)
            if self._backend == "hsemotion" and self._hsemotion is not None:
                result = self._analyze_with_hsemotion(img_rgb)
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

            # Normalize labels: happy→joy, sad→sadness, angry→anger
            emotions_normalized = self._normalize_emotions(emotions_percent)
            primary_emotion = self.LABEL_NORMALIZATION.get(primary_emotion_fer, primary_emotion_fer)

            # Face box
            face_box = {
                "x": int(box[0]),
                "y": int(box[1]),
                "width": int(box[2]),
                "height": int(box[3])
            }

            # XAI explanation (uses FER labels because FACS mapping uses the originals)
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
                logger.warning("Masking detection error: %s", e)

            processing_time = (time.time() - start_time) * 1000

            # Calibrate confidence based on contextual factors
            confidence = self._calibrate_confidence(
                raw_confidence=confidence,
                emotions=emotions_normalized,
                face_box=face_box,
                image_shape=img_rgb.shape,
                masking_result=masking_result,
            )

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
            logger.error("Face analysis error: %s", e, exc_info=True)
            return self._error_result("Face analysis failed")

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

            # Use HSEmotion if available, else FER
            self._initialize()
            if self._backend == "hsemotion" and self._hsemotion is not None:
                frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                result = self._analyze_with_hsemotion(frame_rgb)
            else:
                if self._fast_detector is None:
                    from fer import FER
                    self._fast_detector = FER(mtcnn=False)
                result = self._fast_detector.detect_emotions(small_frame)

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

            # Normalize labels: happy→joy, sad→sadness, angry→anger
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
                logger.warning("Masking detection error (fast): %s", e)

            # Generate XAI explanation
            xai_explanation = self.generate_explanation(emotions_raw, primary_fer)

            # Calibrate confidence based on contextual factors
            raw_confidence = emotions[primary]
            calibrated_confidence = self._calibrate_confidence(
                raw_confidence=raw_confidence,
                emotions=emotions,
                face_box=face_box,
                image_shape=frame.shape,
                masking_result=masking_result,
            )

            response = {
                "face_detected": True,
                "emotions": emotions,
                "primary_emotion": primary,
                "confidence": calibrated_confidence,
                "face_box": face_box,
                "xai_explanation": xai_explanation,
                "timestamp": time.time()
            }

            if masking_result:
                response["masking"] = masking_result

            return response

        except Exception as e:
            logger.error("Fast frame analysis error: %s", e, exc_info=True)
            return {
                "face_detected": False,
                "emotions": {},
                "error": "Frame analysis failed",
                "timestamp": time.time()
            }

    def _analyze_with_hsemotion(self, img_rgb: np.ndarray) -> List[Dict]:
        """
        Analyze face using HSEmotion (AffectNet, state-of-the-art).
        Returns result in same format as FER for compatibility.
        """
        try:
            import cv2

            # HSEmotion expects BGR image and uses its own face detector
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # Detect faces and predict emotions
            emotions_list, scores_list = self._hsemotion.predict_emotions(
                img_bgr, logits=False
            )

            if not emotions_list or len(emotions_list) == 0:
                return []

            converted = []
            for i, (emotion_label, scores) in enumerate(zip(emotions_list, scores_list)):
                # Build emotions dict from scores array
                emotions_dict = {}
                for j, label in enumerate(self.HSEMOTION_LABELS):
                    if j < len(scores):
                        emotions_dict[label] = float(scores[j])

                # HSEmotion doesn't return face boxes directly,
                # use a placeholder (full image)
                h, w = img_rgb.shape[:2]
                box = (0, 0, w, h)

                converted.append({
                    "emotions": emotions_dict,
                    "box": box,
                })

            return converted

        except Exception as e:
            logger.warning("HSEmotion analysis error: %s, falling back to FER", e)
            if self._fer_detector is None:
                from fer import FER
                self._fer_detector = FER(mtcnn=True)
            return self._fer_detector.detect_emotions(img_rgb)

    def generate_explanation(self, emotions: Dict[str, float], primary_emotion: str) -> Dict:
        """Generates XAI explanation for face analysis"""

        # Facial Action Units (FACS) for each emotion
        facial_action_units = {
            "happy": [
                "AU6 (Cheek Raiser) - cheek raising",
                "AU12 (Lip Corner Puller) - smile"
            ],
            "sad": [
                "AU1 (Inner Brow Raiser) - inner brow raising",
                "AU4 (Brow Lowerer) - brow lowering",
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
            "sad": "Lowered eyebrows and lip corners are characteristic of sadness.",
            "angry": "Furrowed brows and a tense facial expression indicate anger.",
            "fear": "Raised eyebrows and wide open eyes suggest fear.",
            "surprise": "Raised eyebrows and an open mouth indicate surprise.",
            "disgust": "A wrinkled nose and lowered lips are characteristic of disgust.",
            "neutral": "The face is relaxed with no pronounced emotions."
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

    def _calibrate_confidence(self, raw_confidence, emotions, face_box=None, image_shape=None, masking_result=None):
        """
        Adjusts raw model confidence based on contextual factors:
        - Face box size relative to image
        - Emotion distribution clarity (gap between top two emotions)
        - Masking detection penalty
        Returns calibrated confidence clamped to [5.0, 99.0].
        """
        calibrated = raw_confidence

        # 1. Face box size penalty
        if face_box is not None and image_shape is not None:
            img_h, img_w = image_shape[:2]
            image_area = img_h * img_w
            if image_area > 0:
                face_area = face_box["width"] * face_box["height"]
                face_ratio = face_area / image_area
                if face_ratio < 0.10:
                    calibrated -= 15.0
                elif face_ratio < 0.30:
                    calibrated -= 5.0

        # 2. Emotion distribution clarity penalty
        if emotions and len(emotions) >= 2:
            sorted_scores = sorted(emotions.values(), reverse=True)
            top_gap = sorted_scores[0] - sorted_scores[1]
            if top_gap < 10.0:
                calibrated -= 10.0
            # No change if gap > 30%; implicit (no addition)

        # 3. Masking penalty
        if masking_result and masking_result.get("masking_detected"):
            masking_conf = masking_result.get("masking_confidence", 0.0)
            calibrated -= masking_conf * 20.0

        # Clamp to [5.0, 99.0]
        calibrated = max(5.0, min(99.0, calibrated))
        return round(calibrated, 1)

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
