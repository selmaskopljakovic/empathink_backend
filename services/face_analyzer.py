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
    Analizira slike lica i detektuje emocije koristeći:
    - DeepFace (default): AffectNet dataset, bolja preciznost
    - FER (fallback): FER2013 dataset, lakši model
    - MTCNN za face detection
    - OpenCV za image processing
    """

    # FER emotion labels
    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    # Normalizacija FER labela na Ekman standard (isti kao text_analyzer i voice_analyzer)
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
        """Normalizira FER labele (happy/sad/angry) na standard (joy/sadness/anger)."""
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
        Analizira sliku i vraća emocije sa procentima.

        Args:
            image_data: Slika kao bytes
            include_xai: Da li uključiti XAI objašnjenja

        Returns:
            Dict sa emocijama i face box koordinatama
        """
        start_time = time.time()

        try:
            import cv2

            # Lazy initialize
            self._initialize()

            # Učitaj sliku iz bytes
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return self._error_result("Could not decode image")

            # Konvertuj u RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detektuj emocije — DeepFace ili FER
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
                        "reasoning": "Lice nije detektovano na slici. Molimo pokušajte sa boljim osvjetljenjem."
                    },
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                    "timestamp": datetime.now().isoformat()
                }

            # Uzmi prvo detektovano lice
            face = result[0]
            emotions = face["emotions"]
            box = face["box"]

            # Konvertuj u procente
            emotions_percent = {k: round(v * 100, 1) for k, v in emotions.items()}

            # Pronađi primarnu emociju (FER labele za XAI)
            primary_emotion_fer = max(emotions, key=emotions.get)
            confidence = round(emotions[primary_emotion_fer] * 100, 1)

            # Normaliziraj labele: happy→joy, sad→sadness, angry→anger
            emotions_normalized = self._normalize_emotions(emotions_percent)
            primary_emotion = self.LABEL_NORMALIZATION.get(primary_emotion_fer, primary_emotion_fer)

            # Face box
            face_box = {
                "x": int(box[0]),
                "y": int(box[1]),
                "width": int(box[2]),
                "height": int(box[3])
            }

            # XAI objašnjenje (koristi FER labele jer FACS mapiranje koristi originalne)
            xai_explanation = None
            if include_xai:
                xai_explanation = self._generate_explanation(emotions_percent, primary_emotion_fer)

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
        Brza analiza jednog frame-a za live camera.
        Ne koristi MTCNN za veću brzinu.

        Args:
            frame_base64: Base64 encoded frame
            emotion_history: Lista prethodnih emocija za temporalnu analizu maskiranja

        Returns:
            Dict sa emocijama za real-time prikaz
        """
        try:
            import cv2
            from fer import FER

            # Kreiraj brzi detektor (bez MTCNN)
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

            # Smanji rezoluciju za brzinu
            scale = 0.5
            small_frame = cv2.resize(frame, None, fx=scale, fy=scale)

            # Detektuj emocije
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

            # Normaliziraj labele: happy→joy, sad→sadness, angry→anger
            emotions = self._normalize_emotions(emotions_raw)
            primary_fer = max(emotions_raw, key=emotions_raw.get)
            primary = self.LABEL_NORMALIZATION.get(primary_fer, primary_fer)

            # Skaliraj box nazad
            box = face["box"]
            face_box = {
                "x": int(box[0] / scale),
                "y": int(box[1] / scale),
                "width": int(box[2] / scale),
                "height": int(box[3] / scale)
            }

            # Masking detection (koristi originalni frame za landmarks)
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

            response = {
                "face_detected": True,
                "emotions": emotions,
                "primary_emotion": primary,
                "confidence": emotions[primary],
                "face_box": face_box,
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

    def _generate_explanation(self, emotions: Dict[str, float], primary_emotion: str) -> Dict:
        """Generiše XAI objašnjenje za face analizu"""

        # Facial Action Units (FACS) za svaku emociju
        facial_action_units = {
            "happy": [
                "AU6 (Cheek Raiser) - podizanje obraza",
                "AU12 (Lip Corner Puller) - osmijeh"
            ],
            "sad": [
                "AU1 (Inner Brow Raiser) - podizanje unutrašnjeg dijela obrva",
                "AU4 (Brow Lowerer) - spuštanje obrva",
                "AU15 (Lip Corner Depressor) - spušteni uglovi usana"
            ],
            "angry": [
                "AU4 (Brow Lowerer) - namrštene obrve",
                "AU5 (Upper Lid Raiser) - široko otvorene oči",
                "AU7 (Lid Tightener) - stisnuti kapci"
            ],
            "fear": [
                "AU1+2 (Brow Raiser) - podignute obrve",
                "AU5 (Upper Lid Raiser) - široko otvorene oči",
                "AU20 (Lip Stretcher) - rastegnute usne"
            ],
            "surprise": [
                "AU1+2 (Brow Raiser) - podignute obrve",
                "AU5 (Upper Lid Raiser) - široko otvorene oči",
                "AU26 (Jaw Drop) - otvorena usta"
            ],
            "disgust": [
                "AU9 (Nose Wrinkler) - naboran nos",
                "AU15 (Lip Corner Depressor) - spuštene usne",
                "AU16 (Lower Lip Depressor) - spuštena donja usna"
            ],
            "neutral": [
                "Nema značajnih aktivacija facijalnih mišića"
            ]
        }

        explanations = {
            "happy": "Podignuti obrazi i osmijeh ukazuju na sreću.",
            "sad": "Spuštene obrve i uglovi usana karakteristični su za tugu.",
            "angry": "Namrštene obrve i stisnut izraz lica ukazuju na ljutnju.",
            "fear": "Podignute obrve i široko otvorene oči sugerišu strah.",
            "surprise": "Podignute obrve i otvorena usta ukazuju na iznenađenje.",
            "disgust": "Naboran nos i spuštene usne karakteristični su za gađenje.",
            "neutral": "Lice je opušteno bez izraženih emocija."
        }

        # Sortiraj emocije za breakdown
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

        return {
            "method": "facial_action_coding_system",
            "reasoning": explanations.get(primary_emotion, "Analiza facijalnih ekspresija."),
            "facial_action_units": facial_action_units.get(primary_emotion, []),
            "confidence_breakdown": {e[0]: e[1] for e in sorted_emotions[:4]},
            "interpretation": f"Model je analizirao facijalne mišiće i detektovao "
                            f"'{primary_emotion}' kao dominantnu ekspresiju sa "
                            f"{emotions[primary_emotion]}% sigurnošću."
        }

    def _error_result(self, error_message: str) -> Dict:
        """Vraća standardni error response"""
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
