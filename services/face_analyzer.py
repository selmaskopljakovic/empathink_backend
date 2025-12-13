"""
Face Emotion Analyzer Service
Uses FER and MediaPipe for facial expression recognition
"""

import time
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
import io
import base64


class FaceEmotionAnalyzer:
    """
    Analizira slike lica i detektuje emocije koristeći:
    - FER (Facial Expression Recognition) biblioteku
    - MediaPipe za face detection i landmarks
    - OpenCV za image processing
    """

    # Mapiranje FER emocija
    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def __init__(self):
        self._fer_detector = None
        self._is_initialized = False

    def _initialize(self):
        """Lazy initialization FER detektora"""
        if self._is_initialized:
            return

        try:
            from fer import FER
            # Koristi MTCNN za bolju detekciju lica
            self._fer_detector = FER(mtcnn=True)
            self._is_initialized = True
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
            from fer import FER

            # Inicijaliziraj FER ako nije
            if self._fer_detector is None:
                self._fer_detector = FER(mtcnn=True)

            # Učitaj sliku iz bytes
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return self._error_result("Could not decode image")

            # Konvertuj u RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detektuj emocije
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

            # Pronađi primarnu emociju
            primary_emotion = max(emotions, key=emotions.get)
            confidence = round(emotions[primary_emotion] * 100, 1)

            # Face box
            face_box = {
                "x": int(box[0]),
                "y": int(box[1]),
                "width": int(box[2]),
                "height": int(box[3])
            }

            # XAI objašnjenje
            xai_explanation = None
            if include_xai:
                xai_explanation = self._generate_explanation(emotions_percent, primary_emotion)

            processing_time = (time.time() - start_time) * 1000

            return {
                "success": True,
                "face_detected": True,
                "emotions": emotions_percent,
                "primary_emotion": primary_emotion,
                "confidence": confidence,
                "face_box": face_box,
                "xai_explanation": xai_explanation,
                "processing_time_ms": round(processing_time, 2),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Face analysis error: {e}")
            return self._error_result(str(e))

    def analyze_frame_fast(self, frame_base64: str) -> Dict:
        """
        Brza analiza jednog frame-a za live camera.
        Ne koristi MTCNN za veću brzinu.

        Args:
            frame_base64: Base64 encoded frame

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
            emotions = {k: round(v * 100, 1) for k, v in face["emotions"].items()}
            primary = max(emotions, key=emotions.get)

            # Skaliraj box nazad
            box = face["box"]
            face_box = {
                "x": int(box[0] / scale),
                "y": int(box[1] / scale),
                "width": int(box[2] / scale),
                "height": int(box[3] / scale)
            }

            return {
                "face_detected": True,
                "emotions": emotions,
                "primary_emotion": primary,
                "confidence": emotions[primary],
                "face_box": face_box,
                "timestamp": time.time()
            }

        except Exception as e:
            print(f"Fast frame analysis error: {e}")
            return {
                "face_detected": False,
                "emotions": {},
                "error": str(e),
                "timestamp": time.time()
            }

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
