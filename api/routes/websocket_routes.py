"""
WebSocket Routes for Live Camera Analysis
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.face_analyzer import face_analyzer
import json
import asyncio

router = APIRouter()


class ConnectionManager:
    """Upravlja WebSocket konekcijama"""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_json(self, websocket: WebSocket, data: dict):
        await websocket.send_json(data)


manager = ConnectionManager()


@router.websocket("/camera")
async def websocket_camera_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint za live camera emotion analysis.

    Client šalje: { "frame": "base64_encoded_image" }
    Server vraća: { "face_detected": bool, "emotions": {...}, "primary_emotion": str }

    Primjer korištenja u Flutter/Dart:
    ```dart
    final channel = WebSocketChannel.connect(Uri.parse('ws://server/live/camera'));
    channel.sink.add(jsonEncode({'frame': base64Frame}));
    channel.stream.listen((response) {
      final data = jsonDecode(response);
      print('Emotions: ${data['emotions']}');
    });
    ```
    """
    await manager.connect(websocket)

    try:
        while True:
            # Prima frame od klijenta
            data = await websocket.receive_text()

            try:
                frame_data = json.loads(data)

                if "frame" not in frame_data:
                    await manager.send_json(websocket, {
                        "error": "Missing 'frame' field",
                        "face_detected": False
                    })
                    continue

                # Analiziraj frame
                result = face_analyzer.analyze_frame_fast(frame_data["frame"])

                # Pošalji rezultat
                await manager.send_json(websocket, result)

            except json.JSONDecodeError:
                await manager.send_json(websocket, {
                    "error": "Invalid JSON",
                    "face_detected": False
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.websocket("/camera/session")
async def websocket_camera_session(websocket: WebSocket):
    """
    WebSocket za kompletnu live sesiju sa agregiranim rezultatima.

    Pored real-time emocija, prati i:
    - Prosječne emocije tokom sesije
    - Timeline emocija
    - Dominantna emocija

    Client poruke:
    - {"action": "start"} - Počni sesiju
    - {"action": "frame", "frame": "base64"} - Pošalji frame
    - {"action": "end"} - Završi sesiju i dobij sumarni rezultat
    """
    await manager.connect(websocket)

    session_data = {
        "frames_analyzed": 0,
        "emotion_timeline": [],
        "emotion_sums": {
            "angry": 0, "disgust": 0, "fear": 0,
            "happy": 0, "sad": 0, "surprise": 0, "neutral": 0
        },
        "start_time": None,
        "is_active": False
    }

    try:
        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                action = message.get("action", "frame")

                if action == "start":
                    # Počni novu sesiju
                    import time
                    session_data = {
                        "frames_analyzed": 0,
                        "emotion_timeline": [],
                        "emotion_sums": {
                            "angry": 0, "disgust": 0, "fear": 0,
                            "happy": 0, "sad": 0, "surprise": 0, "neutral": 0
                        },
                        "start_time": time.time(),
                        "is_active": True
                    }
                    await manager.send_json(websocket, {
                        "status": "session_started",
                        "message": "Live emotion tracking started"
                    })

                elif action == "frame" and session_data["is_active"]:
                    if "frame" not in message:
                        continue

                    # Analiziraj frame
                    result = face_analyzer.analyze_frame_fast(message["frame"])

                    if result.get("face_detected"):
                        # Ažuriraj session data
                        session_data["frames_analyzed"] += 1
                        emotions = result.get("emotions", {})

                        for emotion, score in emotions.items():
                            if emotion in session_data["emotion_sums"]:
                                session_data["emotion_sums"][emotion] += score

                        # Dodaj u timeline (svaki 5. frame)
                        if session_data["frames_analyzed"] % 5 == 0:
                            session_data["emotion_timeline"].append({
                                "timestamp": result["timestamp"],
                                "emotions": emotions,
                                "primary": result.get("primary_emotion")
                            })

                    # Pošalji real-time rezultat
                    await manager.send_json(websocket, result)

                elif action == "end":
                    # Završi sesiju i vrati sumarni rezultat
                    import time
                    session_data["is_active"] = False

                    if session_data["frames_analyzed"] > 0:
                        # Izračunaj prosječne emocije
                        avg_emotions = {
                            k: round(v / session_data["frames_analyzed"], 1)
                            for k, v in session_data["emotion_sums"].items()
                        }

                        # Pronađi dominantnu emociju
                        dominant = max(avg_emotions, key=avg_emotions.get)

                        summary = {
                            "status": "session_ended",
                            "session_duration_seconds": round(
                                time.time() - session_data["start_time"], 2
                            ) if session_data["start_time"] else 0,
                            "frames_analyzed": session_data["frames_analyzed"],
                            "average_emotions": avg_emotions,
                            "dominant_emotion": dominant,
                            "emotion_timeline": session_data["emotion_timeline"][-20:],  # Last 20
                            "xai_explanation": {
                                "method": "temporal_emotion_analysis",
                                "reasoning": f"Tokom sesije od {session_data['frames_analyzed']} "
                                           f"analiziranih frameova, dominantna emocija je bila "
                                           f"'{dominant}' sa prosječnim skorom od "
                                           f"{avg_emotions[dominant]}%."
                            }
                        }
                    else:
                        summary = {
                            "status": "session_ended",
                            "frames_analyzed": 0,
                            "message": "No faces detected during session"
                        }

                    await manager.send_json(websocket, summary)

            except json.JSONDecodeError:
                await manager.send_json(websocket, {"error": "Invalid JSON"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket session error: {e}")
        manager.disconnect(websocket)
