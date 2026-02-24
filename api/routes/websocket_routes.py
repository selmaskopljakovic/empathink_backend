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
    Sada uključuje i detekciju maskiranih emocija (lažni osmijeh, potisnute emocije).

    Client šalje: { "frame": "base64_encoded_image" }
    Server vraća: { "face_detected": bool, "emotions": {...}, "primary_emotion": str, "masking": {...} }

    Primjer korištenja u Flutter/Dart:
    ```dart
    final channel = WebSocketChannel.connect(Uri.parse('ws://server/live/camera'));
    channel.sink.add(jsonEncode({'frame': base64Frame}));
    channel.stream.listen((response) {
      final data = jsonDecode(response);
      print('Emotions: ${data['emotions']}');
      if (data['masking'] != null) print('Masking detected!');
    });
    ```
    """
    await manager.connect(websocket)

    # Emotion history per connection for temporal masking analysis
    emotion_history = []
    max_history = 20

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

                # Analiziraj frame sa emotion history za masking detekciju
                result = face_analyzer.analyze_frame_fast(
                    frame_data["frame"],
                    emotion_history=emotion_history,
                )

                # Dodaj u historiju ako je lice detektovano
                if result.get("face_detected") and result.get("emotions"):
                    emotion_history.append(result["emotions"])
                    if len(emotion_history) > max_history:
                        emotion_history.pop(0)

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

    # Emotion history and masking tracking per session
    emotion_history = []
    max_history = 20
    masking_events = []

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
                    emotion_history.clear()
                    masking_events.clear()
                    await manager.send_json(websocket, {
                        "status": "session_started",
                        "message": "Live emotion tracking started"
                    })

                elif action == "frame" and session_data["is_active"]:
                    if "frame" not in message:
                        continue

                    # Analiziraj frame sa emotion history za masking detekciju
                    result = face_analyzer.analyze_frame_fast(
                        message["frame"],
                        emotion_history=emotion_history,
                    )

                    if result.get("face_detected"):
                        # Ažuriraj session data
                        session_data["frames_analyzed"] += 1
                        emotions = result.get("emotions", {})

                        for emotion, score in emotions.items():
                            if emotion in session_data["emotion_sums"]:
                                session_data["emotion_sums"][emotion] += score

                        # Dodaj u emotion history
                        emotion_history.append(emotions)
                        if len(emotion_history) > max_history:
                            emotion_history.pop(0)

                        # Track masking events
                        if result.get("masking") and result["masking"].get("detected"):
                            masking_events.append({
                                "timestamp": result["timestamp"],
                                "type": result["masking"]["type"],
                                "confidence": result["masking"]["confidence"],
                                "surface_emotion": result["masking"].get("surface_emotion"),
                                "underlying_emotion": result["masking"].get("underlying_emotion"),
                            })

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

                        # Build masking summary
                        masking_summary = None
                        if masking_events:
                            # Count types
                            type_counts = {}
                            total_confidence = 0
                            for event in masking_events:
                                t = event["type"]
                                type_counts[t] = type_counts.get(t, 0) + 1
                                total_confidence += event["confidence"]

                            most_common_type = max(type_counts, key=type_counts.get)
                            masking_summary = {
                                "total_events": len(masking_events),
                                "type_counts": type_counts,
                                "most_common_type": most_common_type,
                                "average_confidence": round(
                                    total_confidence / len(masking_events), 2
                                ),
                                "masking_ratio": round(
                                    len(masking_events) / session_data["frames_analyzed"], 2
                                ),
                                "events": masking_events[-10:],  # Last 10
                            }

                        summary = {
                            "status": "session_ended",
                            "session_duration_seconds": round(
                                time.time() - session_data["start_time"], 2
                            ) if session_data["start_time"] else 0,
                            "frames_analyzed": session_data["frames_analyzed"],
                            "average_emotions": avg_emotions,
                            "dominant_emotion": dominant,
                            "emotion_timeline": session_data["emotion_timeline"][-20:],  # Last 20
                            "masking_summary": masking_summary,
                            "xai_explanation": {
                                "method": "temporal_emotion_analysis",
                                "reasoning": f"Tokom sesije od {session_data['frames_analyzed']} "
                                           f"analiziranih frameova, dominantna emocija je bila "
                                           f"'{dominant}' sa prosječnim skorom od "
                                           f"{avg_emotions[dominant]}%."
                                           + (f" Detektovano je {len(masking_events)} "
                                              f"mogućih maskiranja emocija."
                                              if masking_events else "")
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
