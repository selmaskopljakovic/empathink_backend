"""
WebSocket Routes for Live Camera Analysis
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.face_analyzer import face_analyzer
from services.conversation_engine import conversation_engine
from services.head_gesture_detector import head_gesture_detector
import json
import asyncio
import uuid

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections"""

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
    WebSocket endpoint for live camera emotion analysis.
    Now includes detection of masked emotions (fake smile, suppressed emotions).

    Client sends: { "frame": "base64_encoded_image" }
    Server returns: { "face_detected": bool, "emotions": {...}, "primary_emotion": str, "masking": {...} }

    Usage example in Flutter/Dart:
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
            # Receive frame from client
            data = await websocket.receive_text()

            try:
                frame_data = json.loads(data)

                if "frame" not in frame_data:
                    await manager.send_json(websocket, {
                        "error": "Missing 'frame' field",
                        "face_detected": False
                    })
                    continue

                # Analyze frame with emotion history for masking detection
                result = face_analyzer.analyze_frame_fast(
                    frame_data["frame"],
                    emotion_history=emotion_history,
                )

                # Add to history if face was detected
                if result.get("face_detected") and result.get("emotions"):
                    emotion_history.append(result["emotions"])
                    if len(emotion_history) > max_history:
                        emotion_history.pop(0)

                # Send result
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
    WebSocket for a complete live session with aggregated results.

    In addition to real-time emotions, it also tracks:
    - Average emotions during the session
    - Emotion timeline
    - Dominant emotion

    Client messages:
    - {"action": "start"} - Start session
    - {"action": "frame", "frame": "base64"} - Send frame
    - {"action": "end"} - End session and get summary result
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
                    # Start new session
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

                    # Analyze frame with emotion history for masking detection
                    result = face_analyzer.analyze_frame_fast(
                        message["frame"],
                        emotion_history=emotion_history,
                    )

                    if result.get("face_detected"):
                        # Update session data
                        session_data["frames_analyzed"] += 1
                        emotions = result.get("emotions", {})

                        for emotion, score in emotions.items():
                            if emotion in session_data["emotion_sums"]:
                                session_data["emotion_sums"][emotion] += score

                        # Add to emotion history
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

                        # Add to timeline (every 5th frame)
                        if session_data["frames_analyzed"] % 5 == 0:
                            session_data["emotion_timeline"].append({
                                "timestamp": result["timestamp"],
                                "emotions": emotions,
                                "primary": result.get("primary_emotion")
                            })

                    # Send real-time result
                    await manager.send_json(websocket, result)

                elif action == "end":
                    # End session and return summary result
                    import time
                    session_data["is_active"] = False

                    if session_data["frames_analyzed"] > 0:
                        # Calculate average emotions
                        avg_emotions = {
                            k: round(v / session_data["frames_analyzed"], 1)
                            for k, v in session_data["emotion_sums"].items()
                        }

                        # Find dominant emotion
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
                                "reasoning": f"During the session of {session_data['frames_analyzed']} "
                                           f"analyzed frames, the dominant emotion was "
                                           f"'{dominant}' with an average score of "
                                           f"{avg_emotions[dominant]}%."
                                           + (f" {len(masking_events)} possible "
                                              f"emotion masking events were detected."
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


@router.websocket("/camera/conversation")
async def websocket_camera_conversation(websocket: WebSocket):
    """
    WebSocket endpoint for conversational AI with live camera.

    Multiplexes frames, text, gesture frames and session.

    Client → Server:
    - {"type": "session_start"}
    - {"type": "frame", "frame": "<base64>"}
    - {"type": "text_message", "text": "...", "source": "keyboard|voice"}
    - {"type": "gesture_frames", "frames": ["<base64>", ...]}
    - {"type": "validation", "emotion": "...", "correct": bool}
    - {"type": "session_end"}

    Server → Client:
    - {"type": "emotion_result", ..., "xai_explanation": {...}}
    - {"type": "ai_message", "text": "...", "emotion_observation": "...", "suggested_actions": [...]}
    - {"type": "gesture_result", "gesture": "nod|shake|none", "confidence": 0.85}
    - {"type": "visual_observation", "observations": {...}, "timestamp": float}
    - {"type": "session_summary", ...}
    """
    await manager.connect(websocket)

    session_id = str(uuid.uuid4())
    emotion_history = []
    max_history = 20
    frame_count = 0
    latest_emotions = None
    latest_masking = None
    latest_frame_b64 = None  # Store latest frame for Gemini vision

    try:
        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                msg_type = message.get("type", "")

                if msg_type == "session_start":
                    # Start conversation session
                    greeting = conversation_engine.start_session(session_id)
                    await manager.send_json(websocket, {
                        "type": "ai_message",
                        **greeting,
                    })

                elif msg_type == "frame":
                    if "frame" not in message:
                        continue

                    latest_frame_b64 = message["frame"]

                    # Analyze frame for emotions (non-blocking)
                    result = await asyncio.to_thread(
                        face_analyzer.analyze_frame_fast,
                        message["frame"],
                        emotion_history,
                    )

                    if result.get("face_detected") and result.get("emotions"):
                        emotion_history.append(result["emotions"])
                        if len(emotion_history) > max_history:
                            emotion_history.pop(0)
                        latest_emotions = result["emotions"]
                        latest_masking = result.get("masking")

                    # Send emotion result (now includes xai_explanation from face_analyzer)
                    await manager.send_json(websocket, {
                        "type": "emotion_result",
                        **result,
                    })

                    # Stagger Gemini calls to avoid rate limiting:
                    # - Proactive AI comment at frame 7, 17, 27...
                    # - Vision analysis at frame 3, 13, 23...
                    frame_count += 1

                    if frame_count % 10 == 7 and latest_emotions:
                        try:
                            ai_response = await asyncio.to_thread(
                                conversation_engine.generate_response,
                                session_id,
                                latest_emotions,
                                latest_masking,
                                None,  # no user message
                                None,  # no gesture
                            )
                            await manager.send_json(websocket, {
                                "type": "ai_message",
                                **ai_response,
                            })
                        except Exception as e:
                            print(f"Proactive AI comment error: {e}")

                    # Gemini vision analysis every 10th frame (~20s) - staggered
                    if frame_count % 10 == 3 and latest_frame_b64:
                        try:
                            print(f"[WS] Triggering vision analysis (frame {frame_count})")
                            observations = await asyncio.to_thread(
                                conversation_engine.analyze_visual_details,
                                latest_frame_b64,
                            )
                            if observations:
                                import time
                                await manager.send_json(websocket, {
                                    "type": "visual_observation",
                                    "observations": observations,
                                    "timestamp": time.time(),
                                })
                                print(f"[WS] Vision sent: {list(observations.keys())}")
                            else:
                                print("[WS] Vision returned None")
                        except Exception as e:
                            print(f"Gemini vision analysis error: {e}")

                elif msg_type == "text_message":
                    user_text = message.get("text", "").strip()
                    if not user_text:
                        continue

                    # Generate AI response with emotion context
                    try:
                        ai_response = await asyncio.to_thread(
                            conversation_engine.generate_response,
                            session_id,
                            latest_emotions,
                            latest_masking,
                            user_text,
                            None,  # no gesture
                        )
                        await manager.send_json(websocket, {
                            "type": "ai_message",
                            **ai_response,
                        })
                    except Exception as e:
                        print(f"Text response error: {e}")
                        await manager.send_json(websocket, {
                            "type": "ai_message",
                            "text": "Thank you. Continue when you are ready.",
                            "emotion_observation": None,
                            "suggested_actions": [],
                        })

                elif msg_type == "validation":
                    # Handle user emotion validation
                    emotion = message.get("emotion", "")
                    correct = message.get("correct", False)
                    print(f"[Validation] emotion={emotion}, correct={correct}, session={session_id}")

                    # Feed validation context to conversation engine
                    if not correct and emotion:
                        try:
                            validation_context = f"[User says they are NOT {emotion}. Ask them how they really feel.]"
                            ai_response = await asyncio.to_thread(
                                conversation_engine.generate_response,
                                session_id,
                                latest_emotions,
                                latest_masking,
                                validation_context,
                                None,
                            )
                            await manager.send_json(websocket, {
                                "type": "ai_message",
                                **ai_response,
                            })
                        except Exception as e:
                            print(f"Validation response error: {e}")

                elif msg_type == "gesture_frames":
                    frames = message.get("frames", [])
                    if not frames:
                        continue

                    # Detect gesture (non-blocking)
                    try:
                        gesture_result = await asyncio.to_thread(
                            head_gesture_detector.detect_gesture,
                            frames,
                        )
                        await manager.send_json(websocket, {
                            "type": "gesture_result",
                            **gesture_result,
                        })

                        # If nod or shake detected, feed to conversation engine
                        if gesture_result["gesture"] != "none":
                            ai_response = await asyncio.to_thread(
                                conversation_engine.generate_response,
                                session_id,
                                latest_emotions,
                                latest_masking,
                                None,  # no text
                                gesture_result["gesture"],
                            )
                            await manager.send_json(websocket, {
                                "type": "ai_message",
                                **ai_response,
                            })
                    except Exception as e:
                        print(f"Gesture detection error: {e}")
                        await manager.send_json(websocket, {
                            "type": "gesture_result",
                            "gesture": "none",
                            "confidence": 0.0,
                        })

                elif msg_type == "session_end":
                    # Generate session summary
                    try:
                        summary = await asyncio.to_thread(
                            conversation_engine.generate_summary,
                            session_id,
                        )
                        await manager.send_json(websocket, {
                            "type": "session_summary",
                            **summary,
                        })
                    except Exception as e:
                        print(f"Summary generation error: {e}")
                        await manager.send_json(websocket, {
                            "type": "session_summary",
                            "summary": "Session ended.",
                            "message_count": frame_count,
                        })
                    finally:
                        conversation_engine.cleanup_session(session_id)

            except json.JSONDecodeError:
                await manager.send_json(websocket, {
                    "type": "error",
                    "message": "Invalid JSON",
                })

    except WebSocketDisconnect:
        conversation_engine.cleanup_session(session_id)
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket conversation error: {e}")
        conversation_engine.cleanup_session(session_id)
        manager.disconnect(websocket)
