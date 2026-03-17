"""
WebSocket Routes for Live Camera Analysis
"""

import logging
import time as _time

from typing import Dict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.face_analyzer import face_analyzer
from services.conversation_engine import conversation_engine
from services.head_gesture_detector import head_gesture_detector
from services.usage_tracker import usage_tracker
from api.auth import verify_ws_token
import json
import asyncio
import uuid
import base64
from api.file_validation import validate_image_bytes

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
MAX_CONCURRENT_CONNECTIONS = 20          # max WebSocket clients at once
MAX_FRAME_BASE64_BYTES = 500_000        # ~375 KB decoded image
MAX_FRAMES_PER_SECOND = 5              # per-connection frame throttle
MAX_MASKING_EVENTS = 100               # cap masking_events list per session
MAX_GESTURE_FRAMES = 30                # max frames in a gesture_frames message
MAX_MESSAGE_BYTES = 600_000            # reject raw WS messages bigger than this


class _FrameThrottler:
    """Per-connection token-bucket style frame rate limiter."""

    def __init__(self, max_fps: int = MAX_FRAMES_PER_SECOND):
        self._min_interval = 1.0 / max_fps
        self._last_accepted = 0.0
        self.dropped = 0

    def allow(self) -> bool:
        now = _time.monotonic()
        if now - self._last_accepted < self._min_interval:
            self.dropped += 1
            return False
        self._last_accepted = now
        return True


def _validate_frame(frame_b64: str) -> str | None:
    """Return an error string if the frame is invalid, else None.

    Checks: type, size, base64 decodability, and image magic bytes.
    """
    if not isinstance(frame_b64, str):
        return "Frame must be a string"
    if len(frame_b64) > MAX_FRAME_BASE64_BYTES:
        return f"Frame too large ({len(frame_b64)} bytes, max {MAX_FRAME_BASE64_BYTES})"

    # Strip optional data-URL prefix before decoding
    raw_b64 = frame_b64
    if raw_b64.startswith("data:") and "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw_b64, validate=True)
    except Exception:
        return "Frame is not valid base64"

    if not validate_image_bytes(image_bytes):
        return "Frame does not contain a valid image (JPEG, PNG, or WebP required)"

    return None


class ConnectionManager:
    """Upravlja WebSocket konekcijama"""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._user_connections: Dict[str, int] = {}

    async def connect(self, websocket: WebSocket) -> bool:
        """Accept connection. Returns False if global limit reached."""
        if len(self.active_connections) >= MAX_CONCURRENT_CONNECTIONS:
            await websocket.accept()
            await websocket.send_json({
                "error": "Server at capacity, try again later",
            })
            await websocket.close(code=1013)  # Try Again Later
            logger.warning(
                "WebSocket rejected: %d/%d connections",
                len(self.active_connections), MAX_CONCURRENT_CONNECTIONS,
            )
            return False
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            "WebSocket connected (%d/%d)",
            len(self.active_connections), MAX_CONCURRENT_CONNECTIONS,
        )
        return True

    async def check_user_limit(self, websocket: WebSocket, user_id: str) -> bool:
        """Check per-user connection limit (max 3). Returns False if limit exceeded."""
        current = self._user_connections.get(user_id, 0)
        if current >= 3:
            await websocket.send_json({"error": "Too many concurrent connections"})
            await websocket.close(code=1008)
            return False
        self._user_connections[user_id] = current + 1
        return True

    def disconnect(self, websocket: WebSocket, user_id: str | None = None):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if user_id and user_id in self._user_connections:
            self._user_connections[user_id] = max(0, self._user_connections[user_id] - 1)
        logger.info(
            "WebSocket disconnected (%d/%d)",
            len(self.active_connections), MAX_CONCURRENT_CONNECTIONS,
        )

    async def send_json(self, websocket: WebSocket, data: dict):
        await websocket.send_json(data)


manager = ConnectionManager()


@router.websocket("/camera")
async def websocket_camera_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint za live camera emotion analysis.

    Auth: Client sends {"type": "auth", "token": "..."} as first message.
    Then: Client šalje: { "frame": "base64_encoded_image" }
    Server vraća: { "face_detected": bool, "emotions": {...}, "primary_emotion": str, "masking": {...} }
    """
    if not await manager.connect(websocket):
        return

    # Wait for auth token as first message (not in URL query parameter)
    try:
        first_msg = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        first_data = json.loads(first_msg)
        token = None
        if isinstance(first_data, dict) and first_data.get("type") == "auth":
            token = first_data.get("token")
        ws_user = await verify_ws_token(token)
        if ws_user is None:
            await websocket.send_json({"error": "Authentication required"})
            await websocket.close(code=1008)  # Policy Violation
            manager.disconnect(websocket)
            return
    except (asyncio.TimeoutError, json.JSONDecodeError, Exception):
        await websocket.send_json({"error": "Authentication required"})
        await websocket.close(code=1008)
        manager.disconnect(websocket)
        return

    user_id = ws_user.get("uid", "anonymous")
    if not await manager.check_user_limit(websocket, user_id):
        manager.disconnect(websocket, user_id=None)
        return

    throttler = _FrameThrottler()
    emotion_history = []
    max_history = 20

    try:
        while True:
            data = await websocket.receive_text()

            # Reject oversized raw messages
            if len(data) > MAX_MESSAGE_BYTES:
                await manager.send_json(websocket, {
                    "error": "Message too large",
                    "face_detected": False,
                })
                continue

            try:
                frame_data = json.loads(data)

                if "frame" not in frame_data:
                    await manager.send_json(websocket, {
                        "error": "Missing 'frame' field",
                        "face_detected": False,
                    })
                    continue

                # Validate frame size
                err = _validate_frame(frame_data["frame"])
                if err:
                    await manager.send_json(websocket, {
                        "error": err,
                        "face_detected": False,
                    })
                    continue

                # Throttle frame rate
                if not throttler.allow():
                    continue  # silently drop excess frames

                result = face_analyzer.analyze_frame_fast(
                    frame_data["frame"],
                    emotion_history=emotion_history,
                )

                if result.get("face_detected") and result.get("emotions"):
                    emotion_history.append(result["emotions"])
                    if len(emotion_history) > max_history:
                        emotion_history.pop(0)

                await manager.send_json(websocket, result)

            except json.JSONDecodeError:
                await manager.send_json(websocket, {
                    "error": "Invalid JSON",
                    "face_detected": False,
                })

    except WebSocketDisconnect:
        if throttler.dropped > 0:
            logger.info("WebSocket /camera: dropped %d frames (throttled)", throttler.dropped)
        manager.disconnect(websocket, user_id=user_id)
    except Exception as e:
        logger.error("WebSocket error: %s", e, exc_info=True)
        if throttler.dropped > 0:
            logger.info("WebSocket /camera: dropped %d frames (throttled)", throttler.dropped)
        manager.disconnect(websocket, user_id=user_id)


@router.websocket("/camera/session")
async def websocket_camera_session(websocket: WebSocket):
    """
    WebSocket za kompletnu live sesiju sa agregiranim rezultatima.

    Auth: Client sends {"type": "auth", "token": "..."} as first message.
    Then:
    - {"action": "start"} - Počni sesiju
    - {"action": "frame", "frame": "base64"} - Pošalji frame
    - {"action": "end"} - Završi sesiju i dobij sumarni rezultat
    """
    if not await manager.connect(websocket):
        return

    # Wait for auth token as first message (not in URL query parameter)
    try:
        first_msg = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        first_data = json.loads(first_msg)
        token = None
        if isinstance(first_data, dict) and first_data.get("type") == "auth":
            token = first_data.get("token")
        ws_user = await verify_ws_token(token)
        if ws_user is None:
            await websocket.send_json({"error": "Authentication required"})
            await websocket.close(code=1008)
            manager.disconnect(websocket)
            return
    except (asyncio.TimeoutError, json.JSONDecodeError, Exception):
        await websocket.send_json({"error": "Authentication required"})
        await websocket.close(code=1008)
        manager.disconnect(websocket)
        return

    user_id = ws_user.get("uid", "anonymous")
    if not await manager.check_user_limit(websocket, user_id):
        manager.disconnect(websocket, user_id=None)
        return

    throttler = _FrameThrottler()

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

    emotion_history = []
    max_history = 20
    masking_events = []

    try:
        while True:
            data = await websocket.receive_text()

            if len(data) > MAX_MESSAGE_BYTES:
                await manager.send_json(websocket, {"error": "Message too large"})
                continue

            try:
                message = json.loads(data)
                action = message.get("action", "frame")

                if action == "start":
                    session_data = {
                        "frames_analyzed": 0,
                        "emotion_timeline": [],
                        "emotion_sums": {
                            "angry": 0, "disgust": 0, "fear": 0,
                            "happy": 0, "sad": 0, "surprise": 0, "neutral": 0
                        },
                        "start_time": _time.time(),
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

                    # Validate frame size
                    err = _validate_frame(message["frame"])
                    if err:
                        await manager.send_json(websocket, {"error": err, "face_detected": False})
                        continue

                    # Throttle
                    if not throttler.allow():
                        continue

                    result = face_analyzer.analyze_frame_fast(
                        message["frame"],
                        emotion_history=emotion_history,
                    )

                    if result.get("face_detected"):
                        session_data["frames_analyzed"] += 1
                        emotions = result.get("emotions", {})

                        for emotion, score in emotions.items():
                            if emotion in session_data["emotion_sums"]:
                                session_data["emotion_sums"][emotion] += score

                        emotion_history.append(emotions)
                        if len(emotion_history) > max_history:
                            emotion_history.pop(0)

                        # Track masking events (capped)
                        if result.get("masking") and result["masking"].get("detected"):
                            if len(masking_events) < MAX_MASKING_EVENTS:
                                masking_events.append({
                                    "timestamp": result["timestamp"],
                                    "type": result["masking"]["type"],
                                    "confidence": result["masking"]["confidence"],
                                    "surface_emotion": result["masking"].get("surface_emotion"),
                                    "underlying_emotion": result["masking"].get("underlying_emotion"),
                                })

                        if session_data["frames_analyzed"] % 5 == 0:
                            session_data["emotion_timeline"].append({
                                "timestamp": result["timestamp"],
                                "emotions": emotions,
                                "primary": result.get("primary_emotion")
                            })

                    await manager.send_json(websocket, result)

                elif action == "end":
                    session_data["is_active"] = False

                    if session_data["frames_analyzed"] > 0:
                        avg_emotions = {
                            k: round(v / session_data["frames_analyzed"], 1)
                            for k, v in session_data["emotion_sums"].items()
                        }

                        dominant = max(avg_emotions, key=avg_emotions.get)

                        masking_summary = None
                        if masking_events:
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
                                "events": masking_events[-10:],
                            }

                        summary = {
                            "status": "session_ended",
                            "session_duration_seconds": round(
                                _time.time() - session_data["start_time"], 2
                            ) if session_data["start_time"] else 0,
                            "frames_analyzed": session_data["frames_analyzed"],
                            "average_emotions": avg_emotions,
                            "dominant_emotion": dominant,
                            "emotion_timeline": session_data["emotion_timeline"][-20:],
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
        if throttler.dropped > 0:
            logger.info("WebSocket /camera/session: dropped %d frames (throttled)", throttler.dropped)
        manager.disconnect(websocket, user_id=user_id)
    except Exception as e:
        logger.error("WebSocket session error: %s", e, exc_info=True)
        if throttler.dropped > 0:
            logger.info("WebSocket /camera/session: dropped %d frames (throttled)", throttler.dropped)
        manager.disconnect(websocket, user_id=user_id)


@router.websocket("/camera/conversation")
async def websocket_camera_conversation(websocket: WebSocket):
    """
    WebSocket endpoint za konverzacijski AI sa live kamerom.

    Auth: Client sends {"type": "auth", "token": "..."} as first message.
    Then:
    Client → Server:
    - {"type": "session_start"}
    - {"type": "frame", "frame": "<base64>"}
    - {"type": "text_message", "text": "...", "source": "keyboard|voice"}
    - {"type": "gesture_frames", "frames": ["<base64>", ...]}
    - {"type": "validation", "emotion": "...", "correct": bool}
    - {"type": "session_end"}

    Server → Client:
    - {"type": "emotion_result", ...}
    - {"type": "ai_message", "text": "...", ...}
    - {"type": "gesture_result", "gesture": "nod|shake|none", "confidence": 0.85}
    - {"type": "visual_observation", "observations": {...}, "timestamp": float}
    - {"type": "session_summary", ...}
    """
    if not await manager.connect(websocket):
        return

    # Wait for auth token as first message (not in URL query parameter)
    try:
        first_msg = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        first_data = json.loads(first_msg)
        token = None
        if isinstance(first_data, dict) and first_data.get("type") == "auth":
            token = first_data.get("token")
        ws_user = await verify_ws_token(token)
        if ws_user is None:
            await websocket.send_json({"type": "error", "message": "Authentication required"})
            await websocket.close(code=1008)
            manager.disconnect(websocket)
            return
    except (asyncio.TimeoutError, json.JSONDecodeError, Exception):
        await websocket.send_json({"type": "error", "message": "Authentication required"})
        await websocket.close(code=1008)
        manager.disconnect(websocket)
        return

    user_id = ws_user.get("uid", "anonymous")
    if not await manager.check_user_limit(websocket, user_id):
        manager.disconnect(websocket, user_id=None)
        return

    throttler = _FrameThrottler()
    session_id = str(uuid.uuid4())
    emotion_history = []
    max_history = 20
    frame_count = 0
    latest_emotions = None
    latest_masking = None
    latest_frame_b64 = None
    latest_visual_details = None

    try:
        while True:
            data = await websocket.receive_text()

            if len(data) > MAX_MESSAGE_BYTES:
                await manager.send_json(websocket, {
                    "type": "error",
                    "message": "Message too large",
                })
                continue

            try:
                message = json.loads(data)
                msg_type = message.get("type", "")

                if msg_type == "session_start":
                    greeting = conversation_engine.start_session(session_id)
                    await manager.send_json(websocket, {
                        "type": "ai_message",
                        **greeting,
                    })

                elif msg_type == "frame":
                    if "frame" not in message:
                        continue

                    # Validate frame size
                    err = _validate_frame(message["frame"])
                    if err:
                        await manager.send_json(websocket, {
                            "type": "error",
                            "message": err,
                        })
                        continue

                    # Throttle
                    if not throttler.allow():
                        continue

                    latest_frame_b64 = message["frame"]

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

                    await manager.send_json(websocket, {
                        "type": "emotion_result",
                        **result,
                    })

                    frame_count += 1

                    if frame_count % 10 == 7 and latest_emotions:
                        if not usage_tracker.can_call_gemini(ws_user["uid"]):
                            await manager.send_json(websocket, {
                                "type": "ai_message",
                                "text": "Dnevni limit je dostignut. Pokušaj ponovo sutra.",
                                "emotion_observation": None,
                                "suggested_actions": [],
                            })
                        else:
                            try:
                                ai_response = await asyncio.to_thread(
                                    conversation_engine.generate_response,
                                    session_id,
                                    latest_emotions,
                                    latest_masking,
                                    None,
                                    None,
                                    latest_visual_details,
                                )
                                usage_tracker.record_gemini_call(ws_user["uid"])
                                await manager.send_json(websocket, {
                                    "type": "ai_message",
                                    **ai_response,
                                })
                            except Exception as e:
                                logger.warning("Proactive AI comment error: %s", e)

                    if frame_count % 10 == 3 and latest_frame_b64:
                        if not usage_tracker.can_call_gemini(ws_user["uid"]):
                            logger.debug("[WS] Skipping vision analysis: Gemini limit reached for user %s", ws_user["uid"])
                        else:
                            try:
                                logger.debug("[WS] Triggering vision analysis (frame %d)", frame_count)
                                observations = await asyncio.to_thread(
                                    conversation_engine.analyze_visual_details,
                                    latest_frame_b64,
                                )
                                usage_tracker.record_gemini_call(ws_user["uid"])
                                if observations:
                                    latest_visual_details = observations
                                    await manager.send_json(websocket, {
                                        "type": "visual_observation",
                                        "observations": observations,
                                        "timestamp": _time.time(),
                                    })
                                    logger.debug("[WS] Vision sent: %s", list(observations.keys()))
                                else:
                                    logger.debug("[WS] Vision returned None")
                            except Exception as e:
                                logger.warning("Gemini vision analysis error: %s", e)

                elif msg_type == "text_message":
                    user_text = message.get("text", "").strip()
                    if len(user_text) > 2000:
                        await websocket.send_json({"type": "error", "message": "Message too long (max 2000 characters)"})
                        continue
                    if not user_text:
                        continue

                    if not usage_tracker.can_send_text_message(ws_user["uid"], session_id):
                        await manager.send_json(websocket, {
                            "type": "ai_message",
                            "text": "Limit poruka za ovu sesiju je dostignut.",
                            "emotion_observation": None,
                            "suggested_actions": [],
                        })
                        continue
                    usage_tracker.record_text_message(ws_user["uid"], session_id)

                    if not usage_tracker.can_call_gemini(ws_user["uid"]):
                        await manager.send_json(websocket, {
                            "type": "ai_message",
                            "text": "Dnevni limit je dostignut. Pokušaj ponovo sutra.",
                            "emotion_observation": None,
                            "suggested_actions": [],
                        })
                        continue

                    try:
                        ai_response = await asyncio.to_thread(
                            conversation_engine.generate_response,
                            session_id,
                            latest_emotions,
                            latest_masking,
                            user_text,
                            None,
                            latest_visual_details,
                        )
                        usage_tracker.record_gemini_call(ws_user["uid"])
                        await manager.send_json(websocket, {
                            "type": "ai_message",
                            **ai_response,
                        })
                    except Exception as e:
                        logger.warning("Text response error: %s", e)
                        await manager.send_json(websocket, {
                            "type": "ai_message",
                            "text": "Hvala ti. Nastavi kad budeš spreman/spremna.",
                            "emotion_observation": None,
                            "suggested_actions": [],
                        })

                elif msg_type == "validation":
                    emotion = message.get("emotion", "")
                    correct = message.get("correct", False)
                    logger.info("[Validation] emotion=%s, correct=%s, session=%s", emotion, correct, session_id)

                    if not correct and emotion:
                        if not usage_tracker.can_call_gemini(ws_user["uid"]):
                            await manager.send_json(websocket, {
                                "type": "ai_message",
                                "text": "Dnevni limit je dostignut. Pokušaj ponovo sutra.",
                                "emotion_observation": None,
                                "suggested_actions": [],
                            })
                            continue
                        try:
                            validation_context = f"[Korisnik kaze da NIJE {emotion}. Pitaj ga kako se zaista osjeca.]"
                            ai_response = await asyncio.to_thread(
                                conversation_engine.generate_response,
                                session_id,
                                latest_emotions,
                                latest_masking,
                                validation_context,
                                None,
                                latest_visual_details,
                            )
                            usage_tracker.record_gemini_call(ws_user["uid"])
                            await manager.send_json(websocket, {
                                "type": "ai_message",
                                **ai_response,
                            })
                        except Exception as e:
                            logger.warning("Validation response error: %s", e)

                elif msg_type == "gesture_frames":
                    frames = message.get("frames", [])
                    if not frames:
                        continue

                    # Cap gesture frame count
                    if len(frames) > MAX_GESTURE_FRAMES:
                        frames = frames[:MAX_GESTURE_FRAMES]

                    # Validate each gesture frame size
                    valid = True
                    for f in frames:
                        if _validate_frame(f) is not None:
                            await manager.send_json(websocket, {
                                "type": "error",
                                "message": "Gesture frame too large or invalid",
                            })
                            valid = False
                            break
                    if not valid:
                        continue

                    try:
                        gesture_result = await asyncio.to_thread(
                            head_gesture_detector.detect_gesture,
                            frames,
                        )
                        await manager.send_json(websocket, {
                            "type": "gesture_result",
                            **gesture_result,
                        })

                        if gesture_result["gesture"] != "none":
                            if not usage_tracker.can_call_gemini(ws_user["uid"]):
                                await manager.send_json(websocket, {
                                    "type": "ai_message",
                                    "text": "Dnevni limit je dostignut. Pokušaj ponovo sutra.",
                                    "emotion_observation": None,
                                    "suggested_actions": [],
                                })
                            else:
                                ai_response = await asyncio.to_thread(
                                    conversation_engine.generate_response,
                                    session_id,
                                    latest_emotions,
                                    latest_masking,
                                    None,
                                    gesture_result["gesture"],
                                    latest_visual_details,
                                )
                                usage_tracker.record_gemini_call(ws_user["uid"])
                                await manager.send_json(websocket, {
                                    "type": "ai_message",
                                    **ai_response,
                                })
                    except Exception as e:
                        logger.warning("Gesture detection error: %s", e)
                        await manager.send_json(websocket, {
                            "type": "gesture_result",
                            "gesture": "none",
                            "confidence": 0.0,
                        })

                elif msg_type == "session_end":
                    if not usage_tracker.can_call_gemini(ws_user["uid"]):
                        await manager.send_json(websocket, {
                            "type": "session_summary",
                            "summary": "Dnevni limit je dostignut. Pokušaj ponovo sutra.",
                            "message_count": frame_count,
                        })
                        conversation_engine.cleanup_session(session_id)
                        continue
                    try:
                        summary = await asyncio.to_thread(
                            conversation_engine.generate_summary,
                            session_id,
                        )
                        usage_tracker.record_gemini_call(ws_user["uid"])
                        await manager.send_json(websocket, {
                            "type": "session_summary",
                            **summary,
                        })
                    except Exception as e:
                        logger.error("Summary generation error: %s", e)
                        await manager.send_json(websocket, {
                            "type": "session_summary",
                            "summary": "Sesija završena.",
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
        if throttler.dropped > 0:
            logger.info("WebSocket /camera/conversation: dropped %d frames (throttled)", throttler.dropped)
        conversation_engine.cleanup_session(session_id)
        manager.disconnect(websocket, user_id=user_id)
    except Exception as e:
        logger.error("WebSocket conversation error: %s", e, exc_info=True)
        if throttler.dropped > 0:
            logger.info("WebSocket /camera/conversation: dropped %d frames (throttled)", throttler.dropped)
        conversation_engine.cleanup_session(session_id)
        manager.disconnect(websocket, user_id=user_id)
