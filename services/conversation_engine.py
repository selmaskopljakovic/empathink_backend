"""
ConversationEngine - Gemini 2.0 Flash powered empathic conversation service.

Generates empathic AI responses based on detected emotions, masking signals,
user messages, and head gestures. Designed for PhD research on Trusted Empathic AI.
"""

import os
import time
import json
import base64
import io
from datetime import datetime
from typing import Dict, List, Optional
from PIL import Image

# Gemini API
try:
    import google.generativeai as genai

    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    print("google-generativeai not installed - conversation engine disabled")


SYSTEM_PROMPT_TEMPLATE = """You are EmpaThink — a warm, curious AI friend who reads emotions through facial expressions.

YOUR PERSONALITY:
- You're like a close friend who genuinely cares — casual, warm, real
- Keep it SHORT: 1-2 sentences max. Never lecture or over-explain
- Ask questions more than making statements
- Use 1 emoji per message, max. Don't overdo it
- Mirror the user's language: if they write in English, reply in English. If Bosnian/Croatian/Serbian, reply in that language
- NEVER diagnose, label mental health, or play therapist
- NEVER repeat the same phrase twice in a row
- Comment on emotions only when something CHANGES or is notable — don't narrate every frame

HOW TO TALK ABOUT EMOTIONS:
- Instead of: "I detect happiness at 85% confidence" → Say: "That smile says it all! 😊 What happened?"
- Instead of: "Analyzing your emotional patterns..." → Say: "Something on your mind? 💭"
- Instead of: "I notice sadness in your expression" → Say: "Hey, you look a bit down... wanna talk about it?"
- Instead of: "Your emotional state indicates..." → Say: "I can see something's going on..."
- Be natural. Observe like a friend would, not like a machine

MASKING (when someone hides their real emotion):
- Be gentle and non-confrontational
- Instead of: "Masking detected: fake smile" → Say: "That smile doesn't quite reach your eyes... everything okay? 💙"
- Never accuse. Invite them to share if they want to

CONTEXT:
- Time of day: {time_of_day}
- Detected emotions: {emotions}
- Primary: {primary_emotion} ({confidence}%)
{masking_context}
{gesture_context}

CONVERSATION SO FAR:
{conversation_history}

Respond in JSON:
{{"text": "your message", "emotion_observation": "brief internal note about emotions or null", "suggested_actions": []}}

suggested_actions options: "ask_confirmation" (need yes/no), "suggest_break" (user seems tired), "encourage" (positive emotions).
"""

GREETING_TEMPLATES = {
    "morning": "Good morning! ☀️ How are you feeling today?",
    "afternoon": "Hey there! 👋 How's your day going?",
    "evening": "Good evening! How was your day? 🌙",
    "night": "Late night, huh? Everything okay? 💭",
}

FALLBACK_RESPONSES = [
    {
        "text": "Thanks for sharing your emotions. How does it make you feel? 💭",
        "emotion_observation": None,
        "suggested_actions": [],
    },
    {
        "text": "I hear you. Want to tell me more? 🤝",
        "emotion_observation": None,
        "suggested_actions": [],
    },
    {
        "text": "I'm here for you. Take your time 💙",
        "emotion_observation": None,
        "suggested_actions": [],
    },
    {
        "text": "You are safe here. This space is just for you 🛡️",
        "emotion_observation": None,
        "suggested_actions": [],
    },
    {
        "text": "Your data is safe and will not be shared without your permission 🔒",
        "emotion_observation": None,
        "suggested_actions": [],
    },
    {
        "text": "This is only between you and me. Feel free to express yourself openly 💛",
        "emotion_observation": None,
        "suggested_actions": [],
    },
    {
        "text": "Everything you share here stays private and secure. I'm listening 🤗",
        "emotion_observation": None,
        "suggested_actions": [],
    },
    {
        "text": "There's no right or wrong way to feel. You're doing great by being here 🌟",
        "emotion_observation": None,
        "suggested_actions": [],
    },
]


class ConversationEngine:
    """Manages empathic conversation sessions using Gemini 2.0 Flash."""

    def __init__(self):
        self._sessions: Dict[str, dict] = {}
        self._model = None
        self._initialized = False
        self._fallback_index = 0

    def _initialize(self):
        """Lazy initialization of Gemini API."""
        if self._initialized:
            return

        if not _GEMINI_AVAILABLE:
            print("Gemini API not available - using fallback mode")
            self._initialized = True
            return

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("GEMINI_API_KEY not set - using fallback mode")
            self._initialized = True
            return

        try:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel("gemini-2.0-flash")
            self._initialized = True
            print("Gemini 2.0 Flash initialized successfully")
        except Exception as e:
            print(f"Gemini initialization error: {e}")
            self._initialized = True

    def _get_time_of_day(self) -> str:
        """Returns time-of-day category."""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    def start_session(self, session_id: str) -> dict:
        """Start a new conversation session with time-aware greeting."""
        self._initialize()

        time_of_day = self._get_time_of_day()
        greeting = GREETING_TEMPLATES[time_of_day]

        self._sessions[session_id] = {
            "history": [],
            "emotions_timeline": [],
            "start_time": time.time(),
            "last_masking_mention": 0,
            "message_count": 0,
        }

        # Add greeting to history
        self._sessions[session_id]["history"].append(
            {"role": "ai", "text": greeting, "timestamp": time.time()}
        )

        return {
            "text": greeting,
            "emotion_observation": None,
            "suggested_actions": [],
        }

    def generate_response(
        self,
        session_id: str,
        emotions: Optional[Dict[str, float]] = None,
        masking: Optional[dict] = None,
        user_message: Optional[str] = None,
        gesture: Optional[str] = None,
    ) -> dict:
        """Generate empathic AI response based on current context."""
        self._initialize()

        if session_id not in self._sessions:
            self.start_session(session_id)

        session = self._sessions[session_id]
        session["message_count"] += 1

        # Track emotions
        if emotions:
            session["emotions_timeline"].append(
                {"emotions": emotions, "timestamp": time.time()}
            )

        # Add user message to history
        if user_message:
            session["history"].append(
                {"role": "user", "text": user_message, "timestamp": time.time()}
            )

        if gesture and gesture != "none":
            gesture_text = "da" if gesture == "nod" else "ne"
            session["history"].append(
                {
                    "role": "user",
                    "text": f"[Korisnik klimnuo glavom: {gesture_text}]",
                    "timestamp": time.time(),
                }
            )

        # Build context
        time_of_day = self._get_time_of_day()

        emotions_str = "Nije detektovano"
        primary_emotion = "nepoznato"
        confidence = 0
        if emotions:
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            emotions_str = ", ".join(
                [f"{e}: {v:.1f}%" for e, v in sorted_emotions[:4]]
            )
            primary_emotion = sorted_emotions[0][0]
            confidence = int(sorted_emotions[0][1])

        # Masking context with throttling (max once per 30s)
        masking_context = ""
        if masking and masking.get("detected"):
            now = time.time()
            if now - session["last_masking_mention"] >= 30:
                masking_type = masking.get("type", "unknown")
                surface = masking.get("surface_emotion", "")
                underlying = masking.get("underlying_emotion", "")
                masking_context = (
                    f"- MASKING DETEKTOVAN: {masking_type} "
                    f"(površinska: {surface}, skrivena: {underlying}). "
                    f"Pažljivo i nježno adresirati."
                )
                session["last_masking_mention"] = now

        # Gesture context
        gesture_context = ""
        if gesture and gesture != "none":
            gesture_context = f"- Korisnik je upravo {'klimnuo glavom (DA)' if gesture == 'nod' else 'odmahnuo glavom (NE)'}."

        # Build conversation history string (last 10 messages)
        history_lines = []
        for msg in session["history"][-10:]:
            role = "AI" if msg["role"] == "ai" else "Korisnik"
            history_lines.append(f"{role}: {msg['text']}")
        conversation_history = "\n".join(history_lines) if history_lines else "Nema prethodnih poruka."

        # Generate with Gemini or fallback
        if self._model:
            try:
                prompt = SYSTEM_PROMPT_TEMPLATE.format(
                    time_of_day=time_of_day,
                    emotions=emotions_str,
                    primary_emotion=primary_emotion,
                    confidence=confidence,
                    masking_context=masking_context,
                    gesture_context=gesture_context,
                    conversation_history=conversation_history,
                )

                response = self._model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=300,
                        response_mime_type="application/json",
                    ),
                )

                result = json.loads(response.text)

                # Validate structure
                result.setdefault("text", "Razumijem. Nastavi...")
                result.setdefault("emotion_observation", None)
                result.setdefault("suggested_actions", [])

                # Add to history
                session["history"].append(
                    {"role": "ai", "text": result["text"], "timestamp": time.time()}
                )

                return result

            except Exception as e:
                print(f"Gemini API error: {e}")
                return self._get_fallback_response(session)
        else:
            return self._get_fallback_response(session)

    def _get_fallback_response(self, session: dict) -> dict:
        """Return a fallback response when Gemini is unavailable."""
        response = FALLBACK_RESPONSES[self._fallback_index % len(FALLBACK_RESPONSES)]
        self._fallback_index += 1

        session["history"].append(
            {"role": "ai", "text": response["text"], "timestamp": time.time()}
        )

        return response

    def generate_summary(self, session_id: str) -> dict:
        """Generate end-of-session conversation summary."""
        self._initialize()

        if session_id not in self._sessions:
            return {"summary": "Nema podataka o sesiji.", "message_count": 0}

        session = self._sessions[session_id]
        duration = time.time() - session["start_time"]
        message_count = session["message_count"]

        # Calculate dominant emotions across session
        all_emotions: Dict[str, float] = {}
        for entry in session["emotions_timeline"]:
            for emotion, value in entry["emotions"].items():
                all_emotions[emotion] = all_emotions.get(emotion, 0) + value

        if all_emotions:
            total = sum(all_emotions.values())
            avg_emotions = {e: round(v / total * 100, 1) for e, v in all_emotions.items()}
            dominant = max(avg_emotions, key=avg_emotions.get)
        else:
            avg_emotions = {}
            dominant = "nepoznato"

        # Try Gemini summary or fallback
        summary_text = (
            f"Razgovor je trajao {int(duration // 60)} minuta. "
            f"Razmijenjeno je {message_count} poruka. "
            f"Dominantna emocija: {dominant}."
        )

        if self._model and len(session["history"]) > 2:
            try:
                history_text = "\n".join(
                    [
                        f"{'AI' if m['role'] == 'ai' else 'Korisnik'}: {m['text']}"
                        for m in session["history"][-20:]
                    ]
                )
                prompt = (
                    f"Napravi kratak rezime ovog razgovora (2-3 rečenice, na BHS jeziku):\n\n"
                    f"{history_text}\n\n"
                    f"Dominantna emocija: {dominant}\n"
                    f"Odgovori samo tekst rezimea, bez JSON formata."
                )

                response = self._model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.5,
                        max_output_tokens=200,
                    ),
                )
                summary_text = response.text.strip()
            except Exception as e:
                print(f"Summary generation error: {e}")

        return {
            "summary": summary_text,
            "duration_seconds": round(duration),
            "message_count": message_count,
            "average_emotions": avg_emotions,
            "dominant_emotion": dominant,
        }

    def analyze_visual_details(self, frame_base64: str) -> Optional[Dict]:
        """
        Use Gemini Vision API to analyze visual details beyond emotions:
        gaze, glasses, hairstyle, clothing, hand gestures, winking, age,
        race, eyebrow raising, grimaces, focus level, finger count, lie detection.
        """
        self._initialize()

        if not self._model:
            print("[Vision] No Gemini model available")
            return None

        try:
            # Strip data URL prefix if present
            if "," in frame_base64 and frame_base64.startswith("data:"):
                frame_base64 = frame_base64.split(",", 1)[1]

            image_bytes = base64.b64decode(frame_base64)

            # Try PIL Image first, fallback to inline_data Part
            image_content = None
            try:
                image_content = Image.open(io.BytesIO(image_bytes))
                print(f"[Vision] PIL Image loaded: {image_content.size}")
            except Exception as pil_err:
                print(f"[Vision] PIL failed ({pil_err}), using inline_data")
                image_content = {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": frame_base64,
                    }
                }

            prompt = """Analyze this person carefully. Return ONLY valid JSON with ALL these fields:
{
  "gaze_direction": "looking at camera" or "looking left" or "looking right" or "looking up" or "looking down" or "eyes closed",
  "focus_level": "focused" or "distracted" or "drowsy" or "alert",
  "glasses": true or false,
  "hairstyle": "brief description e.g. long brown wavy hair",
  "clothing": "brief description e.g. blue hoodie",
  "hand_gesture": "none" or "waving" or "pointing" or "thumbs up" or "peace sign" or "hand on face" or "raised hand" or other,
  "finger_count": number of visible raised fingers (0 if hands not visible),
  "winking": true or false,
  "eyebrow_raised": "none" or "left" or "right" or "both",
  "facial_grimace": "none" or description of grimace like "tongue out" or "scrunched nose" or "puffed cheeks",
  "estimated_age_range": "e.g. 20-25",
  "ethnicity": "brief description",
  "deception_indicators": "none" or brief description of micro-expressions inconsistent with stated emotion,
  "facial_expression_note": "any notable observation or empty string"
}
Be precise. Respond ONLY with JSON."""

            print("[Vision] Sending to Gemini...")
            response = self._model.generate_content(
                [prompt, image_content],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=500,
                    response_mime_type="application/json",
                ),
            )

            result = json.loads(response.text)
            print(f"[Vision] Success: {list(result.keys())}")

            # Ensure required fields have defaults
            defaults = {
                "gaze_direction": "unknown",
                "focus_level": "unknown",
                "glasses": False,
                "hairstyle": "",
                "clothing": "",
                "hand_gesture": "none",
                "finger_count": 0,
                "winking": False,
                "eyebrow_raised": "none",
                "facial_grimace": "none",
                "estimated_age_range": "",
                "ethnicity": "",
                "deception_indicators": "none",
                "facial_expression_note": "",
            }
            for field, default in defaults.items():
                if field not in result:
                    result[field] = default

            return result

        except Exception as e:
            print(f"[Vision] ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def cleanup_session(self, session_id: str):
        """Remove session data from memory."""
        self._sessions.pop(session_id, None)


# Singleton instance
conversation_engine = ConversationEngine()
