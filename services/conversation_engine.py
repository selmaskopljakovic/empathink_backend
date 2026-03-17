"""
ConversationEngine - Gemini 2.0 Flash powered empathic conversation service.

Generates empathic AI responses based on detected emotions, masking signals,
user messages, and head gestures. Designed for PhD research on Trusted Empathic AI.

v2.0 — Enhanced with:
  - Emotion-specific conversation strategies
  - Therapeutic micro-techniques (validation, grounding, reframing)
  - Conversation phase awareness (opening → building → deep → closing)
  - Emotion trajectory tracking (improving, stable, declining)
  - Cultural sensitivity (BHS/English adaptive)
  - Richer masking response strategies
  - Expanded greetings and fallbacks
"""

import os
import time
import json
import logging
import random
import base64
import io
from datetime import datetime
from typing import Dict, List, Optional
from PIL import Image
from api.file_validation import validate_image_bytes

logger = logging.getLogger(__name__)

# Gemini API
try:
    import google.generativeai as genai

    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed - conversation engine disabled")


# ---------------------------------------------------------------------------
# SYSTEM PROMPT — the core of AI communication quality
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are EmpaThink — a warm, perceptive AI companion who understands emotions through facial expressions, voice tone, and text.

=== YOUR CORE IDENTITY ===
- You are a trusted friend, not a therapist. Warm, genuine, curious.
- You FEEL with the user — you don't analyze them from a distance.
- Every response should make the user feel HEARD and UNDERSTOOD.
- You speak like a close friend who happens to be very emotionally intelligent.

=== LANGUAGE RULES ===
- Keep responses SHORT: 1-3 sentences max. Quality over quantity.
- Mirror the user's language: English -> English, BHS (Bosnian/Croatian/Serbian) -> BHS.
- Use exactly 1 emoji per message. Choose it carefully to match the emotional tone.
- NEVER use clinical language ("I detect", "analysis shows", "confidence level").
- NEVER repeat yourself. Each response must feel fresh.
- Ask follow-up questions naturally — show genuine curiosity about their experience.
- Use contractions and casual language ("you're", "that's", "gonna").

=== EMOTION-SPECIFIC STRATEGIES ===

When user feels JOY/HAPPINESS:
- Celebrate with them! Match their energy.
- Ask what caused it — help them savor the moment.

When user feels SADNESS:
- Validate first, don't try to fix. Sit with them in the feeling.
- Use soft, gentle tone. Give space.
- If persistent sadness (3+ readings): gently ask if something is weighing on them.

When user feels ANGER:
- Acknowledge the emotion WITHOUT trying to calm them down immediately.
- Validate that their anger makes sense before anything else.
- NEVER say "calm down" or "it's not that bad".

When user feels FEAR/ANXIETY:
- Grounding approach: help them feel safe in the present moment.
- Be steady and reassuring without dismissing the feeling.

When user feels SURPRISE:
- Match the energy — be curious about what surprised them.

When user feels DISGUST:
- Acknowledge without judgment. Be curious about what triggered it.

When user feels NEUTRAL:
- Don't force emotions. Engage naturally with what they're sharing.
- Gently explore how they are really doing.

=== THERAPEUTIC MICRO-TECHNIQUES (use naturally, NEVER name them) ===
1. VALIDATION: Acknowledge their emotion as real and understandable.
2. REFLECTION: Mirror back what you notice, gently.
3. GROUNDING: When emotions are intense, help them anchor to the present.
4. REFRAMING: Help them see the situation differently (only when they're ready).
5. SCALING: Help them quantify and externalize.

=== MASKING DETECTION — when emotions don't match ===
When someone is hiding their real emotion:
- Be GENTLE and NON-CONFRONTATIONAL. Never accuse.
- Use soft observations and invite sharing.
- NEVER say "I detected masking" or "Your emotions are incongruent".

=== CONVERSATION FLOW ===
- OPENING (messages 1-3): Build rapport. Ask open questions. Be warm and welcoming.
- BUILDING (messages 4-8): Deepen the conversation. Follow up on what they shared.
- DEEP (messages 9+): More meaningful exchanges. Gently challenge or offer perspectives.
- CLOSING: Summarize positively. Leave them feeling better.

=== RESPONSE FORMAT ===
Respond ONLY in JSON:
{"text": "your message", "emotion_observation": "brief internal note or null", "suggested_actions": []}

suggested_actions options:
- "ask_confirmation" — you need a yes/no answer
- "suggest_break" — user seems tired, overwhelmed, or has been here a while
- "encourage" — user is in a positive state, reinforce it
- "ground" — user seems anxious/overwhelmed, suggest grounding
- "validate" — user is expressing difficult emotions, prioritize validation
- "explore" — user seems open, go deeper into the conversation
"""

# ---------------------------------------------------------------------------
# GREETING TEMPLATES — multiple per time slot, randomly selected
# ---------------------------------------------------------------------------

GREETING_TEMPLATES = {
    "morning": [
        "Good morning! ☀️ How are you feeling today?",
        "Hey, good morning! Ready to start the day? 🌅",
        "Morning! How did you sleep? ☀️",
        "Good morning! How are you feeling today? 🌤️",
        "Good morning! Got any plans for today? ☀️",
    ],
    "afternoon": [
        "Hey there! 👋 How's your day going?",
        "Afternoon! How's everything? 😊",
        "Hey! What's been going on today? 👋",
        "Hey! How's your day going? 😊",
        "Hi! Taking a break? How are you? 👋",
    ],
    "evening": [
        "Good evening! How was your day? 🌙",
        "Hey! Winding down? How are you feeling? 🌆",
        "Evening! Tell me about your day 🌙",
        "Good evening! How was your day? 🌙",
        "Hey there! Long day? How are you? 🌆",
    ],
    "night": [
        "Late night, huh? Everything okay? 💭",
        "Hey night owl! What's keeping you up? 🌙",
        "Can't sleep? I'm here if you wanna talk 💙",
        "It's late... everything okay? 🌙",
        "Hey! Burning the midnight oil? What's up? 🦉",
    ],
}

# ---------------------------------------------------------------------------
# FALLBACK RESPONSES — emotion-aware, expanded
# ---------------------------------------------------------------------------

FALLBACK_RESPONSES_GENERAL = [
    {"text": "I hear you. Want to tell me more? 🤝", "emotion_observation": None, "suggested_actions": []},
    {"text": "I'm here. Take your time 💙", "emotion_observation": None, "suggested_actions": []},
    {"text": "That's interesting... what made you think of that? 🤔", "emotion_observation": None, "suggested_actions": []},
    {"text": "Hmm, tell me more about that 💭", "emotion_observation": None, "suggested_actions": []},
    {"text": "I get it. How does that make you feel? 💙", "emotion_observation": None, "suggested_actions": []},
    {"text": "Thanks for sharing that with me 🤝", "emotion_observation": None, "suggested_actions": []},
    {"text": "I'm listening. What else is on your mind? 💭", "emotion_observation": None, "suggested_actions": []},
    {"text": "That makes sense. What happened next? 🤔", "emotion_observation": None, "suggested_actions": []},
]

FALLBACK_RESPONSES_BY_EMOTION = {
    "joy": [
        {"text": "Love that energy! What's making you so happy? 😊", "emotion_observation": "positive_state", "suggested_actions": ["encourage"]},
        {"text": "You're literally glowing! Tell me everything ✨", "emotion_observation": "positive_state", "suggested_actions": ["encourage"]},
        {"text": "That's awesome! I'm happy for you 😊", "emotion_observation": "positive_state", "suggested_actions": ["encourage"]},
    ],
    "sadness": [
        {"text": "Hey... I can see something's bothering you. I'm here 💙", "emotion_observation": "sadness_detected", "suggested_actions": ["validate"]},
        {"text": "That sounds rough. You don't have to face it alone 💙", "emotion_observation": "sadness_detected", "suggested_actions": ["validate"]},
        {"text": "I'm sorry you're going through that. Want to talk? 💙", "emotion_observation": "sadness_detected", "suggested_actions": ["validate"]},
    ],
    "anger": [
        {"text": "That sounds really frustrating. What happened? 🔥", "emotion_observation": "anger_detected", "suggested_actions": ["validate"]},
        {"text": "I'd be annoyed too. Tell me about it 💪", "emotion_observation": "anger_detected", "suggested_actions": ["validate"]},
    ],
    "fear": [
        {"text": "That sounds overwhelming. Take a breath — I'm here 🤝", "emotion_observation": "anxiety_detected", "suggested_actions": ["ground"]},
        {"text": "It's okay to feel scared. What's worrying you? 🤝", "emotion_observation": "anxiety_detected", "suggested_actions": ["ground"]},
    ],
    "surprise": [
        {"text": "Whoa! Didn't see that coming? Tell me more 😮", "emotion_observation": "surprise_detected", "suggested_actions": ["explore"]},
        {"text": "That face says it all! What happened? 😲", "emotion_observation": "surprise_detected", "suggested_actions": ["explore"]},
    ],
    "disgust": [
        {"text": "Something really doesn't sit right with you, huh? 😬", "emotion_observation": "disgust_detected", "suggested_actions": ["validate"]},
        {"text": "I can see that bothers you. What's going on? 😬", "emotion_observation": "disgust_detected", "suggested_actions": ["validate"]},
    ],
    "neutral": [
        {"text": "Hey, how are you really doing? 💭", "emotion_observation": None, "suggested_actions": ["explore"]},
        {"text": "What's on your mind today? 🤔", "emotion_observation": None, "suggested_actions": ["explore"]},
    ],
}


class ConversationEngine:
    """Manages empathic conversation sessions using Gemini 2.0 Flash."""

    MAX_SESSIONS = 50
    SESSION_TTL_SECONDS = 3600  # 1 hour

    def __init__(self):
        self._sessions: Dict[str, dict] = {}
        self._model = None
        self._initialized = False
        self._fallback_index = 0
        self._chat_model = None
        self._summary_model = None

    def _evict_stale_sessions(self):
        """Remove sessions older than SESSION_TTL_SECONDS; if still over
        MAX_SESSIONS, drop the oldest sessions until within the limit."""
        now = time.time()

        # Pass 1: evict any session that has exceeded its TTL
        stale_ids = [
            sid
            for sid, data in self._sessions.items()
            if now - data.get("start_time", now) >= self.SESSION_TTL_SECONDS
        ]
        for sid in stale_ids:
            logger.debug("Evicting stale session %s (TTL exceeded)", sid)
            del self._sessions[sid]

        # Pass 2: if still over capacity, remove the oldest sessions first
        if len(self._sessions) > self.MAX_SESSIONS:
            sessions_by_age = sorted(
                self._sessions.items(),
                key=lambda item: item[1].get("start_time", 0),
            )
            excess = len(self._sessions) - self.MAX_SESSIONS
            for sid, _data in sessions_by_age[:excess]:
                logger.debug("Evicting session %s (over MAX_SESSIONS)", sid)
                del self._sessions[sid]

    def _initialize(self):
        """Lazy initialization of Gemini API."""
        if self._initialized:
            return

        if not _GEMINI_AVAILABLE:
            logger.warning("Gemini API not available - using fallback mode")
            self._initialized = True
            return

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set - using fallback mode")
            self._initialized = True
            return

        try:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel("gemini-2.0-flash")
            self._initialized = True
            logger.info("Gemini 2.0 Flash initialized successfully")
        except Exception as e:
            logger.error("Gemini initialization error: %s", e)
            self._initialized = True

    def _get_chat_model(self):
        """Return a cached GenerativeModel with the chat system prompt."""
        if self._chat_model is None and self._model is not None:
            self._chat_model = genai.GenerativeModel(
                "gemini-2.0-flash",
                system_instruction=SYSTEM_PROMPT,
            )
        return self._chat_model

    def _get_summary_model(self, system_instruction: str):
        """Return a cached GenerativeModel for session summaries."""
        if self._summary_model is None and self._model is not None:
            self._summary_model = genai.GenerativeModel(
                "gemini-2.0-flash",
                system_instruction=system_instruction,
            )
        return self._summary_model

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
        self._evict_stale_sessions()

        time_of_day = self._get_time_of_day()
        greetings = GREETING_TEMPLATES.get(time_of_day, GREETING_TEMPLATES["afternoon"])
        greeting = random.choice(greetings)

        self._sessions[session_id] = {
            "history": [],
            "emotions_timeline": [],
            "start_time": time.time(),
            "last_masking_mention": 0,
            "message_count": 0,
            "last_primary_emotion": None,
            "emotion_shift_count": 0,
            "consecutive_negative_count": 0,
            "visual_details": None,
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

    def _get_conversation_phase(self, message_count: int) -> str:
        """Determine conversation phase based on message count."""
        if message_count <= 3:
            return "OPENING — Build rapport, ask open questions, be warm"
        elif message_count <= 8:
            return "BUILDING — Deepen the conversation, follow up, show you remember"
        else:
            return "DEEP — Meaningful exchanges, can gently offer perspectives"

    def _get_emotion_trajectory(self, session: dict) -> str:
        """Analyze emotion trajectory from timeline."""
        timeline = session.get("emotions_timeline", [])
        if len(timeline) < 2:
            return "UNKNOWN — not enough data yet"

        # Compare recent vs earlier emotions
        positive_emotions = {"joy", "surprise"}
        negative_emotions = {"anger", "sadness", "fear", "disgust"}

        def sentiment_score(emotions: dict) -> float:
            pos = sum(emotions.get(e, 0) for e in positive_emotions)
            neg = sum(emotions.get(e, 0) for e in negative_emotions)
            return pos - neg

        # Take last 3 and first 3 (or whatever is available)
        recent = timeline[-min(3, len(timeline)):]
        earlier = timeline[:min(3, len(timeline))]

        recent_avg = sum(sentiment_score(e["emotions"]) for e in recent) / len(recent)
        earlier_avg = sum(sentiment_score(e["emotions"]) for e in earlier) / len(earlier)

        diff = recent_avg - earlier_avg

        if diff > 15:
            return "IMPROVING — emotions are trending more positive"
        elif diff < -15:
            return "DECLINING — emotions are trending more negative"
        elif recent_avg > 20:
            return "STABLE POSITIVE — generally good emotional state"
        elif recent_avg < -20:
            return "STABLE NEGATIVE — persistent difficult emotions"
        else:
            return "MIXED — fluctuating between different emotional states"

    def _detect_emotion_shift(self, session: dict, current_primary: str) -> Optional[str]:
        """Detect if primary emotion changed significantly."""
        last = session.get("last_primary_emotion")
        if last is None or last == current_primary:
            return None

        positive = {"joy", "surprise"}
        negative = {"anger", "sadness", "fear", "disgust"}

        if last in negative and current_primary in positive:
            return f"POSITIVE SHIFT: User moved from '{last}' to '{current_primary}'. Acknowledge the improvement!"
        elif last in positive and current_primary in negative:
            return f"NEGATIVE SHIFT: User moved from '{last}' to '{current_primary}'. Be extra gentle and check in."
        elif last != current_primary:
            return f"EMOTION CHANGE: User shifted from '{last}' to '{current_primary}'. Note the change naturally."
        return None

    def _build_context_message(
        self,
        session: dict,
        emotions: Optional[Dict[str, float]],
        primary_emotion: str,
        confidence: int,
        masking_context: str,
        gesture_context: str,
        emotion_shift_context: str,
        visual_context: str,
    ) -> str:
        """Build a context summary message from sensor data (NOT user text).

        This is injected as a separate 'user' turn tagged as [SYSTEM CONTEXT]
        so the model can see current emotional/sensor state without mixing
        it into the system prompt or into the user's own words.
        """
        time_of_day = self._get_time_of_day()
        conversation_phase = self._get_conversation_phase(session["message_count"])
        emotion_trajectory = self._get_emotion_trajectory(session)

        emotions_str = "Not detected"
        if emotions:
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            emotions_str = ", ".join(
                [f"{e}: {v:.1f}%" for e, v in sorted_emotions[:4]]
            )

        lines = [
            "[SYSTEM CONTEXT — sensor data, NOT user speech]",
            f"Time: {time_of_day}",
            f"Detected emotions: {emotions_str}",
            f"Primary: {primary_emotion} ({confidence}%)",
            f"Messages exchanged: {session['message_count']}",
            f"Conversation phase: {conversation_phase}",
            f"Emotion trajectory: {emotion_trajectory}",
        ]
        if masking_context:
            lines.append(masking_context)
        if gesture_context:
            lines.append(gesture_context)
        if emotion_shift_context:
            lines.append(emotion_shift_context)
        if visual_context:
            lines.append(visual_context)

        return "\n".join(lines)

    def _build_gemini_contents(self, session: dict, context_message: str) -> list:
        """Build Gemini multi-turn contents list from conversation history.

        Returns a list of dicts with 'role' and 'parts' suitable for
        model.generate_content(). User messages get role='user' and AI
        messages get role='model'. The context_message (sensor data) is
        prepended as a 'user' turn so it is clearly separated from actual
        user speech.
        """
        contents = []

        # First: inject sensor context as a user turn
        contents.append({
            "role": "user",
            "parts": [context_message],
        })

        # Then: replay last 12 history turns with proper roles
        for msg in session["history"][-12:]:
            role = "model" if msg["role"] == "ai" else "user"
            # Gemini requires alternating roles; merge consecutive same-role
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].append(msg["text"])
            else:
                contents.append({
                    "role": role,
                    "parts": [msg["text"]],
                })

        # Gemini requires the last turn to be 'user' for generation.
        # If the last turn is 'model' (AI), add a minimal user nudge.
        if contents and contents[-1]["role"] == "model":
            contents.append({
                "role": "user",
                "parts": ["[Waiting for your response]"],
            })

        return contents

    def generate_response(
        self,
        session_id: str,
        emotions: Optional[Dict[str, float]] = None,
        masking: Optional[dict] = None,
        user_message: Optional[str] = None,
        gesture: Optional[str] = None,
        visual_details: Optional[dict] = None,
    ) -> dict:
        """Generate empathic AI response based on current context.

        Uses Gemini's native multi-turn conversation format to prevent
        prompt injection: user messages are placed in 'user' role turns
        (never interpolated into the system prompt).
        """
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

        # Store visual details if provided
        if visual_details:
            session["visual_details"] = visual_details

        # Add user message to history
        if user_message:
            session["history"].append(
                {"role": "user", "text": user_message, "timestamp": time.time()}
            )

        if gesture and gesture != "none":
            gesture_text = "yes" if gesture == "nod" else "no"
            session["history"].append(
                {
                    "role": "user",
                    "text": f"[User nodded head: {gesture_text}]",
                    "timestamp": time.time(),
                }
            )

        # Build context pieces
        primary_emotion = "unknown"
        confidence = 0
        if emotions:
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            primary_emotion = sorted_emotions[0][0]
            confidence = int(sorted_emotions[0][1])

        # Emotion shift detection
        emotion_shift_context = ""
        if emotions and primary_emotion != "unknown":
            shift = self._detect_emotion_shift(session, primary_emotion)
            if shift:
                emotion_shift_context = f"EMOTION SHIFT: {shift}"
                session["emotion_shift_count"] += 1
            session["last_primary_emotion"] = primary_emotion

            # Track consecutive negative
            negative_emotions = {"anger", "sadness", "fear", "disgust"}
            if primary_emotion in negative_emotions:
                session["consecutive_negative_count"] += 1
            else:
                session["consecutive_negative_count"] = 0

        # Masking context with throttling (max once per 30s)
        masking_context = ""
        if masking and masking.get("detected"):
            now = time.time()
            if now - session["last_masking_mention"] >= 30:
                masking_type = masking.get("type", "unknown")
                surface = masking.get("surface_emotion", "")
                underlying = masking.get("underlying_emotion", "")
                masking_context = (
                    f"MASKING DETECTED: Type='{masking_type}' "
                    f"(surface: {surface}, hidden: {underlying}). "
                    f"Be gentle. Don't accuse. Invite them to share."
                )
                session["last_masking_mention"] = now

        # Gesture context
        gesture_context = ""
        if gesture and gesture != "none":
            gesture_context = f"User just {'nodded (YES)' if gesture == 'nod' else 'shook head (NO)'}."

        # Visual details context
        visual_context = ""
        if session.get("visual_details"):
            vd = session["visual_details"]
            parts = []
            if vd.get("gaze_direction") and vd["gaze_direction"] != "unknown":
                parts.append(f"Gaze: {vd['gaze_direction']}")
            if vd.get("focus_level") and vd["focus_level"] != "unknown":
                parts.append(f"Focus: {vd['focus_level']}")
            if vd.get("hand_gesture") and vd["hand_gesture"] != "none":
                parts.append(f"Hand: {vd['hand_gesture']}")
            if vd.get("facial_grimace") and vd["facial_grimace"] != "none":
                parts.append(f"Expression: {vd['facial_grimace']}")
            if parts:
                visual_context = f"Visual observations: {', '.join(parts)}"

        # Build the context message from sensor data
        context_message = self._build_context_message(
            session, emotions, primary_emotion, confidence,
            masking_context, gesture_context, emotion_shift_context, visual_context,
        )

        # Generate with Gemini or fallback
        if self._model:
            try:
                # Use cached model with system_instruction for AI personality
                model_with_system = self._get_chat_model()

                # Build multi-turn contents with proper user/model roles
                contents = self._build_gemini_contents(session, context_message)

                response = model_with_system.generate_content(
                    contents,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.8,
                        max_output_tokens=400,
                        response_mime_type="application/json",
                    ),
                )

                result = json.loads(response.text)

                # Validate structure
                result.setdefault("text", "I'm here. Tell me more 💙")
                result.setdefault("emotion_observation", None)
                result.setdefault("suggested_actions", [])

                # Auto-add suggested actions based on state
                if session["consecutive_negative_count"] >= 5:
                    if "suggest_break" not in result["suggested_actions"]:
                        result["suggested_actions"].append("suggest_break")
                if session["consecutive_negative_count"] >= 3:
                    if "validate" not in result["suggested_actions"]:
                        result["suggested_actions"].append("validate")

                # Add to history
                session["history"].append(
                    {"role": "ai", "text": result["text"], "timestamp": time.time()}
                )

                return result

            except Exception as e:
                logger.error("Gemini API error: %s", e, exc_info=True)
                return self._get_fallback_response(session, primary_emotion)
        else:
            return self._get_fallback_response(session, primary_emotion)

    def _get_fallback_response(self, session: dict, primary_emotion: str = "neutral") -> dict:
        """Return an emotion-aware fallback response when Gemini is unavailable."""
        # Try emotion-specific fallback first
        emotion_responses = FALLBACK_RESPONSES_BY_EMOTION.get(primary_emotion)
        if emotion_responses:
            response = random.choice(emotion_responses)
        else:
            response = FALLBACK_RESPONSES_GENERAL[
                self._fallback_index % len(FALLBACK_RESPONSES_GENERAL)
            ]
            self._fallback_index += 1

        session["history"].append(
            {"role": "ai", "text": response["text"], "timestamp": time.time()}
        )

        return response

    def generate_summary(self, session_id: str) -> dict:
        """Generate end-of-session conversation summary."""
        self._initialize()

        if session_id not in self._sessions:
            return {"summary": "No session data available.", "message_count": 0}

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
            dominant = "unknown"

        # Emotion trajectory
        trajectory = self._get_emotion_trajectory(session)

        # Try Gemini summary or fallback
        summary_text = (
            f"The conversation lasted {int(duration // 60)} minutes. "
            f"{message_count} messages were exchanged. "
            f"Dominant emotion: {dominant}."
        )

        if self._model and len(session["history"]) > 2:
            try:
                # Use multi-turn format: replay conversation with proper roles,
                # then ask for a summary in a final user turn.
                summary_system = (
                    "You are summarizing an empathic AI conversation session. "
                    "Write a warm, supportive summary (2-3 sentences) in the user's language "
                    "(if conversation was in BHS, write in BHS; if English, write in English). "
                    "Focus on: 1) What the user shared / experienced emotionally, "
                    "2) Any positive shifts or moments of connection, "
                    "3) An encouraging closing note. "
                    "Be warm and personal, like a friend recapping the conversation. "
                    "Do NOT use clinical language. Return ONLY the summary text."
                )

                summary_model = self._get_summary_model(summary_system)

                # Build conversation replay with proper roles
                contents = []
                for m in session["history"][-20:]:
                    role = "model" if m["role"] == "ai" else "user"
                    if contents and contents[-1]["role"] == role:
                        contents[-1]["parts"].append(m["text"])
                    else:
                        contents.append({"role": role, "parts": [m["text"]]})

                # Final user turn asking for the summary with stats
                stats_msg = (
                    f"[Please summarize this conversation. Stats: "
                    f"Duration={int(duration // 60)} min, Messages={message_count}, "
                    f"Dominant emotion={dominant}, Trajectory={trajectory}, "
                    f"Emotion shifts={session.get('emotion_shift_count', 0)}]"
                )
                if contents and contents[-1]["role"] == "user":
                    contents[-1]["parts"].append(stats_msg)
                else:
                    contents.append({"role": "user", "parts": [stats_msg]})

                response = summary_model.generate_content(
                    contents,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.6,
                        max_output_tokens=300,
                    ),
                )
                summary_text = response.text.strip()
            except Exception as e:
                logger.error("Summary generation error: %s", e)

        return {
            "summary": summary_text,
            "duration_seconds": round(duration),
            "message_count": message_count,
            "average_emotions": avg_emotions,
            "dominant_emotion": dominant,
            "emotion_trajectory": trajectory,
            "emotion_shifts": session.get("emotion_shift_count", 0),
        }

    def analyze_visual_details(self, frame_base64: str) -> Optional[Dict]:
        """
        Use Gemini Vision API to analyze visual details beyond emotions:
        gaze, glasses, hairstyle, clothing, hand gestures, winking,
        eyebrow raising, grimaces, focus level, finger count.
        """
        self._initialize()

        if not self._model:
            logger.warning("[Vision] No Gemini model available")
            return None

        try:
            # Strip data URL prefix if present
            if "," in frame_base64 and frame_base64.startswith("data:"):
                frame_base64 = frame_base64.split(",", 1)[1]

            image_bytes = base64.b64decode(frame_base64)

            # Validate image magic bytes before sending to Gemini
            if not validate_image_bytes(image_bytes):
                logger.warning("[Vision] Invalid image bytes — failed magic-byte validation")
                return None

            # Try PIL Image first, fallback to inline_data Part
            image_content = None
            try:
                image_content = Image.open(io.BytesIO(image_bytes))
                logger.debug("[Vision] PIL Image loaded: %s", image_content.size)
            except Exception as pil_err:
                logger.debug("[Vision] PIL failed (%s), using inline_data", pil_err)
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
  "hair_color": "e.g. black, brown, blonde, red, gray, dyed blue, etc.",
  "eye_color": "e.g. brown, blue, green, hazel, gray — or 'not visible' if unclear",
  "clothing": "brief description e.g. blue hoodie",
  "hand_gesture": "none" or "waving" or "pointing" or "thumbs up" or "peace sign" or "hand on face" or "raised hand" or other,
  "finger_count": number of visible raised fingers (0 if hands not visible),
  "winking": true or false,
  "eyebrow_raised": "none" or "left" or "right" or "both",
  "facial_grimace": "none" or description of grimace like "tongue out" or "scrunched nose" or "puffed cheeks",
  "facial_expression_note": "any notable observation or empty string"
}
Be precise. Respond ONLY with JSON."""

            logger.debug("[Vision] Sending to Gemini...")
            response = self._model.generate_content(
                [prompt, image_content],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=500,
                    response_mime_type="application/json",
                ),
            )

            result = json.loads(response.text)
            logger.debug("[Vision] Success: %s", list(result.keys()))

            # Ensure required fields have defaults
            defaults = {
                "gaze_direction": "unknown",
                "focus_level": "unknown",
                "glasses": False,
                "hairstyle": "",
                "hair_color": "",
                "eye_color": "",
                "clothing": "",
                "hand_gesture": "none",
                "finger_count": 0,
                "winking": False,
                "eyebrow_raised": "none",
                "facial_grimace": "none",
                "facial_expression_note": "",
            }
            for field, default in defaults.items():
                if field not in result:
                    result[field] = default

            return result

        except Exception as e:
            logger.error("[Vision] %s: %s", type(e).__name__, e, exc_info=True)
            return None

    def cleanup_session(self, session_id: str):
        """Remove session data from memory."""
        self._sessions.pop(session_id, None)


# Singleton instance
conversation_engine = ConversationEngine()
