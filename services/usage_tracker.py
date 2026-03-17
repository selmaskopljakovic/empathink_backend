"""
Per-user API usage tracking and cost control.
Prevents abuse of the Gemini API by capping calls per user per day.
"""

import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
MAX_GEMINI_CALLS_PER_USER_PER_DAY = 100  # ~$0.50/day max per user at current pricing
MAX_GEMINI_CALLS_GLOBAL_PER_HOUR = 500   # Global safety net
MAX_TEXT_MESSAGES_PER_SESSION = 50        # Limit conversation length


class UsageTracker:
    """In-memory per-user usage tracking with daily reset."""

    def __init__(self):
        # {user_uid: {"gemini_calls": int, "reset_at": float, "text_messages": {session_id: int}}}
        self._users: Dict[str, dict] = {}
        self._global_gemini_calls = 0
        self._global_reset_at = time.time() + 3600

    def _get_or_create_user(self, uid: str) -> dict:
        now = time.time()
        if uid not in self._users:
            self._users[uid] = {
                "gemini_calls": 0,
                "reset_at": now + 86400,  # 24 hours
                "text_messages": {},
            }
        user = self._users[uid]
        # Daily reset
        if now >= user["reset_at"]:
            user["gemini_calls"] = 0
            user["reset_at"] = now + 86400
            user["text_messages"] = {}
        return user

    def can_call_gemini(self, uid: str) -> bool:
        """Check if user is allowed to make another Gemini API call."""
        now = time.time()

        # Global hourly reset
        if now >= self._global_reset_at:
            self._global_gemini_calls = 0
            self._global_reset_at = now + 3600

        if self._global_gemini_calls >= MAX_GEMINI_CALLS_GLOBAL_PER_HOUR:
            logger.warning("Global Gemini limit reached (%d/hr)", MAX_GEMINI_CALLS_GLOBAL_PER_HOUR)
            return False

        user = self._get_or_create_user(uid)
        if user["gemini_calls"] >= MAX_GEMINI_CALLS_PER_USER_PER_DAY:
            logger.warning("User %s hit daily Gemini limit (%d)", uid, MAX_GEMINI_CALLS_PER_USER_PER_DAY)
            return False

        return True

    def record_gemini_call(self, uid: str):
        """Record a Gemini API call for the user."""
        user = self._get_or_create_user(uid)
        user["gemini_calls"] += 1
        self._global_gemini_calls += 1

    def can_send_text_message(self, uid: str, session_id: str) -> bool:
        """Check if user can send another text message in this session."""
        user = self._get_or_create_user(uid)
        count = user["text_messages"].get(session_id, 0)
        return count < MAX_TEXT_MESSAGES_PER_SESSION

    def record_text_message(self, uid: str, session_id: str):
        """Record a text message for the user in a session."""
        user = self._get_or_create_user(uid)
        user["text_messages"][session_id] = user["text_messages"].get(session_id, 0) + 1

    def get_user_usage(self, uid: str) -> dict:
        """Get current usage stats for a user."""
        user = self._get_or_create_user(uid)
        return {
            "gemini_calls_today": user["gemini_calls"],
            "gemini_limit": MAX_GEMINI_CALLS_PER_USER_PER_DAY,
            "resets_at": user["reset_at"],
        }

    def cleanup_stale_users(self):
        """Remove users whose reset time has passed and have zero usage."""
        now = time.time()
        stale = [uid for uid, data in self._users.items() if now >= data["reset_at"] + 3600]
        for uid in stale:
            del self._users[uid]


# Singleton
usage_tracker = UsageTracker()
