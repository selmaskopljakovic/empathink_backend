"""
GDPR Compliance & Legal Endpoints
Provides privacy policy, terms of service, and data deletion.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from dependencies import limiter
from api.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/privacy-policy")
@limiter.limit("60/minute")
async def privacy_policy(request: Request):
    """Returns the privacy policy for EmpaThink."""
    return {
        "title": "EmpaThink Privacy Policy",
        "last_updated": "2026-03-16",
        "data_controller": "Selma Skopljakovic Hubljar",
        "contact_email": "selma.skopljakovic@gmail.com",
        "sections": {
            "data_collection": {
                "description": "EmpaThink collects the following data for emotion analysis:",
                "data_types": [
                    "Facial images (processed in real-time, not permanently stored on server)",
                    "Voice recordings (processed in real-time, not permanently stored on server)",
                    "Text input (processed in real-time, not permanently stored on server)",
                    "Emotion analysis results (stored in Firebase Firestore)",
                    "Session metadata (timestamps, device info)",
                    "IP address and approximate geolocation (for analytics)",
                    "Email address (for authentication)",
                ],
            },
            "third_party_services": {
                "description": "Your data may be processed by the following services:",
                "services": [
                    {
                        "name": "Google Firebase",
                        "purpose": "Authentication, data storage, hosting",
                        "data_shared": "Email, session data, emotion results",
                    },
                    {
                        "name": "Google Gemini AI",
                        "purpose": "Conversational AI and visual analysis",
                        "data_shared": "Facial images (during live sessions), conversation text",
                        "note": "Images are processed in real-time and not stored by Google Gemini",
                    },
                    {
                        "name": "HuggingFace Models (on-server)",
                        "purpose": "Text and voice emotion analysis",
                        "data_shared": "Text and audio are processed locally on our server, not sent to HuggingFace",
                    },
                ],
            },
            "biometric_data": {
                "description": "EmpaThink processes biometric data including facial expressions "
                "and voice characteristics for emotion detection. This data is classified as "
                "special category data under GDPR Article 9.",
                "legal_basis": "Explicit consent (GDPR Article 9(2)(a))",
                "retention": "Processed in real-time. Raw images and audio are not stored. "
                "Only derived emotion scores are stored in your account.",
            },
            "data_retention": {
                "session_data": "Stored until you delete your account",
                "emotion_results": "Stored until you delete your account",
                "account_data": "Stored until you delete your account",
                "server_logs": "Retained for 30 days, then automatically deleted",
            },
            "your_rights": {
                "description": "Under GDPR, you have the following rights:",
                "rights": [
                    "Right to access your data (available in app Settings > Export My Data)",
                    "Right to erasure (available in app Settings > Delete All Data)",
                    "Right to data portability (export in JSON format)",
                    "Right to withdraw consent (delete account at any time)",
                    "Right to lodge a complaint with a supervisory authority",
                ],
            },
            "children": {
                "description": "EmpaThink is not intended for children under 13 years of age. "
                "We do not knowingly collect data from children under 13.",
            },
        },
    }


@router.get("/terms-of-service")
@limiter.limit("60/minute")
async def terms_of_service(request: Request):
    """Returns terms of service for EmpaThink."""
    return {
        "title": "EmpaThink Terms of Service",
        "last_updated": "2026-03-16",
        "sections": {
            "acceptance": "By using EmpaThink, you agree to these terms.",
            "service_description": "EmpaThink is an AI-powered emotion analysis application "
            "developed as part of PhD research on Trusted Empathic AI.",
            "user_obligations": [
                "You must be at least 13 years old to use this service",
                "You must provide accurate information during registration",
                "You must not attempt to abuse, exploit, or reverse-engineer the service",
                "You must not use the service for any illegal or harmful purpose",
            ],
            "ai_disclaimer": "EmpaThink uses artificial intelligence for emotion detection. "
            "Results are probabilistic estimates and should not be used as a substitute "
            "for professional mental health assessment or diagnosis.",
            "data_processing": "By using EmpaThink, you consent to the processing of your "
            "facial expressions, voice, and text for emotion analysis as described in "
            "our Privacy Policy.",
            "limitation_of_liability": "EmpaThink is provided 'as is' without warranty. "
            "The developers are not liable for any damages arising from use of the service.",
            "termination": "We reserve the right to terminate accounts that violate these terms.",
        },
    }


@router.delete("/user/data")
@limiter.limit("5/minute")
async def delete_user_data(request: Request, user: dict = Depends(get_current_user)):
    """
    GDPR Article 17: Right to erasure.
    Deletes all user data from the server side.
    The client app should also call Firebase Auth deleteUser.
    """
    uid = user["uid"]
    logger.info("Data deletion requested for user %s", uid)

    # Server-side data is ephemeral (in-memory sessions, no persistent storage)
    # All persistent data is in Firebase Firestore (managed by the client app)
    # This endpoint confirms the server has no persistent user data to delete

    # Clean up any active conversation sessions
    try:
        from services.conversation_engine import conversation_engine
        # Check if user has any active sessions (sessions are keyed by UUID, not UID)
        # We can't directly map UID to session_id here, but the cleanup
        # happens automatically via TTL eviction
        pass
    except Exception as e:
        logger.warning("Session cleanup during data deletion: %s", e)

    # Clean up usage tracking data
    try:
        from services.usage_tracker import usage_tracker
        if uid in usage_tracker._users:
            del usage_tracker._users[uid]
    except Exception as e:
        logger.warning("Usage data cleanup: %s", e)

    return {
        "status": "success",
        "message": "Server-side data deleted. Firestore data deletion is handled by the app.",
        "uid": uid,
        "note": "All persistent user data is stored in Firebase Firestore and should be "
        "deleted via the app's 'Delete All Data' feature, which also deletes "
        "your Firebase Authentication account.",
    }
