"""
Firebase Authentication middleware for EmpaThink backend.

Provides:
- Firebase Admin SDK initialization
- FastAPI dependency for HTTP route authentication
- WebSocket token verification helper
"""

import os
import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Firebase Admin SDK initialization
# ---------------------------------------------------------------------------
_firebase_app = None
_firebase_init_error: Optional[str] = None

_security = HTTPBearer(auto_error=False)


def _init_firebase():
    """Initialize Firebase Admin SDK (once)."""
    global _firebase_app, _firebase_init_error
    if _firebase_app is not None:
        return

    try:
        import firebase_admin
        from firebase_admin import credentials

        # Option 1: GOOGLE_APPLICATION_CREDENTIALS env var (JSON key file path)
        # Option 2: Default credentials (Cloud Run provides these automatically)
        # Option 3: FIREBASE_PROJECT_ID for projectId-only init
        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        project_id = os.getenv("FIREBASE_PROJECT_ID")

        if cred_path:
            cred = credentials.Certificate(cred_path)
            _firebase_app = firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin initialized with service account key")
        elif project_id:
            _firebase_app = firebase_admin.initialize_app(
                options={"projectId": project_id}
            )
            logger.info("Firebase Admin initialized with project ID: %s", project_id)
        else:
            # Default credentials (works on Cloud Run, GCE, GKE)
            _firebase_app = firebase_admin.initialize_app()
            logger.info("Firebase Admin initialized with default credentials")

    except Exception as e:
        _firebase_init_error = str(e)
        logger.error("Firebase Admin initialization failed: %s", e)


# Initialize on import
_init_firebase()


# ---------------------------------------------------------------------------
# HTTP route dependency
# ---------------------------------------------------------------------------
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),
) -> dict:
    """
    FastAPI dependency that verifies Firebase ID tokens.

    Returns a dict with at minimum {"uid": "..."} on success.
    Raises 401 if token is missing or invalid.

    In development mode with ALLOW_ANONYMOUS_DEV=true, allows unauthenticated
    requests for easier local testing.
    """
    # Dev bypass: allow anonymous in development if explicitly opted in
    if (
        ENVIRONMENT != "production"
        and os.getenv("ALLOW_ANONYMOUS_DEV", "").lower() == "true"
    ):
        return {"uid": "dev-anonymous", "dev_bypass": True}

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    if _firebase_app is None:
        logger.warning(
            "Firebase Admin not initialized — allowing request without verification. Error: %s",
            _firebase_init_error,
        )
        return {"uid": "unverified", "firebase_unavailable": True}

    try:
        from firebase_admin import auth

        decoded = auth.verify_id_token(token, check_revoked=True)
        return {
            "uid": decoded["uid"],
            "email": decoded.get("email"),
            "email_verified": decoded.get("email_verified", False),
        }

    except firebase_admin.auth.RevokedIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
        )
    except firebase_admin.auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except firebase_admin.auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )
    except Exception as e:
        logger.error("Token verification failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
        )


# ---------------------------------------------------------------------------
# WebSocket token verification
# ---------------------------------------------------------------------------
async def verify_ws_token(token: Optional[str]) -> Optional[dict]:
    """
    Verify a Firebase ID token passed as a WebSocket query parameter.

    Returns user dict on success, None on failure.
    Does NOT raise — caller decides whether to reject the connection.
    """
    if not token:
        # Dev bypass
        if (
            ENVIRONMENT != "production"
            and os.getenv("ALLOW_ANONYMOUS_DEV", "").lower() == "true"
        ):
            return {"uid": "dev-anonymous", "dev_bypass": True}
        return None

    if _firebase_app is None:
        logger.error("Firebase Admin not initialized — cannot verify WS token")
        return None

    try:
        from firebase_admin import auth

        decoded = auth.verify_id_token(token, check_revoked=True)
        return {
            "uid": decoded["uid"],
            "email": decoded.get("email"),
        }
    except Exception as e:
        logger.warning("WebSocket token verification failed: %s", e)
        return None
