"""
EmpaThink AI Backend
Multimodal Emotion Analysis API

Author: Selma Skopljaković Hubljar
PhD Research: Trusted Empathic AI
"""

from dotenv import load_dotenv
load_dotenv()

import os
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from dependencies import limiter
from api.routes import text, voice, image, websocket_routes, multimodal, compliance
import uvicorn

# ---------------------------------------------------------------------------
# Environment & Logging
# ---------------------------------------------------------------------------
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("empathink")

# ---------------------------------------------------------------------------
# CORS origins
# ---------------------------------------------------------------------------
_PRODUCTION_ORIGINS = [
    "https://empathink-66fb2.web.app",
    "https://empathink-66fb2.firebaseapp.com",
]

_DEV_ORIGINS = [
    "http://localhost:8080",
    "http://localhost:3000",
    "http://127.0.0.1:8080",
]

# In production, only allow known origins. In dev, allow localhost too.
CORS_ORIGINS = (
    _PRODUCTION_ORIGINS
    if ENVIRONMENT == "production"
    else _PRODUCTION_ORIGINS + _DEV_ORIGINS
)

# Allow override via env var (comma-separated)
_extra = os.getenv("CORS_ORIGINS", "")
if _extra:
    CORS_ORIGINS += [o.strip() for o in _extra.split(",") if o.strip()]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="EmpaThink AI Backend",
    description="Multimodal Emotion Analysis API for PhD Research on Trusted Empathic AI",
    version="1.0.0",
    docs_url="/docs" if ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if ENVIRONMENT != "production" else None,
    openapi_url="/openapi.json" if ENVIRONMENT != "production" else None
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# Request body size limit middleware (must be registered before CORS so it
# wraps the CORS layer and runs first on every incoming request)
# ---------------------------------------------------------------------------
MAX_REQUEST_BODY_BYTES = 15 * 1024 * 1024  # 15 MB


@app.middleware("http")
async def limit_request_body(request: StarletteRequest, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body too large"},
                )
        except (ValueError, TypeError):
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid Content-Length header"},
            )
    return await call_next(request)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)


# ---------------------------------------------------------------------------
# Security headers middleware (runs AFTER CORS, BEFORE route handlers)
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if ENVIRONMENT == "production":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# Include routers
app.include_router(text.router, prefix="/analyze", tags=["Text Analysis"])
app.include_router(voice.router, prefix="/analyze", tags=["Voice Analysis"])
app.include_router(image.router, prefix="/analyze", tags=["Image Analysis"])
app.include_router(websocket_routes.router, prefix="/live", tags=["Live Camera"])
app.include_router(multimodal.router, prefix="/analyze", tags=["Multimodal Fusion"])
app.include_router(compliance.router, prefix="", tags=["Compliance & Legal"])

logger.info(
    "EmpaThink backend starting (env=%s, log_level=%s, origins=%d)",
    ENVIRONMENT, LOG_LEVEL, len(CORS_ORIGINS),
)


@app.get("/")
async def root():
    return {
        "service": "EmpaThink AI Backend",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
@limiter.limit("120/minute")
async def health_check(request: Request):
    from services.text_analyzer import _emotion_classifier, _sentiment_analyzer
    from services.face_analyzer import face_analyzer as _fa

    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "text_emotion": "ready" if _emotion_classifier is not None else "not_loaded",
            "text_sentiment": "ready" if _sentiment_analyzer is not None else "not_loaded",
            "face_analysis": "ready" if _fa._is_initialized else "not_loaded",
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=ENVIRONMENT != "production",
    )
