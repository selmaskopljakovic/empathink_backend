"""
EmpaThink AI Backend
Multimodal Emotion Analysis API

Author: Selma Skopljaković Hubljar
PhD Research: Trusted Empathic AI
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import text, voice, image, websocket_routes, multimodal
import uvicorn

app = FastAPI(
    title="EmpaThink AI Backend",
    description="Multimodal Emotion Analysis API for PhD Research on Trusted Empathic AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - dozvoli Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:3000",
        "http://127.0.0.1:8080",
        "https://empathink-66fb2.web.app",
        "https://empathink-66fb2.firebaseapp.com",
        "*",  # Allow all origins for development/research
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(text.router, prefix="/analyze", tags=["Text Analysis"])
app.include_router(voice.router, prefix="/analyze", tags=["Voice Analysis"])
app.include_router(image.router, prefix="/analyze", tags=["Image Analysis"])
app.include_router(websocket_routes.router, prefix="/live", tags=["Live Camera"])
app.include_router(multimodal.router, prefix="/analyze", tags=["Multimodal Fusion"])


@app.get("/")
async def root():
    return {
        "message": "EmpaThink AI Backend",
        "version": "1.0.0",
        "endpoints": {
            "text": "/analyze/text",
            "voice": "/analyze/voice",
            "image": "/analyze/image",
            "multimodal": "/analyze/multimodal",
            "live_camera": "/live/camera (WebSocket)",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "text_analysis": "ready",
            "voice_analysis": "ready",
            "image_analysis": "ready",
            "multimodal_fusion": "ready",
            "live_camera": "ready"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Development mode
    )
