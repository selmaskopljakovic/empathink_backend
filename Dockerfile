# EmpaThink AI Backend - Docker Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Pre-download HuggingFace models during build (so container is self-contained)
RUN python -c "\
from transformers import pipeline, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor; \
pipeline('text-classification', model='SamLowe/roberta-base-go_emotions', top_k=None, device=-1); \
pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest', device=-1); \
Wav2Vec2FeatureExtractor.from_pretrained('superb/wav2vec2-large-superb-er'); \
Wav2Vec2ForSequenceClassification.from_pretrained('superb/wav2vec2-large-superb-er'); \
print('All models downloaded successfully')"

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
