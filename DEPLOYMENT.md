# EmpaThink Backend - Deployment Guide

## Opcija 1: Google Cloud Run (Preporučeno)

### Preduvjeti
1. Google Cloud account (isti kao za Firebase)
2. Google Cloud CLI instaliran
3. Docker instaliran (za lokalno testiranje)

### Koraci

#### 1. Setup Google Cloud projekta
```bash
# Login
gcloud auth login

# Postavi projekat (isti kao Firebase)
gcloud config set project YOUR_PROJECT_ID

# Omogući potrebne API-je
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

#### 2. Deploy na Cloud Run
```bash
cd empathink_backend

# Opcija A: Automatski build i deploy
gcloud run deploy empathink-backend \
  --source . \
  --region europe-west1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2

# Opcija B: Koristi Cloud Build
gcloud builds submit --config cloudbuild.yaml
```

#### 3. Dobijete URL
Nakon deploy-a dobijete URL poput:
```
https://empathink-backend-XXXXX-ew.a.run.app
```

#### 4. Ažuriraj Flutter app
U `lib/services/emotion_api_service.dart` promijeni:
```dart
static const String _baseUrl = 'https://empathink-backend-XXXXX-ew.a.run.app';
static const String _wsUrl = 'wss://empathink-backend-XXXXX-ew.a.run.app';
```

---

## Opcija 2: Railway.app (Najlakše za početak)

### Koraci

1. Idi na https://railway.app
2. Registruj se (može sa GitHub-om)
3. New Project → Deploy from GitHub repo
4. Poveži GitHub i odaberi `empathink_backend` folder
5. Railway automatski detektuje Python i deploya

### URL
Railway daje URL poput:
```
https://empathink-backend.up.railway.app
```

---

## Opcija 3: Render.com

### Koraci

1. Idi na https://render.com
2. New → Web Service
3. Poveži GitHub repo
4. Postavi:
   - Environment: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

---

## Lokalno testiranje

### Sa Docker-om
```bash
cd empathink_backend

# Build
docker build -t empathink-backend .

# Run
docker run -p 8080:8080 empathink-backend

# Test
curl http://localhost:8080/health
```

### Bez Docker-a
```bash
cd empathink_backend

# Kreiraj virtual environment
python -m venv venv

# Aktiviraj (Windows)
venv\Scripts\activate

# Aktiviraj (Mac/Linux)
source venv/bin/activate

# Instaliraj dependencies
pip install -r requirements.txt

# Pokreni
python main.py

# Ili sa uvicorn
uvicorn main:app --reload --port 8000
```

---

## Environment Variables

Za produkciju, postavi ove varijable:

```bash
# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Firebase
FIREBASE_PROJECT_ID=your-project-id

# Optional
LOG_LEVEL=info
CORS_ORIGINS=https://your-flutter-app.web.app
```

---

## Testiranje API-ja

### Health Check
```bash
curl https://YOUR_URL/health
```

### Text Analysis
```bash
curl -X POST https://YOUR_URL/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling great today!", "include_xai": true}'
```

### Voice Analysis
```bash
curl -X POST https://YOUR_URL/analyze/voice \
  -F "audio=@test_audio.wav" \
  -F "include_xai=true"
```

### Image Analysis
```bash
curl -X POST https://YOUR_URL/analyze/image \
  -F "image=@test_face.jpg" \
  -F "include_xai=true"
```

---

## Monitoring

### Google Cloud Run
- Cloud Console → Cloud Run → empathink-backend
- Vidi logs, metrics, requests

### Logging
```bash
gcloud logs read --service=empathink-backend --limit=50
```

---

## Cijena (Procjena)

### Google Cloud Run
- Free tier: 2 million requests/month
- ~$0.00002400 per vCPU-second
- ~$0.00000250 per GiB-second

Za PhD research (low traffic): **~$5-20/mjesec**

### Railway
- Free tier: 500 hours/month
- Starter: $5/month

### Render
- Free tier: 750 hours/month
- Starter: $7/month
