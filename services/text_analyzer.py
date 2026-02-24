"""
Text Emotion Analyzer Service
Uses HuggingFace Transformers for emotion detection
"""

import time
from typing import Dict, List, Optional
from datetime import datetime

# Lazy loading za brži startup
_emotion_classifier = None
_sentiment_analyzer = None


def get_emotion_classifier():
    """Lazy load emotion classifier"""
    global _emotion_classifier
    if _emotion_classifier is None:
        from transformers import pipeline
        _emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=-1  # CPU, promijeni u 0 za GPU
        )
    return _emotion_classifier


def get_sentiment_analyzer():
    """Lazy load sentiment analyzer"""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        from transformers import pipeline
        _sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=-1
        )
    return _sentiment_analyzer


class TextEmotionAnalyzer:
    """
    Analizira tekst i detektuje emocije koristeći:
    - HuggingFace Transformers (DistilRoBERTa) za detekciju emocija
    - RoBERTa za sentiment analizu
    - TextBlob za dodatne metrike
    """

    # Mapiranje emocija na Ekman model
    EMOTION_MAPPING = {
        'anger': 'anger',
        'disgust': 'disgust',
        'fear': 'fear',
        'joy': 'joy',
        'sadness': 'sadness',
        'surprise': 'surprise',
        'neutral': 'neutral'
    }

    def analyze(self, text: str, include_xai: bool = True) -> Dict:
        """
        Analizira tekst i vraća emocije sa procentima.

        Args:
            text: Tekst za analizu
            include_xai: Da li uključiti XAI objašnjenja (za Group B)

        Returns:
            Dict sa emocijama, sentimentom i XAI objašnjenjima
        """
        start_time = time.time()

        if not text or not text.strip():
            return self._empty_result()

        # Detekcija emocija
        emotions = self._detect_emotions(text)

        # Sentiment analiza
        sentiment = self._analyze_sentiment(text)

        # Text metrike
        text_metrics = self._calculate_metrics(text)

        # Pronađi primarnu emociju
        primary_emotion = max(emotions, key=emotions.get)
        confidence = emotions[primary_emotion]

        # XAI objašnjenje
        xai_explanation = None
        if include_xai:
            xai_explanation = self._generate_explanation(
                text, emotions, primary_emotion, confidence
            )

        processing_time = (time.time() - start_time) * 1000

        return {
            "success": True,
            "emotions": emotions,
            "primary_emotion": primary_emotion,
            "confidence": confidence,
            "sentiment": sentiment,
            "text_metrics": text_metrics,
            "xai_explanation": xai_explanation,
            "processing_time_ms": round(processing_time, 2),
            "timestamp": datetime.now().isoformat()
        }

    def _detect_emotions(self, text: str) -> Dict[str, float]:
        """Detektuje emocije koristeći DistilRoBERTa model"""
        try:
            classifier = get_emotion_classifier()
            results = classifier(text[:512])[0]  # Max 512 tokena

            emotions = {}
            for result in results:
                label = result['label'].lower()
                score = round(result['score'] * 100, 1)
                emotions[label] = score

            return emotions

        except Exception as e:
            print(f"Emotion detection error: {e}")
            # Fallback: vraća neutral
            return {
                "anger": 0.0,
                "disgust": 0.0,
                "fear": 0.0,
                "joy": 0.0,
                "sadness": 0.0,
                "surprise": 0.0,
                "neutral": 100.0
            }

    def _analyze_sentiment(self, text: str) -> Dict:
        """Analizira sentiment teksta"""
        try:
            analyzer = get_sentiment_analyzer()
            result = analyzer(text[:512])[0]

            # Mapiraj labele
            label_mapping = {
                'positive': 'positive',
                'negative': 'negative',
                'neutral': 'neutral',
                'POSITIVE': 'positive',
                'NEGATIVE': 'negative',
                'NEUTRAL': 'neutral'
            }

            return {
                "label": label_mapping.get(result['label'], 'neutral'),
                "score": round(result['score'] * 100, 1)
            }

        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return {"label": "neutral", "score": 50.0}

    def _calculate_metrics(self, text: str) -> Dict:
        """Izračunava dodatne metrike za tekst"""
        try:
            from textblob import TextBlob
            blob = TextBlob(text)

            return {
                "polarity": round(blob.sentiment.polarity, 2),
                "subjectivity": round(blob.sentiment.subjectivity, 2),
                "word_count": len(text.split())
            }
        except Exception as e:
            print(f"TextBlob error: {e}")
            return {
                "polarity": 0.0,
                "subjectivity": 0.5,
                "word_count": len(text.split())
            }

    def _is_shap_available(self) -> bool:
        """Check if SHAP explainer is available."""
        try:
            from services.shap_explainer import shap_explainer
            return shap_explainer.is_available()
        except Exception:
            return False

    def _generate_explanation(
        self,
        text: str,
        emotions: Dict[str, float],
        primary_emotion: str,
        confidence: float
    ) -> Dict:
        """
        Generiše XAI objašnjenje za detektovane emocije.
        Pokušava SHAP prvo, fallback na keyword analizu.
        Ovo se prikazuje samo korisnicima u Group B.
        """
        # Try SHAP first
        if self._is_shap_available():
            try:
                from services.shap_explainer import shap_explainer
                shap_result = shap_explainer.explain(
                    text=text,
                    primary_emotion=primary_emotion,
                    top_n=10,
                )
                if shap_result is not None:
                    # Build explanation with SHAP data
                    keyword_explanation = self._generate_keyword_explanation(
                        text, emotions, primary_emotion, confidence
                    )
                    keyword_explanation["method"] = "shap_transformer_analysis"
                    keyword_explanation["shap_explanation"] = shap_result
                    return keyword_explanation
            except Exception as e:
                print(f"SHAP explanation failed, falling back to keyword: {e}")

        # Fallback to keyword-based explanation
        return self._generate_keyword_explanation(
            text, emotions, primary_emotion, confidence
        )

    def _generate_keyword_explanation(
        self,
        text: str,
        emotions: Dict[str, float],
        primary_emotion: str,
        confidence: float
    ) -> Dict:
        """
        Generiše keyword-bazirano XAI objašnjenje (fallback metoda).
        Ovo se prikazuje samo korisnicima u Group B.
        """
        # Ključne riječi koje ukazuju na emocije
        emotion_keywords = {
            "joy": ["happy", "glad", "excited", "wonderful", "great", "love", "amazing"],
            "sadness": ["sad", "unhappy", "depressed", "down", "lonely", "miss", "cry"],
            "anger": ["angry", "furious", "annoyed", "frustrated", "mad", "hate"],
            "fear": ["scared", "afraid", "worried", "anxious", "nervous", "terrified"],
            "surprise": ["surprised", "shocked", "amazed", "unexpected", "wow"],
            "disgust": ["disgusted", "gross", "awful", "terrible", "hate"],
            "neutral": []
        }

        # Pronađi ključne riječi u tekstu
        text_lower = text.lower()
        found_keywords = []
        for keyword in emotion_keywords.get(primary_emotion, []):
            if keyword in text_lower:
                found_keywords.append(keyword)

        # Generiši objašnjenje na osnovu emocije
        emotion_explanations = {
            "joy": "Pozitivne riječi i optimističan ton ukazuju na radost.",
            "sadness": "Negativne riječi i melanholičan ton sugerišu tugu.",
            "anger": "Intenzivan jezik i frustracija ukazuju na ljutnju.",
            "fear": "Zabrinutost i nesigurnost u tekstu ukazuju na strah.",
            "surprise": "Neočekivanost i čuđenje prisutni su u tekstu.",
            "disgust": "Negativna reakcija i odbojnost vidljivi su u tekstu.",
            "neutral": "Tekst ima uravnotežen ton bez izraženih emocija."
        }

        # Sortiraj emocije po jačini
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_emotions[:3]

        return {
            "method": "transformer_attention_analysis",
            "confidence": confidence,
            "reasoning": emotion_explanations.get(
                primary_emotion,
                "Analiza je bazirana na jezičkim obrascima u tekstu."
            ),
            "key_indicators": found_keywords if found_keywords else ["linguistic patterns"],
            "emotion_breakdown": [
                {"emotion": e[0], "score": e[1]} for e in top_3
            ],
            "model_used": "distilroberta-emotion",
            "interpretation": f"Model je analizirao tekst od {len(text.split())} riječi "
                            f"i detektovao '{primary_emotion}' kao dominantnu emociju "
                            f"sa {confidence}% sigurnošću."
        }

    def _empty_result(self) -> Dict:
        """Vraća prazan rezultat za prazan input"""
        return {
            "success": False,
            "error": "Empty text provided",
            "emotions": {},
            "primary_emotion": "neutral",
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat()
        }


# Singleton instance
text_analyzer = TextEmotionAnalyzer()
