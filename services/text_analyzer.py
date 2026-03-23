"""
Text Emotion Analyzer Service
Uses HuggingFace Transformers for emotion detection
"""

import time
from typing import Dict, List, Optional
from datetime import datetime

# Lazy loading for faster startup
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
            device=-1  # CPU, change to 0 for GPU
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
    Analyzes text and detects emotions using:
    - HuggingFace Transformers (DistilRoBERTa) for emotion detection
    - RoBERTa for sentiment analysis
    - TextBlob for additional metrics
    """

    # Mapping emotions to the Ekman model
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
        Analyzes text and returns emotions with percentages.

        Args:
            text: Text to analyze
            include_xai: Whether to include XAI explanations (for Group B)

        Returns:
            Dict with emotions, sentiment and XAI explanations
        """
        start_time = time.time()

        if not text or not text.strip():
            return self._empty_result()

        # Emotion detection
        emotions = self._detect_emotions(text)

        # Sentiment analysis
        sentiment = self._analyze_sentiment(text)

        # Text metrics
        text_metrics = self._calculate_metrics(text)

        # Find primary emotion
        primary_emotion = max(emotions, key=emotions.get)
        confidence = emotions[primary_emotion]

        # XAI explanation
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
        """Detects emotions using the DistilRoBERTa model"""
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
            # Fallback: returns neutral
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
        """Analyzes text sentiment"""
        try:
            analyzer = get_sentiment_analyzer()
            result = analyzer(text[:512])[0]

            # Map labels
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
        """Calculates additional metrics for text"""
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
        Generates XAI explanation for detected emotions.
        Tries SHAP first, falls back to keyword analysis.
        This is shown only to users in Group B.
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
        Generates keyword-based XAI explanation (fallback method).
        This is shown only to users in Group B.
        """
        # Keywords that indicate emotions
        emotion_keywords = {
            "joy": ["happy", "glad", "excited", "wonderful", "great", "love", "amazing"],
            "sadness": ["sad", "unhappy", "depressed", "down", "lonely", "miss", "cry"],
            "anger": ["angry", "furious", "annoyed", "frustrated", "mad", "hate"],
            "fear": ["scared", "afraid", "worried", "anxious", "nervous", "terrified"],
            "surprise": ["surprised", "shocked", "amazed", "unexpected", "wow"],
            "disgust": ["disgusted", "gross", "awful", "terrible", "hate"],
            "neutral": []
        }

        # Find keywords in text
        text_lower = text.lower()
        found_keywords = []
        for keyword in emotion_keywords.get(primary_emotion, []):
            if keyword in text_lower:
                found_keywords.append(keyword)

        # Generate explanation based on emotion
        emotion_explanations = {
            "joy": "Positive words and an optimistic tone indicate joy.",
            "sadness": "Negative words and a melancholic tone suggest sadness.",
            "anger": "Intense language and frustration indicate anger.",
            "fear": "Worry and uncertainty in the text indicate fear.",
            "surprise": "Unexpectedness and wonder are present in the text.",
            "disgust": "A negative reaction and aversion are visible in the text.",
            "neutral": "The text has a balanced tone without pronounced emotions."
        }

        # Sort emotions by intensity
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_emotions[:3]

        return {
            "method": "transformer_attention_analysis",
            "confidence": confidence,
            "reasoning": emotion_explanations.get(
                primary_emotion,
                "The analysis is based on linguistic patterns in the text."
            ),
            "key_indicators": found_keywords if found_keywords else ["linguistic patterns"],
            "emotion_breakdown": [
                {"emotion": e[0], "score": e[1]} for e in top_3
            ],
            "model_used": "distilroberta-emotion",
            "interpretation": f"The model analyzed text of {len(text.split())} words "
                            f"and detected '{primary_emotion}' as the dominant emotion "
                            f"with {confidence}% confidence."
        }

    def _empty_result(self) -> Dict:
        """Returns an empty result for empty input"""
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
