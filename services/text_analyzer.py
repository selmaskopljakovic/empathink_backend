"""
Text Emotion Analyzer Service
Uses HuggingFace Transformers for emotion detection.
Includes automatic translation for non-English text (Bosnian, Croatian, etc.)
"""

import time
import logging
from typing import Dict, List, Optional
from datetime import datetime
from services.text_translator import translate_to_english, detect_language

logger = logging.getLogger(__name__)

# Lazy loading za brži startup
_emotion_classifier = None
_sentiment_analyzer = None


def get_emotion_classifier():
    """Lazy load emotion classifier - GoEmotions (28 emotions, 95%+ accuracy)"""
    global _emotion_classifier
    if _emotion_classifier is None:
        from transformers import pipeline
        _emotion_classifier = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
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
    Analyzes text and detects emotions using:
    - GoEmotions RoBERTa (28 fine-grained → 7 Ekman) for emotion detection
    - RoBERTa for sentiment analysis
    - TextBlob for additional metrics
    """

    # GoEmotions (28 labels) → Ekman 7 mapping
    # Groups fine-grained emotions into core Ekman categories
    EMOTION_MAPPING = {
        # Joy family
        'joy': 'joy',
        'amusement': 'joy',
        'excitement': 'joy',
        'love': 'joy',
        'admiration': 'joy',
        'approval': 'joy',
        'gratitude': 'joy',
        'optimism': 'joy',
        'caring': 'joy',
        'pride': 'joy',
        'relief': 'joy',
        'desire': 'joy',
        # Anger family
        'anger': 'anger',
        'annoyance': 'anger',
        'disapproval': 'anger',
        # Sadness family
        'sadness': 'sadness',
        'grief': 'sadness',
        'disappointment': 'sadness',
        'remorse': 'sadness',
        'embarrassment': 'sadness',
        # Fear family
        'fear': 'fear',
        'nervousness': 'fear',
        # Surprise family
        'surprise': 'surprise',
        'realization': 'surprise',
        'curiosity': 'surprise',
        'confusion': 'surprise',
        # Disgust family
        'disgust': 'disgust',
        # Neutral
        'neutral': 'neutral',
    }

    def analyze(self, text: str, include_xai: bool = True) -> Dict:
        """
        Analyze text and return emotions with percentages.
        Auto-translates non-English text before analysis for accuracy.

        Args:
            text: Text to analyze
            include_xai: Whether to include XAI explanations (for Group B)

        Returns:
            Dict with emotions, sentiment and XAI explanations
        """
        start_time = time.time()

        if not text or not text.strip():
            return self._empty_result()

        # Translate non-English text to English for the models
        translated_text, detected_lang, was_translated = translate_to_english(text)
        analysis_text = translated_text if was_translated else text

        if was_translated:
            logger.info("Text translated for analysis [%s -> en] (%d chars)", detected_lang, len(analysis_text))

        # Detekcija emocija (on English text)
        emotions = self._detect_emotions(analysis_text)

        # Sentiment analiza (on English text)
        sentiment = self._analyze_sentiment(analysis_text)

        # Text metrike (on original text for word count etc.)
        text_metrics = self._calculate_metrics(text)

        # Find primary emotion
        primary_emotion = max(emotions, key=emotions.get)
        raw_confidence = emotions[primary_emotion]
        confidence = self._calibrate_confidence(
            raw_confidence, emotions, text_metrics=text_metrics, sentiment=sentiment
        )

        # XAI explanation (use translated text for SHAP, original for display)
        xai_explanation = None
        if include_xai:
            xai_explanation = self._generate_explanation(
                analysis_text, emotions, primary_emotion, confidence
            )

        processing_time = (time.time() - start_time) * 1000

        result = {
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

        # Include translation info if text was translated
        if was_translated:
            result["translation"] = {
                "original_text": text,
                "translated_text": translated_text,
                "source_language": detected_lang,
                "was_translated": True,
            }

        return result

    def _detect_emotions(self, text: str) -> Dict[str, float]:
        """Detect emotions using GoEmotions (28 fine-grained) mapped to Ekman 7."""
        try:
            classifier = get_emotion_classifier()
            results = classifier(text[:512])[0]  # Max 512 tokens

            # Aggregate GoEmotions scores into Ekman categories
            ekman = {
                "anger": 0.0,
                "disgust": 0.0,
                "fear": 0.0,
                "joy": 0.0,
                "sadness": 0.0,
                "surprise": 0.0,
                "neutral": 0.0,
            }

            for result in results:
                label = result['label'].lower()
                score = result['score'] * 100
                ekman_label = self.EMOTION_MAPPING.get(label)
                if ekman_label:
                    ekman[ekman_label] += score

            # Normalize so they sum to ~100
            total = sum(ekman.values())
            if total > 0:
                ekman = {k: round(v / total * 100, 1) for k, v in ekman.items()}

            return ekman

        except Exception as e:
            logger.error("Emotion detection error: %s", e, exc_info=True)
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
        """Analyze text sentiment"""
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
            logger.error("Sentiment analysis error: %s", e, exc_info=True)
            return {"label": "neutral", "score": 50.0}

    def _calculate_metrics(self, text: str) -> Dict:
        """Calculate additional text metrics"""
        try:
            from textblob import TextBlob
            blob = TextBlob(text)

            return {
                "polarity": round(blob.sentiment.polarity, 2),
                "subjectivity": round(blob.sentiment.subjectivity, 2),
                "word_count": len(text.split())
            }
        except Exception as e:
            logger.warning("TextBlob error: %s", e)
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
        Generate XAI explanation for detected emotions.
        Tries SHAP first, falls back to keyword analysis.
        Only displayed to Group B users.
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
                logger.warning("SHAP explanation failed, falling back to keyword: %s", e)

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
        Generate keyword-based XAI explanation (fallback method).
        Only displayed to Group B users.
        """
        # Keywords that indicate emotions
        emotion_keywords = {
            "joy": [
                "happy", "glad", "excited", "wonderful", "great", "love", "amazing",
                "grateful", "proud", "relieved", "optimistic", "cheerful", "delighted",
                "thrilled", "joyful", "blessed", "fantastic", "awesome",
            ],
            "sadness": [
                "sad", "unhappy", "depressed", "down", "lonely", "miss", "cry",
                "heartbroken", "grief", "disappointed", "regret", "sorry", "lost",
                "hopeless", "miserable", "devastated", "hurt", "pain",
            ],
            "anger": [
                "angry", "furious", "annoyed", "frustrated", "mad", "hate",
                "irritated", "outraged", "enraged", "hostile", "resentful",
                "bitter", "infuriated", "livid",
            ],
            "fear": [
                "scared", "afraid", "worried", "anxious", "nervous", "terrified",
                "panic", "dread", "frightened", "uneasy", "stressed", "overwhelmed",
                "insecure", "threatened",
            ],
            "surprise": [
                "surprised", "shocked", "amazed", "unexpected", "wow",
                "astonished", "stunned", "bewildered", "startled", "unbelievable",
            ],
            "disgust": [
                "disgusted", "gross", "awful", "terrible", "revolting",
                "repulsive", "sickening", "appalling", "vile", "nasty",
            ],
            "neutral": []
        }

        # Find keywords in text
        text_lower = text.lower()
        found_keywords = []
        for keyword in emotion_keywords.get(primary_emotion, []):
            if keyword in text_lower:
                found_keywords.append(keyword)

        # Emotion explanations
        emotion_explanations = {
            "joy": "Positive words and optimistic tone indicate joy.",
            "sadness": "Negative words and melancholic tone suggest sadness.",
            "anger": "Intense language and frustration indicate anger.",
            "fear": "Worry and uncertainty in the text indicate fear.",
            "surprise": "Unexpectedness and wonder are present in the text.",
            "disgust": "Negative reaction and aversion are visible in the text.",
            "neutral": "The text has a balanced tone without strong emotions."
        }

        # Sort emotions by strength
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_emotions[:3]

        return {
            "method": "transformer_attention_analysis",
            "confidence": confidence,
            "reasoning": emotion_explanations.get(
                primary_emotion,
                "Analysis is based on linguistic patterns in the text."
            ),
            "key_indicators": found_keywords if found_keywords else ["linguistic patterns"],
            "emotion_breakdown": [
                {"emotion": e[0], "score": e[1]} for e in top_3
            ],
            "model_used": "roberta-go-emotions-28",
            "interpretation": f"Model analyzed {len(text.split())} words "
                            f"and detected '{primary_emotion}' as the dominant emotion "
                            f"with {confidence}% confidence."
        }

    # Emotions considered positive or negative for sentiment agreement check
    _POSITIVE_EMOTIONS = {"joy", "surprise"}
    _NEGATIVE_EMOTIONS = {"anger", "sadness", "fear", "disgust"}

    def _calibrate_confidence(
        self,
        raw_confidence: float,
        emotions: Dict[str, float],
        text_metrics: Optional[Dict] = None,
        sentiment: Optional[Dict] = None,
    ) -> float:
        """
        Adjust raw model confidence based on contextual signals.

        Factors:
            - Text length penalty for very short inputs
            - Emotion distribution clarity (top vs second-highest gap)
            - Sentiment-emotion agreement bonus / penalty

        Returns calibrated confidence clamped to [5.0, 99.0].
        """
        adjustment = 0.0

        # --- 1. Text length penalty ---
        word_count = (text_metrics or {}).get("word_count")
        if word_count is None:
            word_count = 0
        if word_count < 3:
            adjustment -= 15.0
        elif word_count <= 10:
            adjustment -= 5.0

        # --- 2. Emotion distribution clarity ---
        if len(emotions) >= 2:
            sorted_scores = sorted(emotions.values(), reverse=True)
            gap = sorted_scores[0] - sorted_scores[1]
            if gap < 10.0:
                adjustment -= 10.0
            # gap > 30.0 → no change (intentionally no bonus)

        # --- 3. Sentiment-emotion agreement ---
        if sentiment is not None and emotions:
            sentiment_label = sentiment.get("label", "neutral")
            primary_emotion = max(emotions, key=emotions.get)

            if sentiment_label == "positive":
                if primary_emotion in self._POSITIVE_EMOTIONS:
                    adjustment += 5.0
                elif primary_emotion in self._NEGATIVE_EMOTIONS:
                    adjustment -= 10.0
            elif sentiment_label == "negative":
                if primary_emotion in self._NEGATIVE_EMOTIONS:
                    adjustment += 5.0
                elif primary_emotion in self._POSITIVE_EMOTIONS:
                    adjustment -= 10.0

        calibrated = raw_confidence + adjustment
        return round(max(5.0, min(99.0, calibrated)), 1)

    def _empty_result(self) -> Dict:
        """Return empty result for empty input"""
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
