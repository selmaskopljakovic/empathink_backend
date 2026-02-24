"""
SHAP Text Explainer Service
Provides word-level SHAP explanations for text emotion classification.
Uses the same DistilRoBERTa model as the main text analyzer.
"""

import logging
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Lazy-loaded globals
_shap_explainer = None
_tokenizer = None
_model = None

# In-memory LRU cache (max 50 entries)
_explanation_cache: OrderedDict = OrderedDict()
_CACHE_MAX_SIZE = 50

# Max words for SHAP (performance)
_SHAP_MAX_WORDS = 100


def _get_shap_components():
    """Lazy load the SHAP explainer and model components."""
    global _shap_explainer, _tokenizer, _model

    if _shap_explainer is not None:
        return _shap_explainer, _tokenizer, _model

    try:
        import shap
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        import torch

        model_name = "j-hartmann/emotion-english-distilroberta-base"

        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Create pipeline for SHAP
        emotion_pipeline = pipeline(
            "text-classification",
            model=_model,
            tokenizer=_tokenizer,
            top_k=None,
            device=-1,
        )

        # Use partition masker for text
        masker = shap.maskers.Text(_tokenizer)
        _shap_explainer = shap.Explainer(emotion_pipeline, masker)

        logger.info("SHAP explainer initialized successfully")
        return _shap_explainer, _tokenizer, _model

    except Exception as e:
        logger.error(f"Failed to initialize SHAP explainer: {e}")
        raise


def _truncate_text(text: str, max_words: int = _SHAP_MAX_WORDS) -> str:
    """Truncate text to max_words for SHAP performance."""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text


def _get_cache_key(text: str) -> str:
    """Generate a cache key from the text."""
    return text.strip().lower()


class ShapTextExplainer:
    """
    Generates word-level SHAP explanations for text emotion classification.

    Uses shap.Explainer with partition masker on the same
    j-hartmann/emotion-english-distilroberta-base model.

    Features:
    - Lazy loading of SHAP explainer (heavy on first call)
    - In-memory LRU cache (50 entries)
    - Text truncation to 100 words for performance
    - Returns top 10 words with positive/negative contributions
    """

    # Emotion labels from the model (index order)
    EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

    def is_available(self) -> bool:
        """Check if SHAP is available (can be imported)."""
        try:
            import shap
            return True
        except ImportError:
            return False

    def explain(
        self,
        text: str,
        primary_emotion: str,
        top_n: int = 10,
    ) -> Optional[Dict]:
        """
        Generate SHAP explanation for the given text.

        Args:
            text: Input text to explain
            primary_emotion: The detected primary emotion
            top_n: Number of top contributing words to return

        Returns:
            Dict with word_importance list and metadata, or None on failure
        """
        if not text or not text.strip():
            return None

        # Check cache first
        cache_key = _get_cache_key(text)
        if cache_key in _explanation_cache:
            _explanation_cache.move_to_end(cache_key)
            return _explanation_cache[cache_key]

        try:
            truncated_text = _truncate_text(text)
            explainer, tokenizer, model = _get_shap_components()

            # Run SHAP
            shap_values = explainer([truncated_text])

            # Find the index for the primary emotion
            emotion_idx = self._get_emotion_index(primary_emotion)

            # Extract word-level SHAP values for the primary emotion
            word_importance = self._extract_word_importance(
                shap_values, emotion_idx, top_n
            )

            result = {
                "method": "shap_partition",
                "model": "distilroberta-emotion",
                "target_emotion": primary_emotion,
                "word_importance": word_importance,
                "truncated": len(text.split()) > _SHAP_MAX_WORDS,
                "num_words_analyzed": len(truncated_text.split()),
            }

            # Cache the result
            _explanation_cache[cache_key] = result
            if len(_explanation_cache) > _CACHE_MAX_SIZE:
                _explanation_cache.popitem(last=False)

            return result

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return None

    def _get_emotion_index(self, emotion: str) -> int:
        """Get the index of the emotion in the model's output."""
        emotion_lower = emotion.lower()
        try:
            return self.EMOTION_LABELS.index(emotion_lower)
        except ValueError:
            # Default to neutral if emotion not found
            return self.EMOTION_LABELS.index("neutral")

    def _extract_word_importance(
        self,
        shap_values,
        emotion_idx: int,
        top_n: int,
    ) -> List[Dict]:
        """
        Extract top-N most important words from SHAP values.

        Returns list of dicts with word, contribution (+/-), and rank.
        """
        import numpy as np

        # shap_values.values shape: (1, num_tokens, num_classes)
        values = shap_values.values[0]  # First (only) sample
        data = shap_values.data[0]  # Token strings

        # Get SHAP values for the target emotion
        emotion_values = values[:, emotion_idx]

        # Pair tokens with their SHAP values
        word_shap_pairs: List[Tuple[str, float]] = []
        for token, shap_val in zip(data, emotion_values):
            token_str = str(token).strip()
            if not token_str or token_str in ["", " "]:
                continue
            word_shap_pairs.append((token_str, float(shap_val)))

        # Sort by absolute value (most impactful first)
        word_shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        # Take top N
        top_words = word_shap_pairs[:top_n]

        result = []
        for rank, (word, contribution) in enumerate(top_words, 1):
            result.append({
                "word": word,
                "contribution": round(contribution, 4),
                "direction": "positive" if contribution > 0 else "negative",
                "rank": rank,
            })

        return result


# Singleton instance
shap_explainer = ShapTextExplainer()
