"""
Unit tests for SHAP Text Explainer service.
Tests explanation generation, caching, truncation, and error handling.
Run with: pytest tests/test_shap_explainer.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
from services.shap_explainer import ShapTextExplainer, _truncate_text, _get_cache_key


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestUtilityFunctions:

    def test_truncate_text_short_text_unchanged(self):
        text = "hello world"
        assert _truncate_text(text, max_words=100) == text

    def test_truncate_text_exact_limit(self):
        text = " ".join(["word"] * 100)
        assert _truncate_text(text, max_words=100) == text

    def test_truncate_text_over_limit(self):
        text = " ".join(["word"] * 150)
        result = _truncate_text(text, max_words=100)
        assert len(result.split()) == 100

    def test_truncate_empty_text(self):
        assert _truncate_text("") == ""

    def test_cache_key_strips_and_lowercases(self):
        assert _get_cache_key("  Hello World  ") == "hello world"

    def test_cache_key_same_for_equivalent_texts(self):
        assert _get_cache_key("HELLO") == _get_cache_key("hello")
        assert _get_cache_key("  test  ") == _get_cache_key("test")


# ---------------------------------------------------------------------------
# ShapTextExplainer tests
# ---------------------------------------------------------------------------

class TestShapExplainer:

    def _make_explainer(self):
        return ShapTextExplainer()

    def test_is_available_returns_bool(self):
        explainer = self._make_explainer()
        result = explainer.is_available()
        assert isinstance(result, bool)

    def test_explain_empty_text_returns_none(self):
        explainer = self._make_explainer()
        result = explainer.explain("", "joy")
        assert result is None

    def test_explain_whitespace_returns_none(self):
        explainer = self._make_explainer()
        result = explainer.explain("   ", "joy")
        assert result is None

    def test_emotion_index_known_emotions(self):
        explainer = self._make_explainer()
        assert explainer._get_emotion_index("joy") == 3
        assert explainer._get_emotion_index("anger") == 0
        assert explainer._get_emotion_index("neutral") == 4
        assert explainer._get_emotion_index("sadness") == 5
        assert explainer._get_emotion_index("surprise") == 6
        assert explainer._get_emotion_index("fear") == 2
        assert explainer._get_emotion_index("disgust") == 1

    def test_emotion_index_unknown_defaults_to_neutral(self):
        explainer = self._make_explainer()
        assert explainer._get_emotion_index("happiness") == 4  # neutral index

    @patch("services.shap_explainer._get_shap_components")
    def test_explain_returns_expected_structure(self, mock_components):
        """Mocked SHAP should return properly structured result."""
        import numpy as np

        # Mock SHAP values
        mock_shap_values = MagicMock()
        mock_shap_values.values = np.array([[[0.5, 0.1, 0.0, 0.8, 0.1, -0.2, 0.0],
                                              [0.1, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
                                              [0.0, 0.0, 0.0, 0.1, 0.2, 0.0, 0.0]]])
        mock_shap_values.data = np.array([["I", "am", "happy"]])

        mock_explainer = MagicMock(return_value=mock_shap_values)
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_components.return_value = (mock_explainer, mock_tokenizer, mock_model)

        explainer = self._make_explainer()
        # Clear cache to force computation
        import services.shap_explainer as mod
        mod._explanation_cache.clear()

        result = explainer.explain("I am happy", "joy", top_n=3)

        assert result is not None
        assert result["method"] == "shap_partition"
        assert result["target_emotion"] == "joy"
        assert "word_importance" in result
        assert len(result["word_importance"]) <= 3

        # Check word importance structure
        for wi in result["word_importance"]:
            assert "word" in wi
            assert "contribution" in wi
            assert "direction" in wi
            assert wi["direction"] in ("positive", "negative")
            assert "rank" in wi

    @patch("services.shap_explainer._get_shap_components")
    def test_explain_caches_result(self, mock_components):
        """Second call with same text should use cache."""
        import numpy as np
        import services.shap_explainer as mod

        mock_shap_values = MagicMock()
        mock_shap_values.values = np.array([[[0.5, 0.1, 0.0, 0.8, 0.1, -0.2, 0.0]]])
        mock_shap_values.data = np.array([["happy"]])

        mock_explainer = MagicMock(return_value=mock_shap_values)
        mock_components.return_value = (mock_explainer, MagicMock(), MagicMock())

        mod._explanation_cache.clear()
        explainer = self._make_explainer()

        # First call
        result1 = explainer.explain("happy text", "joy")
        # Second call (should hit cache)
        result2 = explainer.explain("happy text", "joy")

        assert result1 == result2
        # Explainer should only be called once
        assert mock_explainer.call_count == 1

    @patch("services.shap_explainer._get_shap_components")
    def test_explain_handles_exception(self, mock_components):
        """If SHAP crashes, should return None."""
        mock_components.side_effect = RuntimeError("SHAP init failed")

        import services.shap_explainer as mod
        mod._explanation_cache.clear()

        explainer = self._make_explainer()
        result = explainer.explain("test text", "joy")
        assert result is None

    def test_extract_word_importance_filters_empty_tokens(self):
        """Empty/whitespace tokens should be filtered out."""
        import numpy as np

        mock_shap_values = MagicMock()
        mock_shap_values.values = np.array([[[0.5, 0.1, 0.0, 0.8, 0.1, -0.2, 0.0],
                                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              [0.1, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0]]])
        mock_shap_values.data = np.array([["happy", " ", "day"]])

        explainer = self._make_explainer()
        result = explainer._extract_word_importance(mock_shap_values, emotion_idx=3, top_n=10)

        words = [w["word"] for w in result]
        assert " " not in words
        assert "" not in words

    def test_word_importance_sorted_by_absolute_value(self):
        """Words should be sorted by absolute SHAP value."""
        import numpy as np

        mock_shap_values = MagicMock()
        mock_shap_values.values = np.array([[[0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
                                              [0.0, 0.0, 0.0, -0.8, 0.0, 0.0, 0.0],
                                              [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]]])
        mock_shap_values.data = np.array([["word1", "word2", "word3"]])

        explainer = self._make_explainer()
        result = explainer._extract_word_importance(mock_shap_values, emotion_idx=3, top_n=3)

        # word2 has highest absolute value (-0.8), then word3 (0.5), then word1 (0.2)
        assert result[0]["word"] == "word2"
        assert result[0]["direction"] == "negative"
        assert result[1]["word"] == "word3"
        assert result[2]["word"] == "word1"


# ---------------------------------------------------------------------------
# Cache eviction tests
# ---------------------------------------------------------------------------

class TestCacheEviction:

    @patch("services.shap_explainer._get_shap_components")
    def test_cache_evicts_oldest_when_full(self, mock_components):
        """Cache should evict oldest entry when exceeding max size."""
        import numpy as np
        import services.shap_explainer as mod

        mock_shap_values = MagicMock()
        mock_shap_values.values = np.array([[[0.5, 0.1, 0.0, 0.8, 0.1, -0.2, 0.0]]])
        mock_shap_values.data = np.array([["word"]])

        mock_explainer = MagicMock(return_value=mock_shap_values)
        mock_components.return_value = (mock_explainer, MagicMock(), MagicMock())

        mod._explanation_cache.clear()
        explainer = ShapTextExplainer()

        # Fill cache beyond max
        for i in range(mod._CACHE_MAX_SIZE + 5):
            explainer.explain(f"text number {i}", "joy")

        assert len(mod._explanation_cache) <= mod._CACHE_MAX_SIZE
