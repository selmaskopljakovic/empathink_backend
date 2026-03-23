"""
Unit tests for TextEmotionAnalyzer service.
Tests emotion detection, sentiment analysis, XAI explanations, and edge cases.
Run with: pytest tests/test_text_analyzer.py -v
"""

import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# TextEmotionAnalyzer unit tests
# ---------------------------------------------------------------------------

class TestTextEmotionAnalyzer:
    """Tests for the TextEmotionAnalyzer class."""

    def _make_analyzer(self):
        from services.text_analyzer import TextEmotionAnalyzer
        return TextEmotionAnalyzer()

    # --- Basic analyze() contract ---

    @patch("services.text_analyzer.get_emotion_classifier")
    @patch("services.text_analyzer.get_sentiment_analyzer")
    def test_analyze_returns_required_fields(self, mock_sent, mock_emo):
        """analyze() must return all required fields."""
        mock_emo.return_value = MagicMock(return_value=[[
            {"label": "joy", "score": 0.7},
            {"label": "neutral", "score": 0.1},
            {"label": "sadness", "score": 0.05},
            {"label": "anger", "score": 0.05},
            {"label": "fear", "score": 0.04},
            {"label": "surprise", "score": 0.03},
            {"label": "disgust", "score": 0.03},
        ]])
        mock_sent.return_value = MagicMock(return_value=[
            {"label": "positive", "score": 0.9}
        ])

        analyzer = self._make_analyzer()
        result = analyzer.analyze("I am so happy today!", include_xai=False)

        assert result["success"] is True
        assert "emotions" in result
        assert "primary_emotion" in result
        assert "confidence" in result
        assert "sentiment" in result
        assert "text_metrics" in result
        assert "processing_time_ms" in result
        assert "timestamp" in result

    @patch("services.text_analyzer.get_emotion_classifier")
    @patch("services.text_analyzer.get_sentiment_analyzer")
    def test_analyze_emotions_sum_to_100(self, mock_sent, mock_emo):
        """Emotion percentages should approximately sum to 100."""
        mock_emo.return_value = MagicMock(return_value=[[
            {"label": "joy", "score": 0.6},
            {"label": "neutral", "score": 0.15},
            {"label": "sadness", "score": 0.1},
            {"label": "anger", "score": 0.05},
            {"label": "fear", "score": 0.04},
            {"label": "surprise", "score": 0.03},
            {"label": "disgust", "score": 0.03},
        ]])
        mock_sent.return_value = MagicMock(return_value=[
            {"label": "positive", "score": 0.85}
        ])

        analyzer = self._make_analyzer()
        result = analyzer.analyze("test text", include_xai=False)
        total = sum(result["emotions"].values())
        assert abs(total - 100.0) < 1.0, f"Emotions sum to {total}, expected ~100"

    @patch("services.text_analyzer.get_emotion_classifier")
    @patch("services.text_analyzer.get_sentiment_analyzer")
    def test_primary_emotion_is_highest(self, mock_sent, mock_emo):
        """primary_emotion must be the emotion with the highest score."""
        mock_emo.return_value = MagicMock(return_value=[[
            {"label": "anger", "score": 0.8},
            {"label": "neutral", "score": 0.1},
            {"label": "sadness", "score": 0.03},
            {"label": "joy", "score": 0.03},
            {"label": "fear", "score": 0.02},
            {"label": "surprise", "score": 0.01},
            {"label": "disgust", "score": 0.01},
        ]])
        mock_sent.return_value = MagicMock(return_value=[
            {"label": "negative", "score": 0.8}
        ])

        analyzer = self._make_analyzer()
        result = analyzer.analyze("I am furious!", include_xai=False)
        assert result["primary_emotion"] == "anger"
        assert result["confidence"] == result["emotions"]["anger"]

    # --- Empty/invalid input ---

    def test_analyze_empty_string_returns_error(self):
        analyzer = self._make_analyzer()
        result = analyzer.analyze("")
        assert result["success"] is False
        assert "error" in result

    def test_analyze_none_returns_error(self):
        analyzer = self._make_analyzer()
        result = analyzer.analyze(None)
        assert result["success"] is False

    def test_analyze_whitespace_only_returns_error(self):
        analyzer = self._make_analyzer()
        result = analyzer.analyze("   ")
        assert result["success"] is False

    # --- Sentiment analysis ---

    @patch("services.text_analyzer.get_emotion_classifier")
    @patch("services.text_analyzer.get_sentiment_analyzer")
    def test_sentiment_label_normalized(self, mock_sent, mock_emo):
        """Sentiment labels like POSITIVE should be normalized to lowercase."""
        mock_emo.return_value = MagicMock(return_value=[[
            {"label": "joy", "score": 0.9},
            {"label": "neutral", "score": 0.05},
            {"label": "sadness", "score": 0.01},
            {"label": "anger", "score": 0.01},
            {"label": "fear", "score": 0.01},
            {"label": "surprise", "score": 0.01},
            {"label": "disgust", "score": 0.01},
        ]])
        mock_sent.return_value = MagicMock(return_value=[
            {"label": "POSITIVE", "score": 0.95}
        ])

        analyzer = self._make_analyzer()
        result = analyzer.analyze("I love this!", include_xai=False)
        assert result["sentiment"]["label"] == "positive"

    # --- Sentiment fallback on error ---

    @patch("services.text_analyzer.get_emotion_classifier")
    @patch("services.text_analyzer.get_sentiment_analyzer")
    def test_sentiment_fallback_on_error(self, mock_sent, mock_emo):
        """If sentiment analyzer raises, fallback to neutral."""
        mock_emo.return_value = MagicMock(return_value=[[
            {"label": "neutral", "score": 1.0},
        ]])
        mock_sent.return_value = MagicMock(side_effect=RuntimeError("model crash"))

        analyzer = self._make_analyzer()
        result = analyzer.analyze("test", include_xai=False)
        assert result["sentiment"]["label"] == "neutral"

    # --- Emotion detection fallback ---

    @patch("services.text_analyzer.get_emotion_classifier")
    @patch("services.text_analyzer.get_sentiment_analyzer")
    def test_emotion_fallback_on_error(self, mock_sent, mock_emo):
        """If emotion classifier raises, fallback to neutral."""
        mock_emo.return_value = MagicMock(side_effect=RuntimeError("model crash"))
        mock_sent.return_value = MagicMock(return_value=[
            {"label": "neutral", "score": 0.5}
        ])

        analyzer = self._make_analyzer()
        result = analyzer.analyze("test text", include_xai=False)
        assert result["emotions"]["neutral"] == 100.0

    # --- Text metrics ---

    @patch("services.text_analyzer.get_emotion_classifier")
    @patch("services.text_analyzer.get_sentiment_analyzer")
    def test_text_metrics_word_count(self, mock_sent, mock_emo):
        mock_emo.return_value = MagicMock(return_value=[[
            {"label": "neutral", "score": 1.0},
        ]])
        mock_sent.return_value = MagicMock(return_value=[
            {"label": "neutral", "score": 0.5}
        ])

        analyzer = self._make_analyzer()
        result = analyzer.analyze("one two three four five", include_xai=False)
        assert result["text_metrics"]["word_count"] == 5

    # --- XAI explanation ---

    @patch("services.text_analyzer.get_emotion_classifier")
    @patch("services.text_analyzer.get_sentiment_analyzer")
    def test_xai_excluded_when_false(self, mock_sent, mock_emo):
        """include_xai=False should not generate explanation."""
        mock_emo.return_value = MagicMock(return_value=[[
            {"label": "joy", "score": 0.8},
            {"label": "neutral", "score": 0.2},
        ]])
        mock_sent.return_value = MagicMock(return_value=[
            {"label": "positive", "score": 0.9}
        ])

        analyzer = self._make_analyzer()
        result = analyzer.analyze("happy text", include_xai=False)
        assert result["xai_explanation"] is None

    @patch("services.text_analyzer.get_emotion_classifier")
    @patch("services.text_analyzer.get_sentiment_analyzer")
    def test_xai_included_when_true(self, mock_sent, mock_emo):
        """include_xai=True should generate explanation with required fields."""
        mock_emo.return_value = MagicMock(return_value=[[
            {"label": "joy", "score": 0.8},
            {"label": "neutral", "score": 0.1},
            {"label": "sadness", "score": 0.1},
        ]])
        mock_sent.return_value = MagicMock(return_value=[
            {"label": "positive", "score": 0.9}
        ])

        analyzer = self._make_analyzer()
        result = analyzer.analyze("I am so happy and excited!", include_xai=True)
        xai = result["xai_explanation"]
        assert xai is not None
        assert "method" in xai
        assert "reasoning" in xai
        assert "key_indicators" in xai

    # --- Keyword explanation ---

    @patch("services.text_analyzer.get_emotion_classifier")
    @patch("services.text_analyzer.get_sentiment_analyzer")
    def test_keyword_explanation_finds_emotion_words(self, mock_sent, mock_emo):
        mock_emo.return_value = MagicMock(return_value=[[
            {"label": "joy", "score": 0.9},
            {"label": "neutral", "score": 0.1},
        ]])
        mock_sent.return_value = MagicMock(return_value=[
            {"label": "positive", "score": 0.9}
        ])

        analyzer = self._make_analyzer()
        result = analyzer.analyze("I am so happy and excited!", include_xai=True)
        xai = result["xai_explanation"]
        # "happy" is in the joy keyword list
        assert "happy" in xai.get("key_indicators", [])

    # --- Text truncation for model (512 tokens) ---

    @patch("services.text_analyzer.get_emotion_classifier")
    @patch("services.text_analyzer.get_sentiment_analyzer")
    def test_long_text_is_truncated_for_model(self, mock_sent, mock_emo):
        """Text longer than 512 chars should be truncated before model call."""
        mock_classifier = MagicMock(return_value=[[
            {"label": "neutral", "score": 1.0},
        ]])
        mock_emo.return_value = mock_classifier
        mock_sent.return_value = MagicMock(return_value=[
            {"label": "neutral", "score": 0.5}
        ])

        analyzer = self._make_analyzer()
        long_text = "word " * 200  # ~1000 chars
        result = analyzer.analyze(long_text, include_xai=False)

        # Verify the classifier was called with truncated text
        call_args = mock_classifier.call_args[0][0]
        assert len(call_args) <= 512

    # --- All 7 Ekman emotions are present ---

    @patch("services.text_analyzer.get_emotion_classifier")
    @patch("services.text_analyzer.get_sentiment_analyzer")
    def test_all_seven_ekman_emotions_present(self, mock_sent, mock_emo):
        """Result must contain all 7 Ekman emotions."""
        expected_emotions = {"anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"}
        mock_emo.return_value = MagicMock(return_value=[[
            {"label": e, "score": 1.0 / 7} for e in expected_emotions
        ]])
        mock_sent.return_value = MagicMock(return_value=[
            {"label": "neutral", "score": 0.5}
        ])

        analyzer = self._make_analyzer()
        result = analyzer.analyze("test", include_xai=False)
        assert set(result["emotions"].keys()) == expected_emotions

    # --- Processing time is tracked ---

    @patch("services.text_analyzer.get_emotion_classifier")
    @patch("services.text_analyzer.get_sentiment_analyzer")
    def test_processing_time_positive(self, mock_sent, mock_emo):
        mock_emo.return_value = MagicMock(return_value=[[
            {"label": "neutral", "score": 1.0},
        ]])
        mock_sent.return_value = MagicMock(return_value=[
            {"label": "neutral", "score": 0.5}
        ])

        analyzer = self._make_analyzer()
        result = analyzer.analyze("test text", include_xai=False)
        assert result["processing_time_ms"] >= 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestTextEdgeCases:
    """Edge cases for text processing."""

    def _make_analyzer(self):
        from services.text_analyzer import TextEmotionAnalyzer
        return TextEmotionAnalyzer()

    @patch("services.text_analyzer.get_emotion_classifier")
    @patch("services.text_analyzer.get_sentiment_analyzer")
    def test_special_characters_dont_crash(self, mock_sent, mock_emo):
        mock_emo.return_value = MagicMock(return_value=[[
            {"label": "neutral", "score": 1.0},
        ]])
        mock_sent.return_value = MagicMock(return_value=[
            {"label": "neutral", "score": 0.5}
        ])

        analyzer = self._make_analyzer()
        result = analyzer.analyze("🎉🎊 @#$%^&*()", include_xai=False)
        assert result["success"] is True

    @patch("services.text_analyzer.get_emotion_classifier")
    @patch("services.text_analyzer.get_sentiment_analyzer")
    def test_unicode_text_handled(self, mock_sent, mock_emo):
        mock_emo.return_value = MagicMock(return_value=[[
            {"label": "neutral", "score": 1.0},
        ]])
        mock_sent.return_value = MagicMock(return_value=[
            {"label": "neutral", "score": 0.5}
        ])

        analyzer = self._make_analyzer()
        result = analyzer.analyze("Sretan sam! Veoma lijep dan. 日本語テスト", include_xai=False)
        assert result["success"] is True

    @patch("services.text_analyzer.get_emotion_classifier")
    @patch("services.text_analyzer.get_sentiment_analyzer")
    def test_numeric_only_text(self, mock_sent, mock_emo):
        mock_emo.return_value = MagicMock(return_value=[[
            {"label": "neutral", "score": 1.0},
        ]])
        mock_sent.return_value = MagicMock(return_value=[
            {"label": "neutral", "score": 0.5}
        ])

        analyzer = self._make_analyzer()
        result = analyzer.analyze("12345 67890", include_xai=False)
        assert result["success"] is True

    @patch("services.text_analyzer.get_emotion_classifier")
    @patch("services.text_analyzer.get_sentiment_analyzer")
    def test_single_word_input(self, mock_sent, mock_emo):
        mock_emo.return_value = MagicMock(return_value=[[
            {"label": "joy", "score": 0.9},
            {"label": "neutral", "score": 0.1},
        ]])
        mock_sent.return_value = MagicMock(return_value=[
            {"label": "positive", "score": 0.9}
        ])

        analyzer = self._make_analyzer()
        result = analyzer.analyze("happy", include_xai=False)
        assert result["success"] is True
        assert result["primary_emotion"] == "joy"
