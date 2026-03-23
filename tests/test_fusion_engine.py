"""
Unit tests for MultimodalFusionEngine.
Tests emotion fusion, dynamic weighting, incongruence detection,
and XAI explanation generation.
Run with: pytest tests/test_fusion_engine.py -v
"""

import pytest
import numpy as np
from services.fusion_engine import MultimodalFusionEngine, EMOTIONS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return MultimodalFusionEngine()


def _make_result(primary, confidence, **overrides):
    """Helper to create a mock modality result."""
    emotions = {e: 0.0 for e in EMOTIONS}
    emotions[primary] = confidence
    remaining = 100.0 - confidence
    other_emotions = [e for e in EMOTIONS if e != primary]
    for e in other_emotions:
        emotions[e] = round(remaining / len(other_emotions), 1)

    result = {
        "success": True,
        "emotions": emotions,
        "primary_emotion": primary,
        "confidence": confidence,
    }
    result.update(overrides)
    return result


# ---------------------------------------------------------------------------
# Single modality tests
# ---------------------------------------------------------------------------

class TestSingleModality:
    """When only one modality is provided, no fusion occurs."""

    def test_text_only(self, engine):
        result = engine.fuse(text_result=_make_result("joy", 80.0))
        assert result["success"] is True
        assert result["primary_emotion"] == "joy"
        assert result["modalities_used"] == ["text"]
        assert result["weights"] == {"text": 1.0}
        assert result["incongruence"] is None

    def test_voice_only(self, engine):
        result = engine.fuse(voice_result=_make_result("sadness", 60.0))
        assert result["primary_emotion"] == "sadness"
        assert result["modalities_used"] == ["voice"]

    def test_face_only(self, engine):
        result = engine.fuse(face_result=_make_result("anger", 70.0, face_detected=True))
        assert result["primary_emotion"] == "anger"

    def test_face_not_detected_excluded(self, engine):
        """Face result with face_detected=False should be excluded."""
        face_result = _make_result("neutral", 0.0, face_detected=False)
        result = engine.fuse(face_result=face_result)
        assert result["success"] is False  # No valid modalities

    def test_failed_result_excluded(self, engine):
        """A result with success=False should be excluded."""
        text_result = {"success": False, "emotions": {}, "confidence": 0}
        result = engine.fuse(text_result=text_result)
        assert result["success"] is False

    def test_no_modalities_returns_error(self, engine):
        result = engine.fuse()
        assert result["success"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# Two modality fusion tests
# ---------------------------------------------------------------------------

class TestTwoModalityFusion:
    """Tests for fusing two modalities."""

    def test_text_and_voice_fusion(self, engine):
        text = _make_result("joy", 80.0)
        voice = _make_result("joy", 60.0)
        result = engine.fuse(text_result=text, voice_result=voice)

        assert result["success"] is True
        assert result["primary_emotion"] == "joy"
        assert set(result["modalities_used"]) == {"text", "voice"}
        assert len(result["weights"]) == 2
        assert result["weights"]["text"] > 0
        assert result["weights"]["voice"] > 0

    def test_text_and_face_fusion(self, engine):
        text = _make_result("sadness", 70.0)
        face = _make_result("sadness", 50.0, face_detected=True)
        result = engine.fuse(text_result=text, face_result=face)

        assert result["success"] is True
        assert set(result["modalities_used"]) == {"text", "face"}

    def test_fused_emotions_sum_to_100(self, engine):
        text = _make_result("joy", 80.0)
        voice = _make_result("neutral", 60.0)
        result = engine.fuse(text_result=text, voice_result=voice)

        total = sum(result["final_emotions"].values())
        assert abs(total - 100.0) < 1.0, f"Fused emotions sum to {total}"

    def test_all_seven_emotions_in_fused_result(self, engine):
        text = _make_result("joy", 80.0)
        voice = _make_result("anger", 60.0)
        result = engine.fuse(text_result=text, voice_result=voice)
        assert set(result["final_emotions"].keys()) == set(EMOTIONS)


# ---------------------------------------------------------------------------
# Three modality fusion tests
# ---------------------------------------------------------------------------

class TestThreeModalityFusion:

    def test_all_three_modalities(self, engine):
        text = _make_result("joy", 80.0)
        voice = _make_result("joy", 70.0)
        face = _make_result("joy", 90.0, face_detected=True)
        result = engine.fuse(text_result=text, voice_result=voice, face_result=face)

        assert result["success"] is True
        assert result["primary_emotion"] == "joy"
        assert set(result["modalities_used"]) == {"text", "voice", "face"}

    def test_conflicting_three_modalities(self, engine):
        text = _make_result("joy", 80.0)
        voice = _make_result("sadness", 70.0)
        face = _make_result("anger", 60.0, face_detected=True)
        result = engine.fuse(text_result=text, voice_result=voice, face_result=face)

        assert result["success"] is True
        assert result["incongruence"] is not None
        assert result["incongruence"]["is_incongruent"] is True

    def test_three_modalities_fused_sums_to_100(self, engine):
        text = _make_result("joy", 50.0)
        voice = _make_result("sadness", 40.0)
        face = _make_result("neutral", 60.0, face_detected=True)
        result = engine.fuse(text_result=text, voice_result=voice, face_result=face)

        total = sum(result["final_emotions"].values())
        assert abs(total - 100.0) < 1.0


# ---------------------------------------------------------------------------
# Dynamic weight tests
# ---------------------------------------------------------------------------

class TestDynamicWeights:

    def test_higher_confidence_gets_more_weight(self, engine):
        """Modality with higher confidence should receive higher weight."""
        text = _make_result("joy", 95.0)   # High confidence
        voice = _make_result("joy", 30.0)  # Low confidence
        result = engine.fuse(text_result=text, voice_result=voice)

        assert result["weights"]["text"] > result["weights"]["voice"]

    def test_weights_sum_to_one(self, engine):
        text = _make_result("joy", 80.0)
        voice = _make_result("sadness", 60.0)
        face = _make_result("anger", 70.0, face_detected=True)
        result = engine.fuse(text_result=text, voice_result=voice, face_result=face)

        total_weight = sum(result["weights"].values())
        assert abs(total_weight - 1.0) < 0.01

    def test_equal_confidence_reflects_base_weights(self, engine):
        """With equal confidence, text should get ~40%, voice/face ~30% each."""
        text = _make_result("joy", 50.0)
        voice = _make_result("joy", 50.0)
        face = _make_result("joy", 50.0, face_detected=True)
        result = engine.fuse(text_result=text, voice_result=voice, face_result=face)

        # Text base weight (0.40) > voice/face base weight (0.30)
        assert result["weights"]["text"] > result["weights"]["voice"]
        assert result["weights"]["text"] > result["weights"]["face"]


# ---------------------------------------------------------------------------
# Incongruence detection tests
# ---------------------------------------------------------------------------

class TestIncongruenceDetection:

    def test_congruent_modalities(self, engine):
        """Same emotion across modalities = no incongruence."""
        text = _make_result("joy", 80.0)
        voice = _make_result("joy", 75.0)
        result = engine.fuse(text_result=text, voice_result=voice)

        assert result["incongruence"]["is_incongruent"] is False
        assert result["incongruence"]["possible_masking"] is False

    def test_incongruent_modalities(self, engine):
        """Very different emotions = incongruence detected."""
        text = _make_result("joy", 90.0)
        voice = _make_result("sadness", 85.0)
        result = engine.fuse(text_result=text, voice_result=voice)

        assert result["incongruence"]["is_incongruent"] is True

    def test_high_incongruence_flags_masking(self, engine):
        """Very high incongruence should flag possible_masking."""
        text = _make_result("joy", 95.0)
        voice = _make_result("anger", 90.0)
        result = engine.fuse(text_result=text, voice_result=voice)

        assert result["incongruence"]["possible_masking"] is True

    def test_pairwise_similarities_computed(self, engine):
        text = _make_result("joy", 80.0)
        voice = _make_result("sadness", 70.0)
        face = _make_result("anger", 60.0, face_detected=True)
        result = engine.fuse(text_result=text, voice_result=voice, face_result=face)

        pairwise = result["incongruence"]["pairwise_similarities"]
        # Should have 3 pairs for 3 modalities
        assert len(pairwise) == 3
        for key, value in pairwise.items():
            assert 0.0 <= value <= 1.0

    def test_incongruent_details_generated(self, engine):
        text = _make_result("joy", 90.0)
        voice = _make_result("sadness", 85.0)
        result = engine.fuse(text_result=text, voice_result=voice)

        if result["incongruence"]["is_incongruent"]:
            assert result["incongruence"]["details"] is not None
            assert len(result["incongruence"]["details"]) > 0


# ---------------------------------------------------------------------------
# Cosine similarity tests
# ---------------------------------------------------------------------------

class TestCosineSimilarity:

    def test_identical_vectors(self, engine):
        v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert engine._cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self, engine):
        v1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert engine._cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_opposite_vectors(self, engine):
        v1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert engine._cosine_similarity(v1, v2) == pytest.approx(-1.0)

    def test_zero_vector(self, engine):
        v1 = np.zeros(7)
        v2 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert engine._cosine_similarity(v1, v2) == 0.0

    def test_similar_vectors_high_similarity(self, engine):
        v1 = np.array([80.0, 10.0, 5.0, 2.0, 1.0, 1.0, 1.0])
        v2 = np.array([75.0, 12.0, 6.0, 3.0, 2.0, 1.0, 1.0])
        sim = engine._cosine_similarity(v1, v2)
        assert sim > 0.99


# ---------------------------------------------------------------------------
# XAI explanation tests
# ---------------------------------------------------------------------------

class TestFusionXAI:

    def test_single_modality_explanation(self, engine):
        result = engine.fuse(text_result=_make_result("joy", 80.0))
        xai = result["xai_explanation"]
        assert xai["method"] == "single_modality"
        assert "text" in xai["reasoning"]

    def test_multi_modality_explanation(self, engine):
        text = _make_result("joy", 80.0)
        voice = _make_result("joy", 70.0)
        result = engine.fuse(text_result=text, voice_result=voice)
        xai = result["xai_explanation"]
        assert xai["method"] == "confidence_weighted_multimodal_fusion"
        assert "modality_contributions" in xai
        assert "text" in xai["modality_contributions"]
        assert "voice" in xai["modality_contributions"]

    def test_incongruence_warning_in_explanation(self, engine):
        text = _make_result("joy", 90.0)
        voice = _make_result("sadness", 85.0)
        result = engine.fuse(text_result=text, voice_result=voice)

        if result["incongruence"]["is_incongruent"]:
            xai = result["xai_explanation"]
            assert "incongruence_warning" in xai

    def test_masking_warning_in_explanation(self, engine):
        text = _make_result("joy", 95.0)
        voice = _make_result("anger", 92.0)
        result = engine.fuse(text_result=text, voice_result=voice)

        if result["incongruence"]["possible_masking"]:
            xai = result["xai_explanation"]
            assert "masking_warning" in xai


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestFusionEdgeCases:

    def test_all_none_inputs(self, engine):
        result = engine.fuse(text_result=None, voice_result=None, face_result=None)
        assert result["success"] is False

    def test_empty_emotions_dict(self, engine):
        result = engine.fuse(text_result={"success": True, "emotions": {}, "confidence": 0})
        assert result["success"] is False

    def test_all_zero_confidence(self, engine):
        text = _make_result("neutral", 0.0)
        voice = _make_result("neutral", 0.0)
        result = engine.fuse(text_result=text, voice_result=voice)
        # Should still produce a result (even if low quality)
        assert result["success"] is True

    def test_processing_time_tracked(self, engine):
        result = engine.fuse(text_result=_make_result("joy", 80.0))
        assert result["processing_time_ms"] >= 0

    def test_timestamp_present(self, engine):
        result = engine.fuse(text_result=_make_result("joy", 80.0))
        assert "timestamp" in result
