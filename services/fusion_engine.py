"""
Multimodal Emotion Fusion Engine
Combines text, voice, and face emotion vectors using confidence-weighted fusion.
Includes incongruence detection for fake/masked emotion identification.

This is a core PhD contribution: detecting when users mask their true emotions
by comparing emotion signals across modalities (text, voice, face).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time

# Standard 7 Ekman emotions (normalized labels)
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]


class MultimodalFusionEngine:
    """
    Fuses emotion vectors from text, voice, and face modalities.

    Fusion method: Confidence-weighted average
    Incongruence: Cosine similarity between modality pairs
    """

    # Default base weights per modality
    DEFAULT_WEIGHTS = {
        "text": 0.40,   # Text is generally most reliable for explicit emotion
        "voice": 0.30,  # Voice carries prosodic/implicit emotion
        "face": 0.30,   # Face provides visual cues
    }

    # Incongruence thresholds
    INCONGRUENCE_THRESHOLD = 0.70       # Cosine sim below this = incongruent
    HIGH_INCONGRUENCE_THRESHOLD = 0.50  # Below this = possible masking

    def fuse(
        self,
        text_result: Optional[Dict] = None,
        voice_result: Optional[Dict] = None,
        face_result: Optional[Dict] = None,
    ) -> Dict:
        """
        Fuse emotion results from 1-3 modalities.

        Args:
            text_result: Result dict from text_analyzer (must have 'emotions', 'confidence')
            voice_result: Result dict from voice_analyzer
            face_result: Result dict from face_analyzer

        Returns:
            Dict with fused emotions, weights, individual results, incongruence info
        """
        start_time = time.time()

        # Collect available modality results
        modalities = {}
        if text_result and text_result.get("success", True) and text_result.get("emotions"):
            modalities["text"] = text_result
        if voice_result and voice_result.get("success", True) and voice_result.get("emotions"):
            modalities["voice"] = voice_result
        if face_result and face_result.get("success", True) and face_result.get("emotions"):
            # Only include face if face was actually detected
            if face_result.get("face_detected", True):
                modalities["face"] = face_result

        if not modalities:
            return self._empty_result("No valid modality results provided")

        # If only one modality, return it directly (no fusion needed)
        if len(modalities) == 1:
            mod_name = list(modalities.keys())[0]
            mod_result = modalities[mod_name]
            return {
                "success": True,
                "final_emotions": mod_result["emotions"],
                "primary_emotion": mod_result.get("primary_emotion", self._get_primary(mod_result["emotions"])),
                "confidence": mod_result.get("confidence", 0.0),
                "modalities_used": [mod_name],
                "weights": {mod_name: 1.0},
                "individual_results": {mod_name: mod_result["emotions"]},
                "incongruence": None,
                "xai_explanation": {
                    "method": "single_modality",
                    "reasoning": f"Only {mod_name} modality was available. No fusion performed.",
                    "modality_contributions": {mod_name: {"weight": 1.0, "contribution": "100%"}},
                },
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "timestamp": datetime.now().isoformat(),
            }

        # Extract emotion vectors and confidences
        vectors = {}
        confidences = {}
        for name, result in modalities.items():
            vectors[name] = self._to_vector(result["emotions"])
            confidences[name] = result.get("confidence", 50.0)

        # Compute dynamic weights based on confidence
        weights = self._compute_dynamic_weights(modalities, confidences)

        # Weighted fusion of emotion vectors
        fused_vector = np.zeros(len(EMOTIONS))
        for name, vec in vectors.items():
            fused_vector += weights[name] * vec

        # Normalize to sum to ~100
        total = np.sum(fused_vector)
        if total > 0:
            fused_vector = (fused_vector / total) * 100.0

        fused_emotions = {
            EMOTIONS[i]: round(float(fused_vector[i]), 1) for i in range(len(EMOTIONS))
        }
        primary_emotion = self._get_primary(fused_emotions)
        fused_confidence = fused_emotions[primary_emotion]

        # Detect incongruence between modalities
        incongruence = self._detect_incongruence(vectors, modalities)

        # Individual results for display
        individual_results = {name: result["emotions"] for name, result in modalities.items()}

        processing_time = (time.time() - start_time) * 1000

        return {
            "success": True,
            "final_emotions": fused_emotions,
            "primary_emotion": primary_emotion,
            "confidence": fused_confidence,
            "modalities_used": list(modalities.keys()),
            "weights": {k: round(v, 3) for k, v in weights.items()},
            "individual_results": individual_results,
            "incongruence": incongruence,
            "xai_explanation": self._generate_fusion_explanation(
                weights, individual_results, fused_emotions, primary_emotion, incongruence
            ),
            "processing_time_ms": round(processing_time, 2),
            "timestamp": datetime.now().isoformat(),
        }

    def _to_vector(self, emotions: Dict[str, float]) -> np.ndarray:
        """Convert emotion dict to numpy vector in standard EMOTIONS order."""
        return np.array([emotions.get(e, 0.0) for e in EMOTIONS])

    def _get_primary(self, emotions: Dict[str, float]) -> str:
        """Get the emotion with highest score."""
        return max(emotions, key=emotions.get) if emotions else "neutral"

    def _compute_dynamic_weights(
        self, modalities: Dict[str, Dict], confidences: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Dynamic weights based on base weights * confidence.
        Higher confidence modality gets more weight.
        """
        raw_weights = {}
        for name in modalities:
            base = self.DEFAULT_WEIGHTS.get(name, 0.33)
            conf = confidences.get(name, 50.0) / 100.0  # Normalize to 0-1
            raw_weights[name] = base * (0.5 + conf)  # Confidence boosts weight

        # Normalize to sum to 1
        total = sum(raw_weights.values())
        if total > 0:
            return {k: v / total for k, v in raw_weights.items()}
        else:
            n = len(modalities)
            return {k: 1.0 / n for k in modalities}

    def _detect_incongruence(
        self, vectors: Dict[str, np.ndarray], modalities: Dict[str, Dict]
    ) -> Dict:
        """
        Compare emotion vectors across modalities using cosine similarity.
        Low similarity = incongruent = possible emotional masking.
        """
        mod_names = list(vectors.keys())

        if len(mod_names) < 2:
            return {
                "is_incongruent": False,
                "overall_score": 0.0,
                "pairwise_scores": {},
                "details": None,
                "possible_masking": False,
            }

        # Pairwise cosine similarities
        pairwise = {}
        incongruence_scores = []

        for i in range(len(mod_names)):
            for j in range(i + 1, len(mod_names)):
                name_i, name_j = mod_names[i], mod_names[j]
                sim = self._cosine_similarity(vectors[name_i], vectors[name_j])
                pair_key = f"{name_i}_vs_{name_j}"
                pairwise[pair_key] = round(sim, 3)
                incongruence_scores.append(1.0 - sim)

        # Overall incongruence = max pairwise incongruence
        overall_incongruence = max(incongruence_scores) if incongruence_scores else 0.0
        min_similarity = min(pairwise.values()) if pairwise else 1.0

        is_incongruent = min_similarity < self.INCONGRUENCE_THRESHOLD
        possible_masking = min_similarity < self.HIGH_INCONGRUENCE_THRESHOLD

        # Generate human-readable details
        details = None
        if is_incongruent:
            details = self._generate_incongruence_details(vectors, modalities, pairwise)

        return {
            "is_incongruent": is_incongruent,
            "overall_score": round(overall_incongruence, 3),
            "pairwise_similarities": pairwise,
            "details": details,
            "possible_masking": possible_masking,
        }

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity between two emotion vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def _generate_incongruence_details(
        self,
        vectors: Dict[str, np.ndarray],
        modalities: Dict[str, Dict],
        pairwise: Dict[str, float],
    ) -> str:
        """Generate human-readable incongruence explanation."""
        parts = []

        for name, result in modalities.items():
            primary = result.get("primary_emotion", self._get_primary(result.get("emotions", {})))
            conf = result.get("confidence", 0)
            parts.append(f"{name}: {primary} ({conf}%)")

        modality_summary = ", ".join(parts)

        # Find the most incongruent pair
        if pairwise:
            worst_pair = min(pairwise, key=pairwise.get)
            worst_sim = pairwise[worst_pair]
            mod_a, mod_b = worst_pair.split("_vs_")

            primary_a = modalities[mod_a].get("primary_emotion", "unknown")
            primary_b = modalities[mod_b].get("primary_emotion", "unknown")

            if primary_a != primary_b:
                return (
                    f"Emotional incongruence detected: {mod_a} indicates '{primary_a}' "
                    f"but {mod_b} indicates '{primary_b}' "
                    f"(similarity: {worst_sim:.1%}). "
                    f"This may indicate emotional masking or complex mixed emotions. "
                    f"All modalities: {modality_summary}"
                )

        return f"Moderate emotional inconsistency across modalities: {modality_summary}"

    def _generate_fusion_explanation(
        self,
        weights: Dict[str, float],
        individual_results: Dict[str, Dict[str, float]],
        fused_emotions: Dict[str, float],
        primary_emotion: str,
        incongruence: Dict,
    ) -> Dict:
        """Generate XAI explanation for the fusion process."""
        modality_contributions = {}
        for name, weight in weights.items():
            mod_emotions = individual_results[name]
            mod_primary = max(mod_emotions, key=mod_emotions.get)
            modality_contributions[name] = {
                "weight": round(weight, 3),
                "contribution": f"{weight * 100:.0f}%",
                "detected_emotion": mod_primary,
                "detected_confidence": mod_emotions[mod_primary],
            }

        explanation = {
            "method": "confidence_weighted_multimodal_fusion",
            "reasoning": (
                f"Final emotion '{primary_emotion}' was determined by combining "
                f"{len(weights)} modalities with confidence-weighted fusion."
            ),
            "modality_contributions": modality_contributions,
        }

        # Add incongruence warning if detected
        if incongruence and incongruence.get("is_incongruent"):
            explanation["incongruence_warning"] = incongruence.get("details", "Emotional incongruence detected.")
            if incongruence.get("possible_masking"):
                explanation["masking_warning"] = (
                    "Significant mismatch between modalities suggests possible emotional masking. "
                    "The user may be concealing their true emotional state."
                )

        return explanation

    def _empty_result(self, reason: str) -> Dict:
        """Return empty result when no modalities available."""
        return {
            "success": False,
            "error": reason,
            "final_emotions": {e: 0.0 for e in EMOTIONS},
            "primary_emotion": "neutral",
            "confidence": 0.0,
            "modalities_used": [],
            "weights": {},
            "individual_results": {},
            "incongruence": None,
            "xai_explanation": None,
            "processing_time_ms": 0.0,
            "timestamp": datetime.now().isoformat(),
        }


# Singleton
fusion_engine = MultimodalFusionEngine()
