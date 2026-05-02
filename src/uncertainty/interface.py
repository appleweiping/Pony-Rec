from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UncertaintyEstimate:
    estimator_name: str
    raw_confidence: float | None
    uncertainty_score: float
    diagnostics: dict[str, Any]


class UncertaintyEstimator(ABC):
    estimator_name: str

    @abstractmethod
    def estimate(self, prediction: dict[str, Any]) -> UncertaintyEstimate:
        raise NotImplementedError


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


class VerbalizedConfidenceEstimator(UncertaintyEstimator):
    estimator_name = "verbalized_confidence"

    def estimate(self, prediction: dict[str, Any]) -> UncertaintyEstimate:
        confidence = prediction.get("raw_confidence", prediction.get("confidence"))
        confidence = _clip01(float(confidence)) if confidence is not None else 0.5
        return UncertaintyEstimate(self.estimator_name, confidence, 1.0 - confidence, {})


class SelfConsistencyEstimator(UncertaintyEstimator):
    estimator_name = "self_consistency_confidence"

    def estimate(self, prediction: dict[str, Any]) -> UncertaintyEstimate:
        samples = prediction.get("self_consistency_predictions", []) or []
        if not samples:
            return UncertaintyEstimate(self.estimator_name, None, 0.5, {"num_samples": 0})
        top = prediction.get("predicted_item_id")
        agreement = sum(1 for item in samples if item == top) / len(samples)
        return UncertaintyEstimate(self.estimator_name, agreement, 1.0 - agreement, {"num_samples": len(samples)})


class PerturbationConsistencyEstimator(SelfConsistencyEstimator):
    estimator_name = "perturbation_consistency_confidence"


class HybridConfidenceEstimator(UncertaintyEstimator):
    estimator_name = "hybrid_verbalized_consistency"

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = _clip01(alpha)
        self.verbalized = VerbalizedConfidenceEstimator()
        self.consistency = SelfConsistencyEstimator()

    def estimate(self, prediction: dict[str, Any]) -> UncertaintyEstimate:
        verbal = self.verbalized.estimate(prediction)
        consistency = self.consistency.estimate(prediction)
        confidence = self.alpha * (verbal.raw_confidence or 0.5) + (1.0 - self.alpha) * (
            consistency.raw_confidence or 0.5
        )
        return UncertaintyEstimate(
            self.estimator_name,
            confidence,
            1.0 - confidence,
            {"alpha": self.alpha, "verbalized": verbal.diagnostics, "consistency": consistency.diagnostics},
        )


class LogprobEntropyEstimator(UncertaintyEstimator):
    estimator_name = "logprob_entropy_margin"

    def estimate(self, prediction: dict[str, Any]) -> UncertaintyEstimate:
        probs = prediction.get("candidate_probabilities") or []
        probs = [_clip01(float(p)) for p in probs if p is not None]
        total = sum(probs)
        if not probs or total <= 0:
            return UncertaintyEstimate(self.estimator_name, None, 0.5, {"available": False})
        normalized = [p / total for p in probs]
        entropy = -sum(p * math.log(p + 1e-12) for p in normalized) / math.log(len(normalized))
        sorted_probs = sorted(normalized, reverse=True)
        margin = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0)
        uncertainty = _clip01(0.5 * entropy + 0.5 * (1.0 - margin))
        return UncertaintyEstimate(self.estimator_name, 1.0 - uncertainty, uncertainty, {"available": True})


class SemanticDispersionEstimator(UncertaintyEstimator):
    estimator_name = "candidate_semantic_dispersion"

    def estimate(self, prediction: dict[str, Any]) -> UncertaintyEstimate:
        dispersion = prediction.get("semantic_dispersion")
        if dispersion is None:
            return UncertaintyEstimate(self.estimator_name, None, 0.5, {"available": False})
        uncertainty = _clip01(float(dispersion))
        return UncertaintyEstimate(self.estimator_name, 1.0 - uncertainty, uncertainty, {"available": True})


def attach_conditioned_diagnostics(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "popularity_bucket": prediction.get("target_popularity_bucket") or prediction.get("item_popularity_bucket"),
        "domain": prediction.get("domain"),
        "history_length_bucket": prediction.get("history_length_bucket"),
    }
