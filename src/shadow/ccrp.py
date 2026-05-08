from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    return max(0.0, min(1.0, parsed))


@dataclass(frozen=True)
class CcrpWeights:
    boundary: float = 0.5
    calibration_gap: float = 0.3
    evidence: float = 0.2

    def normalized(self) -> "CcrpWeights":
        total = self.boundary + self.calibration_gap + self.evidence
        if total <= 0:
            return CcrpWeights()
        return CcrpWeights(
            boundary=self.boundary / total,
            calibration_gap=self.calibration_gap / total,
            evidence=self.evidence / total,
        )


CCRP_ABLATIONS = {
    "full": CcrpWeights(0.5, 0.3, 0.2),
    "without_calibration_gap": CcrpWeights(0.7, 0.0, 0.3),
    "without_evidence_support": CcrpWeights(0.625, 0.375, 0.0),
    "without_counterevidence": CcrpWeights(0.5, 0.3, 0.2),
    "without_risk_penalty": CcrpWeights(0.5, 0.3, 0.2),
}

CCRP_SCORE_MODES = {
    "confidence_only",
    "evidence_only",
    "confidence_plus_evidence",
    "full",
}


def compute_ccrp_record(
    record: dict[str, Any],
    *,
    weights: CcrpWeights | None = None,
    eta: float = 1.0,
    ablation: str = "full",
    score_mode: str = "full",
    confidence_weight: float = 0.7,
) -> dict[str, float]:
    """Compute Calibrated Candidate Relevance Posterior scores.

    The signal is centered on a calibrated relevance posterior. Uncertainty is
    not reduced to ``1 - p``; it combines boundary ambiguity, calibration gap,
    and evidence insufficiency. Weights must be fixed or selected on validation
    data before test evaluation.
    """

    weights = (weights or CCRP_ABLATIONS.get(ablation, CCRP_ABLATIONS["full"])).normalized()
    raw_probability = clamp01(
        record.get(
            "relevance_probability",
            record.get("shadow_primary_score", record.get("shadow_score", record.get("confidence", 0.0))),
        )
    )
    calibrated_probability = clamp01(
        record.get(
            "calibrated_relevance_probability",
            record.get("shadow_calibrated_score", record.get("calibrated_confidence", raw_probability)),
        ),
        default=raw_probability,
    )
    evidence_support = clamp01(record.get("evidence_support", record.get("evidence", 0.0)))
    counterevidence = clamp01(record.get("counterevidence_strength", record.get("counterevidence", 0.0)))
    if ablation == "without_counterevidence":
        counterevidence = 0.0

    evidence_score = clamp01(evidence_support - counterevidence)
    boundary_uncertainty = 4.0 * calibrated_probability * (1.0 - calibrated_probability)
    calibration_gap = abs(raw_probability - calibrated_probability)
    evidence_uncertainty = 1.0 - evidence_score
    uncertainty = clamp01(
        weights.boundary * boundary_uncertainty
        + weights.calibration_gap * calibration_gap
        + weights.evidence * evidence_uncertainty
    )
    score_mode = str(score_mode or "full").strip().lower()
    if score_mode not in CCRP_SCORE_MODES:
        raise ValueError(f"Unsupported C-CRP score_mode={score_mode!r}; expected one of {sorted(CCRP_SCORE_MODES)}.")
    confidence_weight = clamp01(confidence_weight, default=0.7)
    if score_mode == "confidence_only":
        base_score = calibrated_probability
    elif score_mode == "evidence_only":
        base_score = evidence_score
    elif score_mode == "confidence_plus_evidence":
        base_score = confidence_weight * calibrated_probability + (1.0 - confidence_weight) * evidence_score
    else:
        base_score = calibrated_probability

    if ablation == "without_risk_penalty" or score_mode in {"confidence_only", "evidence_only"}:
        risk_adjusted_score = base_score
    else:
        risk_adjusted_score = base_score * ((1.0 - uncertainty) ** max(0.0, float(eta)))

    return {
        "ccrp_raw_probability": raw_probability,
        "ccrp_calibrated_probability": calibrated_probability,
        "ccrp_boundary_uncertainty": boundary_uncertainty,
        "ccrp_calibration_gap": calibration_gap,
        "ccrp_evidence_score": evidence_score,
        "ccrp_evidence_uncertainty": evidence_uncertainty,
        "ccrp_uncertainty": uncertainty,
        "ccrp_base_score": clamp01(base_score),
        "ccrp_risk_adjusted_score": clamp01(risk_adjusted_score),
        "ccrp_weight_boundary": weights.boundary,
        "ccrp_weight_calibration_gap": weights.calibration_gap,
        "ccrp_weight_evidence": weights.evidence,
        "ccrp_score_mode": score_mode,
        "ccrp_confidence_weight": confidence_weight,
    }


def apply_ccrp_scores(
    df: pd.DataFrame,
    *,
    weights: CcrpWeights | None = None,
    eta: float = 1.0,
    ablation: str = "full",
    score_mode: str = "full",
    confidence_weight: float = 0.7,
) -> pd.DataFrame:
    out = df.copy()
    rows = [
        compute_ccrp_record(
            record,
            weights=weights,
            eta=eta,
            ablation=ablation,
            score_mode=score_mode,
            confidence_weight=confidence_weight,
        )
        for record in out.to_dict(orient="records")
    ]
    score_df = pd.DataFrame(rows)
    return pd.concat([out.reset_index(drop=True), score_df.reset_index(drop=True)], axis=1)


def parse_weights(values: Iterable[float] | None) -> CcrpWeights:
    if values is None:
        return CcrpWeights()
    parsed = list(values)
    if len(parsed) != 3:
        raise ValueError("C-CRP weights must contain exactly three values: boundary calibration_gap evidence.")
    return CcrpWeights(float(parsed[0]), float(parsed[1]), float(parsed[2])).normalized()
