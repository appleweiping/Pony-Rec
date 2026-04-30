from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


RELEVANCE_CALIBRATABLE = {
    "self_reported_confidence",
    "calibrated_relevance_probability",
}

SELECTIVE_ANALYSIS_COMPATIBLE = {
    "self_reported_confidence",
    "calibrated_relevance_probability",
    "score_margin_reliability",
    "rank_margin_reliability",
    "stability_reliability",
}


@dataclass(frozen=True)
class BaselineReliabilityProxy:
    baseline_name: str
    baseline_family: str
    confidence_semantics: str
    calibration_target: str
    risk_of_unfair_comparison: str = "medium"
    protocol_gap: str = "none"
    status_label: str = "proxy_only"

    @property
    def is_relevance_calibratable(self) -> bool:
        return self.confidence_semantics in RELEVANCE_CALIBRATABLE and self.calibration_target == "relevance"

    @property
    def can_compute_ece(self) -> bool:
        return self.is_relevance_calibratable

    @property
    def can_run_selective_analysis(self) -> bool:
        return self.confidence_semantics in SELECTIVE_ANALYSIS_COMPATIBLE

    @property
    def main_table_eligible(self) -> bool:
        return self.status_label == "completed_result" and self.protocol_gap == "none"

    def to_record(self) -> dict[str, Any]:
        return {
            "baseline_name": self.baseline_name,
            "baseline_family": self.baseline_family,
            "confidence_semantics": self.confidence_semantics,
            "calibration_target": self.calibration_target,
            "is_relevance_calibratable": self.is_relevance_calibratable,
            "can_compute_ece": self.can_compute_ece,
            "can_run_selective_analysis": self.can_run_selective_analysis,
            "risk_of_unfair_comparison": self.risk_of_unfair_comparison,
            "protocol_gap": self.protocol_gap,
            "status_label": self.status_label,
            "main_table_eligible": self.main_table_eligible,
        }


def proxy_from_mapping(row: dict[str, Any]) -> BaselineReliabilityProxy:
    return BaselineReliabilityProxy(
        baseline_name=str(row.get("baseline_name", "")).strip(),
        baseline_family=str(row.get("baseline_family", "")).strip(),
        confidence_semantics=str(row.get("confidence_semantics", "non_isomorphic")).strip(),
        calibration_target=str(row.get("calibration_target", "none")).strip(),
        risk_of_unfair_comparison=str(row.get("risk_of_unfair_comparison", "medium")).strip(),
        protocol_gap=str(row.get("protocol_gap", "none")).strip(),
        status_label=str(row.get("status_label", "proxy_only")).strip(),
    )


def build_proxy_audit(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records = [proxy_from_mapping(row).to_record() for row in rows]
    df = pd.DataFrame(records)
    if df.empty:
        return df

    invalid_ece = df[
        (df["can_compute_ece"])
        & (~df["is_relevance_calibratable"])
    ]
    if not invalid_ece.empty:
        names = ", ".join(invalid_ece["baseline_name"].astype(str).tolist())
        raise ValueError(f"ECE requested for non-calibratable proxy baselines: {names}")

    non_isomorphic_main = df[
        (df["confidence_semantics"] == "non_isomorphic")
        & (df["main_table_eligible"])
    ]
    if not non_isomorphic_main.empty:
        names = ", ".join(non_isomorphic_main["baseline_name"].astype(str).tolist())
        raise ValueError(f"Non-isomorphic proxy baselines cannot enter the same-schema main table: {names}")

    exposure_ece = df[
        (df["confidence_semantics"] == "exposure_policy_certainty")
        & (df["can_compute_ece"])
    ]
    if not exposure_ece.empty:
        names = ", ".join(exposure_ece["baseline_name"].astype(str).tolist())
        raise ValueError(f"Exposure policy certainty must not compute relevance ECE: {names}")

    return df


DEFAULT_BASELINE_RELIABILITY_PROXIES = [
    {
        "baseline_name": "llm_direct_ranking",
        "baseline_family": "llm_direct_ranking",
        "confidence_semantics": "non_isomorphic",
        "calibration_target": "none",
        "risk_of_unfair_comparison": "low_when_same_prompt_schema",
        "protocol_gap": "no_confidence_output",
        "status_label": "proxy_only",
    },
    {
        "baseline_name": "raw_verbalized_confidence",
        "baseline_family": "uncertainty_baseline",
        "confidence_semantics": "self_reported_confidence",
        "calibration_target": "relevance",
        "risk_of_unfair_comparison": "low",
        "protocol_gap": "none",
        "status_label": "completed_result",
    },
    {
        "baseline_name": "calibrated_confidence_platt",
        "baseline_family": "uncertainty_baseline",
        "confidence_semantics": "calibrated_relevance_probability",
        "calibration_target": "relevance",
        "risk_of_unfair_comparison": "low",
        "protocol_gap": "none",
        "status_label": "completed_result",
    },
    {
        "baseline_name": "popularity_prior",
        "baseline_family": "simple_recommendation_prior",
        "confidence_semantics": "exposure_policy_certainty",
        "calibration_target": "exposure_policy",
        "risk_of_unfair_comparison": "high",
        "protocol_gap": "not_relevance_calibratable",
        "status_label": "proxy_only",
    },
    {
        "baseline_name": "rank_margin",
        "baseline_family": "score_margin_prior",
        "confidence_semantics": "rank_margin_reliability",
        "calibration_target": "ranking_stability",
        "risk_of_unfair_comparison": "medium",
        "protocol_gap": "not_relevance_calibratable",
        "status_label": "proxy_only",
    },
]
