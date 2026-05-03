"""CARE-oriented uncertainty / exposure feature extraction from ranking prediction rows.

Designed for `rank_predictions.jsonl` rows (listwise pilots, LoRA, or API backends).
Downstream: calibration diagnostics, CARE reranking, population–confidence slices.
"""

from __future__ import annotations

import math
from typing import Any

from src.uncertainty.interface import LogprobEntropyEstimator, VerbalizedConfidenceEstimator


def _bucket_for_item(item_id: str, candidates: list[str], buckets: list[str]) -> str | None:
    if not item_id or not candidates:
        return None
    try:
        idx = candidates.index(str(item_id))
    except ValueError:
        return None
    if idx < len(buckets):
        return str(buckets[idx])
    return None


def extract_care_probe_features(row: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable feature dict (no mutation of ``row``)."""
    candidates = [str(x) for x in row.get("candidate_item_ids", [])]
    buckets = [str(x) for x in row.get("candidate_popularity_buckets", [])]
    raw_rank = row.get("predicted_ranking") or row.get("reranked_item_ids") or []
    ranking = [str(x) for x in raw_rank]
    target = str(row.get("target_item_id", "") or "")
    predicted = str(row.get("predicted_item_id", "") or (ranking[0] if ranking else "") or "")

    raw_conf = row.get("raw_confidence")
    try:
        raw_conf_f = float(raw_conf) if raw_conf is not None else None
    except (TypeError, ValueError):
        raw_conf_f = None

    verbal = VerbalizedConfidenceEstimator().estimate(
        {"raw_confidence": raw_conf_f if raw_conf_f is not None else row.get("confidence")}
    )

    rank_target: int | None = None
    if target and ranking:
        try:
            rank_target = ranking.index(target) + 1
        except ValueError:
            rank_target = None

    pred_bucket = _bucket_for_item(predicted, candidates, buckets)
    tgt_bucket = row.get("item_popularity_bucket") or _bucket_for_item(target, candidates, buckets)

    correctness = bool(row.get("correctness", row.get("is_correct", False)))
    missing_conf = bool(row.get("missing_confidence", False))

    logprob_diag: dict[str, Any] = {}
    if row.get("candidate_probabilities"):
        lp = LogprobEntropyEstimator().estimate({"candidate_probabilities": row.get("candidate_probabilities")})
        logprob_diag = {
            "logprob_entropy_uncertainty": lp.uncertainty_score,
            "logprob_raw_confidence": lp.raw_confidence,
            "logprob_available": lp.diagnostics.get("available", False),
        }

    # Recommendation-specific risk proxies (not final CARE losses).
    high_conf_wrong = bool(raw_conf_f is not None and raw_conf_f >= 0.7 and not correctness)
    low_conf_right = bool(raw_conf_f is not None and raw_conf_f <= 0.4 and correctness)
    head_top_wrong = bool(pred_bucket and str(pred_bucket).lower() == "head" and not correctness)

    return {
        "verbalized_raw_confidence": raw_conf_f,
        "verbalized_uncertainty_score": verbal.uncertainty_score,
        "missing_confidence": missing_conf,
        "predicted_top_item_id": predicted or None,
        "predicted_top_popularity_bucket": pred_bucket,
        "target_popularity_bucket": tgt_bucket,
        "rank_target_in_predicted_list": rank_target,
        "predicted_list_len": len(ranking),
        "candidate_count": len(candidates),
        "high_confidence_wrong": high_conf_wrong,
        "low_confidence_correct": low_conf_right,
        "head_top1_wrong": head_top_wrong,
        "confidence_minus_correctness": (raw_conf_f - float(correctness)) if raw_conf_f is not None else None,
        **logprob_diag,
    }


def summarize_probe_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate rates for a JSON sidecar (pilot-scale, not paper)."""
    n = len(rows)
    if not n:
        return {"n_rows": 0}

    def rate(key: str) -> float:
        return sum(1 for r in rows if r.get("care_features", {}).get(key)) / n

    miss_conf = sum(1 for r in rows if r.get("care_features", {}).get("missing_confidence")) / n
    cor = [bool(r.get("correctness", r.get("is_correct", False))) for r in rows]
    acc = sum(cor) / n

    raw_list: list[float] = []
    for r in rows:
        v = r.get("care_features", {}).get("verbalized_raw_confidence")
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            raw_list.append(float(v))

    return {
        "n_rows": n,
        "accuracy": acc,
        "missing_confidence_rate": miss_conf,
        "high_confidence_wrong_rate": rate("high_confidence_wrong"),
        "low_confidence_correct_rate": rate("low_confidence_correct"),
        "head_top1_wrong_rate": rate("head_top1_wrong"),
        "mean_verbalized_raw_confidence": sum(raw_list) / len(raw_list) if raw_list else None,
    }
