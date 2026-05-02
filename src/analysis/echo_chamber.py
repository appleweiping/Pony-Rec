from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def popularity_confidence_diagnostics(records: list[dict[str, Any]]) -> dict[str, float]:
    counts = np.asarray([float(r.get("item_popularity_count") or 0.0) for r in records])
    confidences = np.asarray([float(r.get("calibrated_confidence", r.get("raw_confidence", 0.5)) or 0.5) for r in records])
    if len(records) < 2 or np.std(counts) == 0 or np.std(confidences) == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(counts, confidences)[0, 1])
    by_bucket: dict[str, list[float]] = {}
    for record, confidence in zip(records, confidences):
        by_bucket.setdefault(str(record.get("item_popularity_bucket", "unknown")), []).append(float(confidence))
    return {
        "popularity_confidence_correlation": corr,
        "head_avg_confidence": float(np.mean(by_bucket.get("head", [np.nan]))),
        "mid_avg_confidence": float(np.mean(by_bucket.get("mid", [np.nan]))),
        "tail_avg_confidence": float(np.mean(by_bucket.get("tail", [np.nan]))),
        "head_tail_confidence_gap": float(np.mean(by_bucket.get("head", [np.nan])) - np.mean(by_bucket.get("tail", [np.nan]))),
    }


def exposure_concentration(records: list[dict[str, Any]]) -> dict[str, float]:
    top_items = []
    top_buckets = []
    for record in records:
        ranking = record.get("reranked_item_ids", record.get("predicted_ranking", []))
        if not ranking:
            continue
        item_id = str(ranking[0])
        top_items.append(item_id)
        candidates = [str(x) for x in record.get("candidate_item_ids", [])]
        buckets = [str(x) for x in record.get("candidate_popularity_buckets", [])]
        top_buckets.append(buckets[candidates.index(item_id)] if item_id in candidates and candidates.index(item_id) < len(buckets) else "unknown")
    n = max(1, len(top_items))
    counts = Counter(top_items)
    bucket_counts = Counter(top_buckets)
    return {
        "unique_recommended_items": float(len(counts)),
        "top1_item_exposure_share": max(counts.values()) / n if counts else 0.0,
        "head_item_exposure_share": bucket_counts.get("head", 0) / n,
        "tail_item_exposure_share": bucket_counts.get("tail", 0) / n,
        "long_tail_coverage": len([item for item, cnt in counts.items() if cnt > 0]) / n,
    }


def echo_chamber_report(records: list[dict[str, Any]]) -> dict[str, float]:
    return {
        **popularity_confidence_diagnostics(records),
        **exposure_concentration(records),
    }
