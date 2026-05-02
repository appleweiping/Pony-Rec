from __future__ import annotations

import math
from typing import Any

import numpy as np


def ranking_metrics(records: list[dict[str, Any]], ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    metrics: dict[str, float] = {"num_users": float(len(records))}
    for k in ks:
        hits = []
        recalls = []
        ndcgs = []
        mrrs = []
        maps = []
        for record in records:
            target = str(record.get("target_item_id", ""))
            ranking = [str(x) for x in record.get("reranked_item_ids", record.get("predicted_ranking", []))]
            topk = ranking[:k]
            hit = target in topk
            hits.append(float(hit))
            recalls.append(float(hit))
            if hit:
                rank = topk.index(target) + 1
                ndcgs.append(1.0 / math.log2(rank + 1))
                mrrs.append(1.0 / rank)
                maps.append(1.0 / rank)
            else:
                ndcgs.append(0.0)
                mrrs.append(0.0)
                maps.append(0.0)
        metrics[f"HR@{k}"] = float(np.mean(hits)) if hits else float("nan")
        metrics[f"Recall@{k}"] = float(np.mean(recalls)) if recalls else float("nan")
        metrics[f"NDCG@{k}"] = float(np.mean(ndcgs)) if ndcgs else float("nan")
        metrics[f"MRR@{k}"] = float(np.mean(mrrs)) if mrrs else float("nan")
        metrics[f"MAP@{k}"] = float(np.mean(maps)) if maps else float("nan")
    return metrics


def ece_mce(y_true: list[int] | np.ndarray, y_prob: list[float] | np.ndarray, n_bins: int = 10) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    for idx in range(n_bins):
        lower = edges[idx]
        upper = edges[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)
        if not np.any(mask):
            continue
        gap = abs(float(np.mean(y_true[mask])) - float(np.mean(y_prob[mask])))
        ece += float(np.mean(mask)) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def brier_score(y_true: list[int] | np.ndarray, y_prob: list[float] | np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def risk_coverage(records: list[dict[str, Any]]) -> list[dict[str, float]]:
    rows = sorted(
        records,
        key=lambda r: float(r.get("uncertainty_score", r.get("uncertainty", 1.0))),
    )
    out = []
    total = len(rows)
    if not total:
        return out
    mistakes = 0
    for idx, record in enumerate(rows, start=1):
        correct = int(bool(record.get("correctness", record.get("is_correct", False))))
        mistakes += 1 - correct
        out.append({"coverage": idx / total, "risk": mistakes / idx})
    return out


def exposure_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    top_items = []
    tail_hits = 0
    head_hits = 0
    categories = []
    for record in records:
        ranking = record.get("reranked_item_ids", record.get("predicted_ranking", []))
        if not ranking:
            continue
        top_item = str(ranking[0])
        top_items.append(top_item)
        candidates = [str(x) for x in record.get("candidate_item_ids", [])]
        buckets = [str(x).lower() for x in record.get("candidate_popularity_buckets", [])]
        bucket = buckets[candidates.index(top_item)] if top_item in candidates and candidates.index(top_item) < len(buckets) else "unknown"
        if bucket == "tail":
            tail_hits += 1
        if bucket == "head":
            head_hits += 1
        categories.extend(record.get("recommended_categories", []) or [])
    denom = max(1, len(top_items))
    return {
        "head_exposure_share": head_hits / denom,
        "tail_exposure_share": tail_hits / denom,
        "long_tail_coverage": len(set(top_items)) / denom,
        "unique_top_items": float(len(set(top_items))),
    }


def groupwise_ece(records: list[dict[str, Any]], *, group_key: str, n_bins: int = 10) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get(group_key, "unknown")), []).append(record)
    out: dict[str, float] = {}
    for group, rows in grouped.items():
        y_true = [int(bool(row.get("correctness", row.get("is_correct", False)))) for row in rows]
        y_prob = [float(row.get("calibrated_confidence", row.get("raw_confidence", 0.5)) or 0.5) for row in rows]
        out[group] = ece_mce(y_true, y_prob, n_bins=n_bins)[0]
    return out
