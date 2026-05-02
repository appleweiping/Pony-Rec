from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RerankConfig:
    lambda_penalty: float = 0.5
    popularity_penalty: float = 0.0
    exploration_bonus: float = 0.0
    abstention_threshold: float | None = None
    truncation_threshold: float | None = None


def uncertainty_aware_score(relevance_score: float, uncertainty: float, lambda_penalty: float) -> float:
    return float(relevance_score) - float(lambda_penalty) * float(uncertainty)


def popularity_aware_score(
    *,
    relevance_score: float,
    uncertainty: float,
    popularity_bucket: str,
    confidence: float,
    lambda_penalty: float,
    popularity_penalty: float,
    exploration_bonus: float,
) -> float:
    score = uncertainty_aware_score(relevance_score, uncertainty, lambda_penalty)
    bucket = str(popularity_bucket).lower()
    if bucket == "head" and confidence > 0.8 and uncertainty > 0.3:
        score -= float(popularity_penalty)
    if bucket == "tail":
        score += float(exploration_bonus) * max(0.0, 1.0 - uncertainty)
    return score


def rerank_candidate_record(record: dict[str, Any], config: RerankConfig) -> dict[str, Any]:
    candidate_ids = [str(x) for x in record.get("candidate_item_ids", [])]
    original_ranking = [str(x) for x in record.get("predicted_ranking", record.get("ranked_item_ids", []))]
    if not original_ranking:
        original_ranking = list(candidate_ids)
    rank_lookup = {item_id: idx + 1 for idx, item_id in enumerate(original_ranking)}
    n = max(len(candidate_ids), 1)
    uncertainty = float(record.get("uncertainty_score", record.get("uncertainty", 1.0 - float(record.get("raw_confidence", 0.5)))))
    confidence = float(record.get("calibrated_confidence", record.get("raw_confidence", 0.5)))
    buckets = [str(x) for x in record.get("candidate_popularity_buckets", [])]
    bucket_lookup = {item_id: buckets[idx] if idx < len(buckets) else "unknown" for idx, item_id in enumerate(candidate_ids)}
    rows = []
    for item_id in candidate_ids:
        original_rank = rank_lookup.get(item_id, n + 1)
        relevance = (n - original_rank + 1) / n if original_rank <= n else 0.0
        score = popularity_aware_score(
            relevance_score=relevance,
            uncertainty=uncertainty,
            popularity_bucket=bucket_lookup.get(item_id, "unknown"),
            confidence=confidence,
            lambda_penalty=config.lambda_penalty,
            popularity_penalty=config.popularity_penalty,
            exploration_bonus=config.exploration_bonus,
        )
        rows.append({"item_id": item_id, "score": score, "original_rank": original_rank})
    reranked = [row["item_id"] for row in sorted(rows, key=lambda row: (-row["score"], row["original_rank"], row["item_id"]))]
    if config.truncation_threshold is not None and uncertainty > config.truncation_threshold:
        reranked = reranked[:1]
    abstained = config.abstention_threshold is not None and uncertainty > config.abstention_threshold
    out = dict(record)
    out.update(
        {
            "reranked_item_ids": reranked,
            "predicted_item_id": "" if abstained else (reranked[0] if reranked else ""),
            "abstained": bool(abstained),
            "rerank_lambda": float(config.lambda_penalty),
            "rerank_variant": "uncertainty_popularity_aware" if config.popularity_penalty else "uncertainty_linear",
        }
    )
    return out
