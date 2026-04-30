from __future__ import annotations

from collections import defaultdict
from math import isnan
from statistics import mean
from typing import Any, Iterable

from src.shadow.scoring import compute_shadow_scores


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _is_nan(value: Any) -> bool:
    try:
        return isnan(float(value))
    except Exception:
        return False


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    if value is None or _is_nan(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _first_float(record: dict[str, Any], fields: Iterable[str], default: float | None = None) -> float | None:
    for field in fields:
        if field in record and record[field] is not None and not _is_nan(record[field]):
            parsed = _safe_float(record[field], default=None)
            if parsed is not None:
                return parsed
    return default


def _normalize_item_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None or _is_nan(value):
        return []
    text = str(value).strip()
    return [text] if text else []


def _list_value(values: list[Any], idx: int, default: Any = "") -> Any:
    if idx < len(values):
        return values[idx]
    return default


def _event_id(record: dict[str, Any]) -> str:
    explicit = str(record.get("source_event_id") or "").strip()
    if explicit:
        return explicit
    return f"{record.get('user_id', '')}::{record.get('timestamp', '')}"


def _position_to_anchor_score(rank_position: int, num_candidates: int) -> float:
    if num_candidates <= 0 or rank_position > num_candidates:
        return 0.0
    return float((num_candidates - rank_position + 1) / num_candidates)


def build_signal_lookup(
    signal_records: Iterable[dict[str, Any]],
    *,
    user_col: str = "user_id",
    item_col: str = "candidate_item_id",
    signal_score_col: str = "shadow_calibrated_score",
    signal_uncertainty_col: str = "shadow_uncertainty",
) -> tuple[dict[tuple[str, str], dict[str, Any]], float]:
    """Index calibrated winner-signal rows by user and candidate item."""

    score_fields = (
        signal_score_col,
        "shadow_calibrated_score",
        "calibrated_confidence",
        "shadow_score",
        "confidence",
    )
    uncertainty_fields = (
        signal_uncertainty_col,
        "shadow_uncertainty",
        "uncertainty",
    )

    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    uncertainties: list[float] = []

    for record in signal_records:
        user_id = str(record.get(user_col, "")).strip()
        item_id = str(record.get(item_col, "")).strip()
        if not user_id or not item_id:
            continue

        signal_score = _first_float(record, score_fields, default=None)
        if signal_score is None:
            continue

        signal_score = _clamp01(signal_score)
        signal_uncertainty = _first_float(
            record,
            uncertainty_fields,
            default=1.0 - signal_score,
        )
        signal_uncertainty = _clamp01(float(signal_uncertainty))
        uncertainties.append(signal_uncertainty)

        lookup[(user_id, item_id)] = {
            "signal_score": signal_score,
            "signal_uncertainty": signal_uncertainty,
            "source_shadow_variant": record.get("shadow_variant"),
            "source_parse_success": record.get("parse_success"),
        }

    fallback_uncertainty = float(mean(uncertainties)) if uncertainties else 1.0
    return lookup, fallback_uncertainty


def build_shadow_v6_decision(
    *,
    signal_score: float,
    signal_uncertainty: float,
    anchor_score: float,
    matched_signal: bool,
    gate_threshold: float = 0.15,
    uncertainty_threshold: float = 0.65,
    anchor_conflict_penalty: float = 0.5,
) -> dict[str, Any]:
    """Convert one winner-signal row and one anchor score into a v6 decision target."""

    signal_score = _clamp01(signal_score)
    signal_uncertainty = _clamp01(signal_uncertainty)
    anchor_score = _clamp01(anchor_score)
    anchor_disagreement = abs(signal_score - anchor_score)

    if not matched_signal:
        decision = {
            "decision_score": anchor_score,
            "signal_score": signal_score,
            "signal_uncertainty": signal_uncertainty,
            "anchor_score": anchor_score,
            "anchor_disagreement": anchor_disagreement,
            "signal_reliability": 0.0,
            "raw_correction_gate": 0.0,
            "correction_gate": 0.0,
            "fallback_flag": True,
            "fallback_reason": "missing_signal",
            "pair_type": "anchor_preferred_over_shadow",
            "pair_weight": 0.0,
            "reason": "Missing winner-signal row, so the anchor ranking is retained.",
        }
    else:
        signal_reliability = 1.0 - signal_uncertainty
        raw_gate = _clamp01(signal_reliability * (1.0 - float(anchor_conflict_penalty) * anchor_disagreement))
        fallback_flag = signal_uncertainty >= float(uncertainty_threshold) or raw_gate < float(gate_threshold)
        correction_gate = 0.0 if fallback_flag else raw_gate
        decision_score = correction_gate * signal_score + (1.0 - correction_gate) * anchor_score

        if signal_uncertainty >= float(uncertainty_threshold):
            fallback_reason = "high_signal_uncertainty"
        elif raw_gate < float(gate_threshold):
            fallback_reason = "low_correction_gate"
        else:
            fallback_reason = "signal_accepted"

        if fallback_flag or signal_score < anchor_score:
            pair_type = "anchor_preferred_over_shadow"
        else:
            pair_type = "shadow_preferred_over_anchor"

        if fallback_flag:
            weight_basis = max(signal_uncertainty, 1.0 - raw_gate)
        else:
            weight_basis = signal_reliability * correction_gate

        decision = {
            "decision_score": _clamp01(decision_score),
            "signal_score": signal_score,
            "signal_uncertainty": signal_uncertainty,
            "anchor_score": anchor_score,
            "anchor_disagreement": anchor_disagreement,
            "signal_reliability": signal_reliability,
            "raw_correction_gate": raw_gate,
            "correction_gate": correction_gate,
            "fallback_flag": fallback_flag,
            "fallback_reason": fallback_reason,
            "pair_type": pair_type,
            "pair_weight": _clamp01(anchor_disagreement * weight_basis),
            "reason": "The winner signal is accepted." if not fallback_flag else "The anchor is retained by the v6 gate.",
        }

    score_fields = compute_shadow_scores(decision, variant="shadow_v6")
    decision.update(score_fields)
    return decision


def build_shadow_v6_bridge_rows(
    ranking_records: Iterable[dict[str, Any]],
    signal_records: Iterable[dict[str, Any]],
    *,
    winner_signal_variant: str = "shadow_v1",
    user_col: str = "user_id",
    item_col: str = "candidate_item_id",
    signal_score_col: str = "shadow_calibrated_score",
    signal_uncertainty_col: str = "shadow_uncertainty",
    gate_threshold: float = 0.15,
    uncertainty_threshold: float = 0.65,
    anchor_conflict_penalty: float = 0.5,
) -> list[dict[str, Any]]:
    signal_lookup, global_fallback_uncertainty = build_signal_lookup(
        signal_records,
        user_col=user_col,
        item_col=item_col,
        signal_score_col=signal_score_col,
        signal_uncertainty_col=signal_uncertainty_col,
    )

    rows: list[dict[str, Any]] = []
    for record in ranking_records:
        event_id = _event_id(record)
        user_id = str(record.get(user_col, "")).strip()
        candidate_ids = _normalize_item_list(record.get("candidate_item_ids"))
        ranked_ids = _normalize_item_list(record.get("pred_ranked_item_ids")) or _normalize_item_list(record.get("topk_item_ids"))
        if not ranked_ids:
            ranked_ids = candidate_ids

        rank_lookup = {item_id: idx + 1 for idx, item_id in enumerate(ranked_ids)}
        num_candidates = len(candidate_ids)
        positive_item_id = str(record.get("positive_item_id", "")).strip()
        popularity_groups = _normalize_item_list(record.get("candidate_popularity_groups"))
        candidate_titles = _normalize_item_list(record.get("candidate_titles"))
        candidate_texts = _normalize_item_list(record.get("candidate_texts"))
        candidate_labels = record.get("candidate_labels") if isinstance(record.get("candidate_labels"), list) else []

        matched_uncertainties = [
            float(signal_lookup[(user_id, item_id)]["signal_uncertainty"])
            for item_id in candidate_ids
            if (user_id, item_id) in signal_lookup
        ]
        event_fallback_uncertainty = (
            float(mean(matched_uncertainties))
            if matched_uncertainties
            else global_fallback_uncertainty
        )

        for idx, item_id in enumerate(candidate_ids):
            anchor_rank = int(rank_lookup.get(item_id, num_candidates + 1))
            anchor_score = _position_to_anchor_score(anchor_rank, num_candidates)
            signal_entry = signal_lookup.get((user_id, item_id))
            matched_signal = signal_entry is not None

            if signal_entry is None:
                signal_score = anchor_score
                signal_uncertainty = event_fallback_uncertainty
            else:
                signal_score = float(signal_entry["signal_score"])
                signal_uncertainty = float(signal_entry["signal_uncertainty"])

            decision = build_shadow_v6_decision(
                signal_score=signal_score,
                signal_uncertainty=signal_uncertainty,
                anchor_score=anchor_score,
                matched_signal=matched_signal,
                gate_threshold=gate_threshold,
                uncertainty_threshold=uncertainty_threshold,
                anchor_conflict_penalty=anchor_conflict_penalty,
            )

            label = int(str(item_id) == positive_item_id) if positive_item_id else int(_list_value(candidate_labels, idx, 0) or 0)
            rows.append(
                {
                    "shadow_variant": "shadow_v6",
                    "winner_signal_variant": winner_signal_variant,
                    "user_id": user_id,
                    "source_event_id": event_id,
                    "split_name": record.get("split_name"),
                    "timestamp": record.get("timestamp"),
                    "positive_item_id": positive_item_id,
                    "candidate_item_id": str(item_id),
                    "candidate_title": _list_value(candidate_titles, idx, ""),
                    "candidate_text": _list_value(candidate_texts, idx, ""),
                    "candidate_popularity_group": str(_list_value(popularity_groups, idx, "unknown")).lower(),
                    "label": label,
                    "num_candidates": int(num_candidates),
                    "anchor_rank": anchor_rank,
                    "matched_signal": matched_signal,
                    "event_signal_coverage_rate": float(len(matched_uncertainties) / num_candidates) if num_candidates else 0.0,
                    **decision,
                }
            )

    return rows


def rank_shadow_v6_bridge_rows(
    rows: Iterable[dict[str, Any]],
    *,
    group_col: str = "source_event_id",
    score_col: str = "decision_score",
    rank_col: str = "decision_rank",
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_col, ""))].append(dict(row))

    ranked_rows: list[dict[str, Any]] = []
    for _, group_rows in groups.items():
        ordered = sorted(
            group_rows,
            key=lambda row: (
                -float(row.get(score_col, 0.0) or 0.0),
                str(row.get("candidate_item_id", "")),
            ),
        )
        for idx, row in enumerate(ordered, start=1):
            row[rank_col] = idx
            ranked_rows.append(row)
    return ranked_rows


def build_shadow_v6_decision_predictions(
    ranked_rows: Iterable[dict[str, Any]],
    ranking_records: Iterable[dict[str, Any]],
    *,
    topk: int = 10,
) -> list[dict[str, Any]]:
    original_lookup = {_event_id(record): record for record in ranking_records}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked_rows:
        groups[str(row.get("source_event_id", ""))].append(dict(row))

    predictions: list[dict[str, Any]] = []
    for event_id, event_rows in groups.items():
        ordered = sorted(event_rows, key=lambda row: int(row.get("decision_rank", 10**9)))
        first = ordered[0]
        original = original_lookup.get(event_id, {})
        ranked_item_ids = [str(row["candidate_item_id"]) for row in ordered]
        topk_item_ids = ranked_item_ids[:topk]
        original_ranked_ids = _normalize_item_list(original.get("pred_ranked_item_ids"))
        if not original_ranked_ids:
            original_ranked_ids = _normalize_item_list(original.get("topk_item_ids"))

        candidate_scores = {
            str(row["candidate_item_id"]): {
                "anchor_rank": int(row["anchor_rank"]),
                "anchor_score": float(row["anchor_score"]),
                "signal_score": float(row["signal_score"]),
                "signal_uncertainty": float(row["signal_uncertainty"]),
                "anchor_disagreement": float(row["anchor_disagreement"]),
                "correction_gate": float(row["correction_gate"]),
                "fallback_flag": bool(row["fallback_flag"]),
                "pair_type": row["pair_type"],
                "pair_weight": float(row["pair_weight"]),
                "decision_score": float(row["decision_score"]),
                "matched_signal": bool(row["matched_signal"]),
            }
            for row in ordered
        }

        predictions.append(
            {
                "user_id": first["user_id"],
                "source_event_id": event_id,
                "positive_item_id": first.get("positive_item_id"),
                "split_name": first.get("split_name"),
                "timestamp": first.get("timestamp"),
                "candidate_item_ids": _normalize_item_list(original.get("candidate_item_ids")) or ranked_item_ids,
                "candidate_titles": _normalize_item_list(original.get("candidate_titles")),
                "candidate_popularity_groups": _normalize_item_list(original.get("candidate_popularity_groups")),
                "original_pred_ranked_item_ids": original_ranked_ids,
                "pred_ranked_item_ids": ranked_item_ids,
                "topk_item_ids": topk_item_ids,
                "confidence": float(mean([float(row["decision_score"]) for row in ordered])) if ordered else 0.0,
                "parse_success": bool(original.get("parse_success", True)),
                "latency": float(_safe_float(original.get("latency"), default=0.0) or 0.0),
                "contains_out_of_candidate_item": bool(original.get("contains_out_of_candidate_item", False)),
                "out_of_candidate_item_ids": _normalize_item_list(original.get("out_of_candidate_item_ids")),
                "candidate_scores": candidate_scores,
                "shadow_variant": "shadow_v6",
                "winner_signal_variant": first.get("winner_signal_variant", "shadow_v1"),
                "signal_coverage_rate": float(mean([1.0 if row["matched_signal"] else 0.0 for row in ordered])) if ordered else 0.0,
                "uncertainty_coverage_rate": float(mean([1.0 if row["matched_signal"] else 0.0 for row in ordered])) if ordered else 0.0,
                "fallback_rate": float(mean([1.0 if row["fallback_flag"] else 0.0 for row in ordered])) if ordered else 0.0,
                "mean_correction_gate": float(mean([float(row["correction_gate"]) for row in ordered])) if ordered else 0.0,
                "mean_pair_weight": float(mean([float(row["pair_weight"]) for row in ordered])) if ordered else 0.0,
                "local_swap_applied": False,
                "raw_response": original.get("raw_response"),
            }
        )

    return predictions


def summarize_shadow_v6_bridge_rows(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    materialized = list(rows)
    if not materialized:
        return {
            "bridge_rows": 0,
            "matched_signal_rate": 0.0,
            "fallback_rate": 0.0,
            "mean_correction_gate": 0.0,
            "mean_pair_weight": 0.0,
            "mean_signal_uncertainty": 0.0,
            "mean_anchor_disagreement": 0.0,
        }

    return {
        "bridge_rows": float(len(materialized)),
        "matched_signal_rate": float(mean([1.0 if row["matched_signal"] else 0.0 for row in materialized])),
        "fallback_rate": float(mean([1.0 if row["fallback_flag"] else 0.0 for row in materialized])),
        "mean_correction_gate": float(mean([float(row["correction_gate"]) for row in materialized])),
        "mean_pair_weight": float(mean([float(row["pair_weight"]) for row in materialized])),
        "mean_signal_uncertainty": float(mean([float(row["signal_uncertainty"]) for row in materialized])),
        "mean_anchor_disagreement": float(mean([float(row["anchor_disagreement"]) for row in materialized])),
    }
