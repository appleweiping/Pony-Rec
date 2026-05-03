"""CARE-aware training sample construction for LoRA debug (pilot, not paper)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.methods.care_rerank import (
    compute_care_components,
    item_bucket,
    load_care_rerank_config,
    stress_unreliability,
)

POLICIES = (
    "vanilla_lora_baseline",
    "prune_high_uncertainty",
    "downweight_high_confidence_wrong_risk",
    "tail_safe_recovery",
    "CARE_full_training",
)

DEFAULT_DECAY = 0.14
DEFAULT_CENTER = 0.5


def teacher_listwise_response(target_item_id: str, candidate_item_ids: list[str], *, confidence: float = 0.88) -> str:
    tid = str(target_item_id)
    order = [tid] + [str(x) for x in candidate_item_ids if str(x) != tid]
    return json.dumps(
        {
            "ranked_item_ids": order,
            "topk_item_ids": order[: min(5, len(order))],
            "confidence": float(confidence),
            "reason": "teacher_rank_target_first",
        },
        ensure_ascii=False,
    )


def _target_bucket(row: dict[str, Any]) -> str:
    b = row.get("target_popularity_bucket")
    if b is not None and str(b).strip():
        return str(b).lower()
    buckets = [str(x).lower() for x in row.get("candidate_popularity_buckets", [])]
    cids = [str(x) for x in row.get("candidate_item_ids", [])]
    tid = str(row.get("target_item_id", ""))
    return item_bucket(tid, cids, buckets)


def _bucket_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = {"head": 0, "mid": 0, "tail": 0, "unknown": 0}
    for row in rows:
        b = _target_bucket(row)
        counts[b] = counts.get(b, 0) + 1
    return counts


def _normalize_buckets(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values()) or 1
    return {k: counts.get(k, 0) / total for k in ("head", "mid", "tail", "unknown")}


@dataclass
class PolicyOutcome:
    keep: bool
    sample_weight: float
    reason: str
    care_risk_features: dict[str, Any] = field(default_factory=dict)
    components: dict[str, float] = field(default_factory=dict)


def _pred_feat(
    row: dict[str, Any],
    pred: dict[str, Any] | None,
    feat: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    p = dict(pred) if pred else {}
    f = dict(feat) if feat else {}
    if not p and row:
        p = {
            "user_id": row.get("user_id"),
            "candidate_item_ids": row.get("candidate_item_ids", []),
            "candidate_popularity_buckets": row.get("candidate_popularity_buckets", []),
            "predicted_ranking": [],
            "raw_confidence": 0.5,
            "missing_confidence": True,
            "is_valid": False,
            "correctness": False,
        }
    if "predicted_ranking" not in p or not p["predicted_ranking"]:
        tid = str(row.get("target_item_id", ""))
        cids = [str(x) for x in row.get("candidate_item_ids", [])]
        p["predicted_ranking"] = [tid] + [c for c in cids if c != tid] if tid in cids else list(cids)
    if "candidate_item_ids" not in p:
        p["candidate_item_ids"] = list(row.get("candidate_item_ids", []))
    if "candidate_popularity_buckets" not in p:
        p["candidate_popularity_buckets"] = list(row.get("candidate_popularity_buckets", []))
    return p, f


def _global_conf(pred: dict[str, Any]) -> float:
    try:
        return float(pred.get("raw_confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _rank_of_target(pred: dict[str, Any], target: str) -> int:
    pr = [str(x) for x in pred.get("predicted_ranking", [])]
    for i, x in enumerate(pr):
        if x == str(target):
            return i + 1
    return len(pr) + 50


def evaluate_policies(
    row: dict[str, Any],
    pred: dict[str, Any] | None,
    feat: dict[str, Any] | None,
    *,
    care_yaml: dict[str, Any] | None = None,
) -> dict[str, PolicyOutcome]:
    p, f = _pred_feat(row, pred, feat)
    tid = str(row.get("target_item_id", ""))
    gconf = _global_conf(p)
    decay = float((care_yaml or {}).get("global", {}).get("confidence_decay", DEFAULT_DECAY))
    center = float((care_yaml or {}).get("global", {}).get("confidence_center", DEFAULT_CENTER))
    vfull = ((care_yaml or {}).get("variants") or {}).get("care_full", {})
    alpha = float(vfull.get("alpha", 0.35))
    beta = float(vfull.get("beta", 0.42))
    gamma = float(vfull.get("gamma", 0.48))
    delta = float(vfull.get("delta", 0.18))

    comps = compute_care_components(
        p,
        f,
        tid,
        global_conf=gconf,
        decay=decay,
        center=center,
    )
    stress = stress_unreliability(p, f)
    ur = float(comps["uncertainty_risk"])

    # Reliability weight: high when uncertainty risk is low
    rel = float(max(0.12, min(1.35, 1.05 - 0.55 * min(ur, 1.2))))
    echo = float(comps["echo_risk"])
    echo_down = float(1.0 / (1.0 + gamma * echo))
    tail_w = float(1.0 + delta * float(comps["tail_recovery"]))

    care_risk = {
        "stress": stress,
        "uncertainty_risk": ur,
        "echo_risk": echo,
        "high_confidence_wrong": bool((f.get("care_features") or {}).get("high_confidence_wrong")),
        "head_top1_wrong": bool((f.get("care_features") or {}).get("head_top1_wrong")),
        "missing_confidence": bool(p.get("missing_confidence")),
        "is_valid": bool(p.get("is_valid", True)),
        "target_bucket": _target_bucket(row),
        "raw_confidence": gconf,
    }

    def vanilla() -> PolicyOutcome:
        return PolicyOutcome(
            keep=True,
            sample_weight=1.0,
            reason="vanilla_keep_all",
            care_risk_features=dict(care_risk),
            components={"base": float(comps["base_score"])},
        )

    def prune() -> PolicyOutcome:
        reasons: list[str] = []
        if not care_risk["is_valid"]:
            reasons.append("invalid_prediction")
        if care_risk["missing_confidence"] or gconf < 0.05:
            reasons.append("missing_or_tiny_confidence")
        if ur > 1.12:
            reasons.append("very_high_uncertainty_risk")
        if not reasons and gconf < 0.18 and not p.get("correctness", True):
            reasons.append("low_reliability_wrong")
        if reasons:
            return PolicyOutcome(
                keep=False,
                sample_weight=0.0,
                reason="prune:" + ",".join(reasons),
                care_risk_features=dict(care_risk),
                components={"uncertainty_risk": ur},
            )
        return PolicyOutcome(
            keep=True,
            sample_weight=1.0,
            reason="prune_keep",
            care_risk_features=dict(care_risk),
            components={"uncertainty_risk": ur},
        )

    def downweight_hc() -> PolicyOutcome:
        w = 1.0
        if care_risk["high_confidence_wrong"] and gconf >= 0.7:
            w *= 0.42
        if care_risk["head_top1_wrong"] or (_target_bucket(row) == "head" and not p.get("correctness") and gconf >= 0.65):
            w *= 0.78
        reason = "downweight_hc" if w < 1.0 else "downweight_neutral"
        return PolicyOutcome(
            keep=True,
            sample_weight=float(max(0.08, min(1.5, w))),
            reason=reason,
            care_risk_features=dict(care_risk),
            components={"weight": w},
        )

    def tail_safe() -> PolicyOutcome:
        b = _target_bucket(row)
        w = 1.0
        if b in {"tail", "mid"} and 0.32 <= gconf <= 0.9 and int(comps["original_rank"]) <= 8:
            w *= 1.12
        return PolicyOutcome(
            keep=True,
            sample_weight=float(max(0.1, min(1.45, w))),
            reason="tail_safe_boost" if w > 1.0 else "tail_safe_neutral",
            care_risk_features=dict(care_risk),
            components={"tail_recovery": float(comps["tail_recovery"])},
        )

    def care_full() -> PolicyOutcome:
        conf_term = alpha * float(comps["confidence_term"])
        unc_pen = beta * ur
        echo_pen = gamma * echo
        tail_b = delta * float(comps["tail_recovery"])
        # Expected-utility style scalar on sample weight (not additive logits)
        util = float(comps["base_score"]) + conf_term - unc_pen - echo_pen + tail_b
        w = float(math.exp(util) / (1.0 + math.exp(util)))  # squash to (0,1)
        w = 0.25 + 1.1 * w
        w *= rel * echo_down * tail_w
        w = float(max(0.06, min(1.55, w)))
        pr = prune()
        if not pr.keep:
            return PolicyOutcome(
                keep=False,
                sample_weight=0.0,
                reason="care_full_pruned:" + pr.reason.split(":", 1)[-1],
                care_risk_features=dict(care_risk),
                components={
                    "util": util,
                    "reliability": rel,
                    "echo_down": echo_down,
                    "tail_factor": tail_w,
                },
            )
        return PolicyOutcome(
            keep=True,
            sample_weight=w,
            reason="care_full_weighted",
            care_risk_features=dict(care_risk),
            components={
                "util": util,
                "reliability": rel,
                "echo_down": echo_down,
                "tail_factor": tail_w,
                "confidence_term": conf_term,
            },
        )

    return {
        "vanilla_lora_baseline": vanilla(),
        "prune_high_uncertainty": prune(),
        "downweight_high_confidence_wrong_risk": downweight_hc(),
        "tail_safe_recovery": tail_safe(),
        "CARE_full_training": care_full(),
    }


def build_training_sample(
    *,
    row: dict[str, Any],
    prompt: str,
    pred: dict[str, Any] | None,
    feat: dict[str, Any] | None,
    policy: str,
    outcome: PolicyOutcome,
    source_paths: dict[str, str],
) -> dict[str, Any]:
    tid = str(row.get("target_item_id", ""))
    gconf = _global_conf(pred or {})
    conf_teacher = 0.88 if outcome.keep else gconf
    return {
        "user_id": str(row.get("user_id", "")),
        "domain": "amazon_beauty",
        "split_source": str(row.get("_split_source", "valid")),
        "prompt": prompt,
        "response": teacher_listwise_response(tid, [str(x) for x in row.get("candidate_item_ids", [])], confidence=conf_teacher),
        "target_item_id": tid,
        "candidate_item_ids": [str(x) for x in row.get("candidate_item_ids", [])],
        "original_rank": int(_rank_of_target(_pred_feat(row, pred, feat)[0], tid)),
        "confidence": float(gconf),
        "popularity_bucket": _target_bucket(row),
        "care_risk_features": outcome.care_risk_features,
        "sample_weight": float(outcome.sample_weight),
        "policy": policy,
        "keep": bool(outcome.keep),
        "keep_drop_reason": outcome.reason,
        "source_paths": source_paths,
        "run_type": "pilot",
        "is_paper_result": False,
        "care_components": outcome.components,
    }


def summarize_policy_run(
    rows: list[dict[str, Any]],
    outcomes_per_policy: dict[str, list[PolicyOutcome]],
) -> dict[str, Any]:
    before = _normalize_buckets(_bucket_counts(rows))
    after: dict[str, Any] = {}
    for pol, outs in outcomes_per_policy.items():
        kept_rows = [rows[i] for i, o in enumerate(outs) if o.keep]
        after[pol] = _normalize_buckets(_bucket_counts(kept_rows)) if kept_rows else {k: 0.0 for k in before}
    counts = {pol: {"kept": sum(1 for o in outs if o.keep), "dropped": sum(1 for o in outs if not o.keep)} for pol, outs in outcomes_per_policy.items()}
    return {"bucket_before": before, "bucket_after": after, "counts": counts}


def load_care_yaml(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return load_care_rerank_config(p)
