"""CARE reranking: base listwise utility + confidence reliability − uncertainty − echo + tail recovery."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

VARIANT_ORDER = (
    "original_deepseek",
    "confidence_only",
    "popularity_penalty_only",
    "uncertainty_only",
    "care_full",
)


def sanitize_listwise_ranking(pred: dict[str, Any]) -> dict[str, Any]:
    """Drop hallucinated / duplicate IDs; append missing candidates in slate order (pilot-safe)."""
    out = dict(pred)
    cands = [str(x) for x in out.get("candidate_item_ids", [])]
    pr = [str(x) for x in out.get("predicted_ranking", [])]
    seen: set[str] = set()
    new_pr: list[str] = []
    for x in pr:
        if x in seen or x not in cands:
            continue
        seen.add(x)
        new_pr.append(x)
    for c in cands:
        if c not in seen:
            new_pr.append(c)
            seen.add(c)
    out["predicted_ranking"] = new_pr
    return out


def load_care_rerank_config(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "variants" not in data:
        raise ValueError("care_rerank config must contain 'variants'")
    return data


def _rank_of(predicted_ranking: list[str], item_id: str) -> int:
    rid = str(item_id)
    for i, x in enumerate(predicted_ranking):
        if str(x) == rid:
            return i + 1
    return len(predicted_ranking) + 100


def base_score_from_rank(rank: int) -> float:
    r = max(1, int(rank))
    return float(1.0 / math.log2(r + 1.0))


def verbalized_confidence_per_item(
    global_conf: float,
    rank: int,
    *,
    decay: float,
) -> float:
    """Single verbalized scalar spread down the list (documented limitation)."""
    g = float(np.clip(global_conf, 0.0, 1.0))
    return float(np.clip(g / (1.0 + decay * max(0, rank - 1)), 0.0, 1.0))


def calibrated_or_verbalized(pred: dict[str, Any], conf_i: float) -> float:
    cal = pred.get("calibrated_confidence")
    try:
        if cal is not None and np.isfinite(float(cal)):
            return float(np.clip(float(cal), 0.0, 1.0))
    except (TypeError, ValueError):
        pass
    return conf_i


def stress_unreliability(pred: dict[str, Any], feat: dict[str, Any]) -> float:
    """High when global calibration diagnostics suggest overconfidence-wrong pattern."""
    cf = feat.get("care_features") if isinstance(feat.get("care_features"), dict) else {}
    hc = bool(cf.get("high_confidence_wrong"))
    try:
        g = float(pred.get("raw_confidence") or 0.0)
    except (TypeError, ValueError):
        g = 0.0
    g = float(np.clip(g, 0.0, 1.0))
    corr = bool(pred.get("correctness"))
    tail = 0.15
    if hc:
        tail = max(tail, 0.9)
    if g >= 0.7 and not corr:
        tail = max(tail, 0.75)
    return float(np.clip(tail, 0.0, 1.0))


def echo_risk_item(bucket: str, stress: float, *, blind_popularity: bool) -> float:
    b = str(bucket).lower()
    if blind_popularity:
        return {"head": 1.0, "mid": 0.45, "tail": 0.0}.get(b, 0.25)
    w = {"head": 1.0, "mid": 0.32, "tail": 0.04}.get(b, 0.18)
    return float(w * min(1.0, stress))


def uncertainty_risk_item(pred: dict[str, Any], feat: dict[str, Any], rank: int) -> float:
    cf = feat.get("care_features") if isinstance(feat.get("care_features"), dict) else {}
    try:
        g = float(pred.get("raw_confidence") or 0.0)
    except (TypeError, ValueError):
        g = 0.5
    g = float(np.clip(g, 0.0, 1.0))
    miss = bool(pred.get("missing_confidence")) or bool(cf.get("missing_confidence"))
    uver = cf.get("verbalized_uncertainty_score")
    try:
        u = float(uver) if uver is not None and np.isfinite(float(uver)) else (1.0 - g)
    except (TypeError, ValueError):
        u = 1.0 - g
    if miss:
        u = max(u, 0.62)
    rank_boost = 1.0 + 0.06 * max(0, rank - 1)
    return float(np.clip(u * rank_boost, 0.0, 1.35))


def tail_recovery_bonus(bucket: str, rank: int, global_conf: float) -> float:
    b = str(bucket).lower()
    if b == "head" or rank > 6:
        return 0.0
    try:
        g = float(global_conf)
    except (TypeError, ValueError):
        return 0.0
    if not (0.32 <= g <= 0.9):
        return 0.0
    depth = {"tail": 1.0, "mid": 0.55}.get(b, 0.0)
    prox = (7 - min(rank, 6)) / 6.0
    return float(depth * prox)


def item_bucket(item_id: str, candidates: list[str], buckets: list[str]) -> str:
    try:
        idx = candidates.index(str(item_id))
    except ValueError:
        return "unknown"
    if idx < len(buckets):
        return str(buckets[idx]).lower()
    return "unknown"


def compute_care_components(
    pred: dict[str, Any],
    feat: dict[str, Any],
    item_id: str,
    *,
    global_conf: float,
    decay: float,
    center: float,
) -> dict[str, float]:
    candidates = [str(x) for x in pred.get("candidate_item_ids", [])]
    buckets = [str(x) for x in pred.get("candidate_popularity_buckets", [])]
    pr = [str(x) for x in pred.get("predicted_ranking", [])]
    rank = _rank_of(pr, item_id)
    base = base_score_from_rank(rank)
    conf_raw = verbalized_confidence_per_item(global_conf, rank, decay=decay)
    conf_use = calibrated_or_verbalized(pred, conf_raw)
    stress = stress_unreliability(pred, feat)
    echo_blind = False  # set by caller for popularity_penalty_only
    return {
        "item_id": str(item_id),
        "original_rank": float(rank),
        "base_score": base,
        "confidence_term": float(conf_use - center),
        "confidence_value": conf_use,
        "uncertainty_risk": uncertainty_risk_item(pred, feat, rank),
        "echo_risk": echo_risk_item(item_bucket(item_id, candidates, buckets), stress, blind_popularity=False),
        "echo_risk_blind": echo_risk_item(item_bucket(item_id, candidates, buckets), 1.0, blind_popularity=True),
        "tail_recovery": tail_recovery_bonus(item_bucket(item_id, candidates, buckets), rank, global_conf),
        "_stress": stress,
    }


def total_care_score(components: dict[str, float], w: dict[str, Any]) -> float:
    if w.get("identity"):
        return float(components["base_score"])
    blind = bool(w.get("blind_popularity_ablation"))
    echo = float(components["echo_risk_blind"] if blind else components["echo_risk"])
    return (
        float(components["base_score"])
        + float(w.get("alpha", 0.0)) * float(components["confidence_term"])
        - float(w.get("beta", 0.0)) * float(components["uncertainty_risk"])
        - float(w.get("gamma", 0.0)) * echo
        + float(w.get("delta", 0.0)) * float(components["tail_recovery"])
    )


def rerank_candidates_for_user(
    pred: dict[str, Any],
    feat: dict[str, Any],
    variant: str,
    cfg: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return (new_ranking, score_rows one per candidate)."""
    pred = sanitize_listwise_ranking(pred)
    gconf = float(pred.get("raw_confidence") or 0.5)
    try:
        gconf = float(np.clip(gconf, 0.0, 1.0))
    except (TypeError, ValueError):
        gconf = 0.5
    global_conf = gconf
    decay = float(cfg.get("global", {}).get("confidence_decay", 0.14))
    center = float(cfg.get("global", {}).get("confidence_center", 0.5))
    vw = cfg["variants"][variant]
    candidates = [str(x) for x in pred.get("candidate_item_ids", [])]
    pr = [str(x) for x in pred.get("predicted_ranking", [])]
    if set(candidates) != set(pr) or len(candidates) != len(set(candidates)):
        raise ValueError("predicted_ranking must be a permutation of candidate_item_ids with no duplicates")
    if vw.get("identity"):
        new_rank = list(pr)
        rows: list[dict[str, Any]] = []
        for cid in new_rank:
            comp = compute_care_components(pred, feat, cid, global_conf=global_conf, decay=decay, center=center)
            if bool(vw.get("blind_popularity_ablation")):
                comp["echo_risk"] = comp["echo_risk_blind"]
            rows.append({"user_id": pred.get("user_id"), "item_id": cid, **comp, "care_total": comp["base_score"]})
        return new_rank, rows

    scored: list[tuple[str, float, dict[str, Any], int]] = []
    orig_rank = {c: _rank_of(pr, c) for c in candidates}
    for cid in candidates:
        comp = compute_care_components(pred, feat, cid, global_conf=global_conf, decay=decay, center=center)
        if bool(vw.get("blind_popularity_ablation")):
            comp["echo_risk"] = comp["echo_risk_blind"]
        sc = total_care_score(comp, vw)
        scored.append((cid, sc, comp, orig_rank[cid]))
    scored.sort(key=lambda t: (-t[1], t[3], candidates.index(t[0])))
    new_rank = [t[0] for t in scored]
    rows = []
    for cid, sc, comp, _ in scored:
        rows.append({"user_id": pred.get("user_id"), "item_id": cid, **comp, "care_total": sc})
    return new_rank, rows


def build_reranked_prediction_row(
    pred: dict[str, Any],
    new_ranking: list[str],
    *,
    variant: str,
) -> dict[str, Any]:
    out = dict(pred)
    out["predicted_ranking"] = list(new_ranking)
    out["reranked_item_ids"] = list(new_ranking)
    out["predicted_item_id"] = str(new_ranking[0]) if new_ranking else ""
    tgt = str(out.get("target_item_id", ""))
    out["correctness"] = bool(out["predicted_item_id"] == tgt)
    out["method"] = f"care_rerank_{variant}"
    out["backend"] = "rerank"
    out["run_type"] = "pilot"
    out["is_paper_result"] = False
    out["backend_type"] = "rerank"
    return out


def high_confidence_wrong_top1(pred: dict[str, Any]) -> bool:
    try:
        g = float(pred.get("raw_confidence") or 0.0)
    except (TypeError, ValueError):
        g = 0.0
    g = float(np.clip(g, 0.0, 1.0))
    return bool(g >= 0.7 and not bool(pred.get("correctness")))


def top1_bucket(pred: dict[str, Any]) -> str:
    ranking = [str(x) for x in pred.get("predicted_ranking", []) or pred.get("reranked_item_ids", [])]
    if not ranking:
        return "unknown"
    cands = [str(x) for x in pred.get("candidate_item_ids", [])]
    bks = [str(x) for x in pred.get("candidate_popularity_buckets", [])]
    return item_bucket(ranking[0], cands, bks)


def risk_row_changed_top1(pred_before: dict[str, Any], pred_after: dict[str, Any]) -> bool:
    b0 = [str(x) for x in pred_before.get("predicted_ranking", [])]
    b1 = [str(x) for x in pred_after.get("predicted_ranking", [])]
    if not b0 or not b1:
        return False
    high_risk = high_confidence_wrong_top1(pred_before) or (
        top1_bucket(pred_before) == "head" and float(np.clip(float(pred_before.get("raw_confidence") or 0), 0, 1)) >= 0.72
    )
    return bool(high_risk and b0[0] != b1[0])
