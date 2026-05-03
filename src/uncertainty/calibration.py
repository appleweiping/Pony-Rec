# src/uncertainty/calibration.py

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class SplitResult:
    valid_df: pd.DataFrame
    test_df: pd.DataFrame


def user_level_split(
    df: pd.DataFrame,
    user_col: str = "user_id",
    valid_ratio: float = 0.5,
    random_state: int = 42
) -> SplitResult:
    """
    Split dataframe by user to avoid leakage.
    All samples from one user go to the same split.
    """
    if user_col not in df.columns:
        raise ValueError(f"Column `{user_col}` not found in dataframe.")

    unique_users = sorted(df[user_col].dropna().unique().tolist())
    if len(unique_users) < 2:
        raise ValueError("Need at least 2 unique users for valid/test split.")

    rng = np.random.default_rng(random_state)
    shuffled_users = unique_users.copy()
    rng.shuffle(shuffled_users)

    n_valid_users = max(1, int(round(len(shuffled_users) * valid_ratio)))
    n_valid_users = min(n_valid_users, len(shuffled_users) - 1)

    valid_users = set(shuffled_users[:n_valid_users])
    test_users = set(shuffled_users[n_valid_users:])

    valid_df = df[df[user_col].isin(valid_users)].copy().reset_index(drop=True)
    test_df = df[df[user_col].isin(test_users)].copy().reset_index(drop=True)

    return SplitResult(valid_df=valid_df, test_df=test_df)


class IsotonicCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip"
        )
        self.is_fitted = False

    def fit(self, confidence: np.ndarray, correctness: np.ndarray) -> None:
        x = np.asarray(confidence).astype(float)
        y = np.asarray(correctness).astype(float)
        self.model.fit(x, y)
        self.is_fitted = True

    def predict(self, confidence: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("IsotonicCalibrator is not fitted yet.")
        x = np.asarray(confidence).astype(float)
        return np.clip(self.model.predict(x), 0.0, 1.0)


class ConstantCalibrator:
    """
    Fallback calibrator for degenerate valid splits with a single target value.
    """
    def __init__(self, constant: float):
        self.constant = float(np.clip(constant, 0.0, 1.0))
        self.is_fitted = True

    def fit(self, confidence: np.ndarray, correctness: np.ndarray) -> None:
        return None

    def predict(self, confidence: np.ndarray) -> np.ndarray:
        x = np.asarray(confidence).astype(float)
        return np.full(shape=x.shape, fill_value=self.constant, dtype=float)


class PlattCalibrator:
    """
    Logistic regression over confidence -> correctness
    """
    def __init__(self):
        self.model = LogisticRegression()
        self.is_fitted = False

    def fit(self, confidence: np.ndarray, correctness: np.ndarray) -> None:
        x = np.asarray(confidence).astype(float).reshape(-1, 1)
        y = np.asarray(correctness).astype(int)
        self.model.fit(x, y)
        self.is_fitted = True

    def predict(self, confidence: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("PlattCalibrator is not fitted yet.")
        x = np.asarray(confidence).astype(float).reshape(-1, 1)
        return np.clip(self.model.predict_proba(x)[:, 1], 0.0, 1.0)


def _build_calibrator(method: str):
    method = method.lower()
    if method == "isotonic":
        return IsotonicCalibrator()
    if method == "platt":
        return PlattCalibrator()
    raise ValueError("method must be either 'isotonic' or 'platt'.")


def fit_calibrator(
    valid_df: pd.DataFrame,
    method: str = "isotonic",
    confidence_col: str = "confidence",
    target_col: str = "is_correct"
):
    if confidence_col not in valid_df.columns:
        raise ValueError(f"Column `{confidence_col}` not found in valid_df.")
    if target_col not in valid_df.columns:
        raise ValueError(f"Column `{target_col}` not found in valid_df.")
    if valid_df.empty:
        raise ValueError("valid_df is empty; cannot fit calibrator.")

    x = valid_df[confidence_col].to_numpy()
    y = valid_df[target_col].to_numpy()

    unique_targets = np.unique(y.astype(int))
    if len(unique_targets) < 2:
        return ConstantCalibrator(constant=float(unique_targets[0]))

    calibrator = _build_calibrator(method)
    calibrator.fit(x, y)
    return calibrator


def fit_isotonic_calibrator(
    valid_df: pd.DataFrame,
    confidence_col: str = "confidence",
    target_col: str = "is_correct"
):
    return fit_calibrator(
        valid_df=valid_df,
        method="isotonic",
        confidence_col=confidence_col,
        target_col=target_col,
    )


def fit_platt_calibrator(
    valid_df: pd.DataFrame,
    confidence_col: str = "confidence",
    target_col: str = "is_correct"
):
    return fit_calibrator(
        valid_df=valid_df,
        method="platt",
        confidence_col=confidence_col,
        target_col=target_col,
    )


def apply_calibrator(
    df: pd.DataFrame,
    calibrator,
    input_col: str = "confidence",
    output_col: str = "calibrated_confidence"
) -> pd.DataFrame:
    out = df.copy()
    if input_col not in out.columns:
        raise ValueError(f"Column `{input_col}` not found in dataframe.")

    out[output_col] = calibrator.predict(out[input_col].to_numpy())
    out[output_col] = out[output_col].astype(float).clip(0.0, 1.0)
    return out


def build_split_metadata(valid_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, int]:
    return {
        "num_valid_samples": int(len(valid_df)),
        "num_test_samples": int(len(test_df)),
        "num_valid_users": int(valid_df["user_id"].nunique()) if "user_id" in valid_df.columns else -1,
        "num_test_users": int(test_df["user_id"].nunique()) if "user_id" in test_df.columns else -1,
    }


# =============================================================================
# CARE-Rec diagnostic calibration (numpy). Separate from isotonic / Platt fit.
# =============================================================================


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float, List[Any]]:
    """Equal-width [0,1] bins; delegates to ``src.eval.calibration_metrics``."""
    from src.eval.calibration_metrics import expected_calibration_error as _ece

    return _ece(np.asarray(y_true), np.asarray(y_prob), n_bins=n_bins)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from src.eval.calibration_metrics import brier_score as _brier

    return float(_brier(np.asarray(y_true), np.asarray(y_prob)))


def confidence_correctness_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC that a higher score predicts correctness (no sklearn)."""
    from src.eval.calibration_metrics import roc_auc_score_manual

    return float(roc_auc_score_manual(np.asarray(y_true).astype(int), np.asarray(y_score).astype(float)))


def negative_log_likelihood(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    """Binary NLL; returns NaN if any required probability is invalid."""
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_prob, dtype=float).reshape(-1)
    if len(y) == 0 or len(y) != len(p):
        return float("nan")
    if not np.all(np.isfinite(p)):
        return float("nan")
    p = np.clip(p, eps, 1.0 - eps)
    nll = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.mean(nll))


def adaptive_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Equal-mass (quantile) bins on predicted confidence — 'adaptive' ECE variant."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    n = len(y_true)
    if n < 2 or not np.all(np.isfinite(y_prob)):
        return float("nan")
    order = np.argsort(y_prob)
    ece = 0.0
    bin_size = max(1, n // n_bins)
    for b in range(n_bins):
        start = b * bin_size
        end = n if b == n_bins - 1 else min(n, (b + 1) * bin_size)
        idx = order[start:end]
        if len(idx) == 0:
            continue
        acc = float(np.mean(y_true[idx]))
        conf = float(np.mean(y_prob[idx]))
        w = len(idx) / n
        ece += w * abs(acc - conf)
    return float(ece)


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    """Table-friendly reliability bins (equal-width on [0,1])."""
    _, _, stats = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    out: list[dict[str, float]] = []
    for b in stats:
        gap = float("nan")
        if b.count > 0 and np.isfinite(b.avg_confidence) and np.isfinite(b.accuracy):
            gap = abs(b.accuracy - b.avg_confidence)
        out.append(
            {
                "bin_lower": float(b.bin_lower),
                "bin_upper": float(b.bin_upper),
                "bin_center": float(b.bin_center),
                "count": int(b.count),
                "avg_confidence": float(b.avg_confidence) if np.isfinite(b.avg_confidence) else float("nan"),
                "accuracy": float(b.accuracy) if np.isfinite(b.accuracy) else float("nan"),
                "gap": gap,
            }
        )
    return out


def high_confidence_wrong_rate(confidence: np.ndarray, is_correct: np.ndarray, *, thresh: float = 0.7) -> float:
    c = np.asarray(confidence, dtype=float)
    y = np.asarray(is_correct, dtype=bool)
    mask = np.isfinite(c) & (c >= thresh)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(~y[mask]))


def low_confidence_correct_rate(confidence: np.ndarray, is_correct: np.ndarray, *, thresh: float = 0.4) -> float:
    c = np.asarray(confidence, dtype=float)
    y = np.asarray(is_correct, dtype=bool)
    mask = np.isfinite(c) & (c <= thresh)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(y[mask]))


def group_calibration_by_popularity_bucket(
    rows: list[dict[str, Any]],
    *,
    bucket_key: str = "popularity_bucket",
    confidence_key: str = "confidence",
    correct_key: str = "is_correct_at_1",
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Per-bucket ECE/Brier/sample counts (diagnostic, not paper-claim)."""
    buckets: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        b = str(row.get(bucket_key) or "unknown")
        conf = row.get(confidence_key)
        cor = row.get(correct_key)
        if conf is None or not np.isfinite(float(conf)):
            continue
        if cor is None:
            continue
        buckets.setdefault(b, []).append((float(conf), float(bool(cor))))

    out: list[dict[str, Any]] = []
    for b, pairs in sorted(buckets.items()):
        confs = np.asarray([p[0] for p in pairs], dtype=float)
        y = np.asarray([p[1] for p in pairs], dtype=float)
        ece, _, _ = expected_calibration_error(y, confs, n_bins=min(n_bins, max(2, len(y))))
        out.append(
            {
                "popularity_bucket": b,
                "n": int(len(y)),
                "ece": float(ece),
                "brier": float(brier_score(y, confs)),
                "accuracy": float(np.mean(y)) if len(y) else float("nan"),
                "avg_confidence": float(np.mean(confs)) if len(y) else float("nan"),
                "auroc": float(confidence_correctness_auc(y, confs)) if len(np.unique(y)) > 1 else float("nan"),
            }
        )
    return out


def high_confidence_error_by_bucket(
    rows: list[dict[str, Any]],
    *,
    bucket_key: str = "popularity_bucket",
    confidence_key: str = "confidence",
    correct_key: str = "is_correct_at_1",
    thresh: float = 0.7,
) -> list[dict[str, Any]]:
    agg: dict[str, dict[str, float]] = {}
    for row in rows:
        b = str(row.get(bucket_key) or "unknown")
        conf = row.get(confidence_key)
        cor = row.get(correct_key)
        if conf is None or not np.isfinite(float(conf)):
            continue
        if cor is None:
            continue
        slot = agg.setdefault(b, {"n": 0.0, "high_conf_wrong": 0.0})
        slot["n"] += 1.0
        if float(conf) >= thresh and not bool(cor):
            slot["high_conf_wrong"] += 1.0
    return [
        {
            "popularity_bucket": b,
            "n": int(v["n"]),
            "high_confidence_wrong_count": int(v["high_conf_wrong"]),
            "high_confidence_wrong_rate": float(v["high_conf_wrong"] / v["n"]) if v["n"] else float("nan"),
        }
        for b, v in sorted(agg.items())
    ]


def compare_confidence_sources(
    y_true: np.ndarray,
    sources: dict[str, np.ndarray],
    *,
    n_bins: int = 10,
) -> dict[str, dict[str, float]]:
    """Metrics per confidence channel (finite-masked)."""
    y = np.asarray(y_true, dtype=float).reshape(-1)
    out: dict[str, dict[str, float]] = {}
    for name, conf in sources.items():
        p = np.asarray(conf, dtype=float).reshape(-1)
        if len(p) != len(y):
            out[name] = {"n_valid": 0, "error": "length_mismatch"}
            continue
        mask = np.isfinite(p) & np.isfinite(y)
        if int(mask.sum()) < 2:
            out[name] = {"n_valid": int(mask.sum()), "ece": float("nan"), "brier": float("nan"), "auroc": float("nan")}
            continue
        yt = y[mask]
        pp = np.clip(p[mask], 0.0, 1.0)
        ece, _, _ = expected_calibration_error(yt, pp, n_bins=n_bins)
        out[name] = {
            "n_valid": int(mask.sum()),
            "ece": float(ece),
            "brier": float(brier_score(yt, pp)),
            "auroc": float(confidence_correctness_auc(yt, pp)) if len(np.unique(yt)) > 1 else float("nan"),
            "nll": float(negative_log_likelihood(yt, pp)),
            "adaptive_ece": float(adaptive_ece(yt, pp, n_bins=n_bins)),
        }
    return out


def _entropy_confidence_from_probs(probs: list[float] | None) -> float | None:
    if not probs:
        return None
    arr = np.asarray([float(x) for x in probs if x is not None and np.isfinite(float(x))], dtype=float)
    if arr.size < 2:
        return None
    arr = np.clip(arr, 1e-12, None)
    s = float(arr.sum())
    if s <= 0:
        return None
    p = arr / s
    ent = float(-np.sum(p * np.log(p)))
    ent_norm = ent / math.log(len(p))
    return float(np.clip(1.0 - ent_norm, 0.0, 1.0))


def _margin_confidence_from_probs(probs: list[float] | None) -> float | None:
    if not probs:
        return None
    arr = np.asarray([float(x) for x in probs if x is not None and np.isfinite(float(x))], dtype=float)
    if arr.size < 2:
        return None
    arr = np.clip(arr, 0.0, None)
    s = float(arr.sum())
    if s <= 0:
        return None
    p = sorted([float(x) for x in arr / s], reverse=True)
    margin = float(p[0] - p[1]) if len(p) >= 2 else 0.0
    return float(np.clip(margin, 0.0, 1.0))


def _self_consistency_score(row: dict[str, Any]) -> float | None:
    samples = row.get("self_consistency_predictions") or row.get("self_consistency_samples")
    if not samples:
        return None
    top = row.get("predicted_item_id")
    if top is None:
        return None
    top = str(top)
    hits = sum(1 for x in samples if str(x) == top)
    return float(hits / len(samples))


def build_calibration_diagnostic_rows(
    predictions: list[dict[str, Any]],
    features_by_user: dict[str, dict[str, Any]] | None,
    *,
    confidence_field: str = "auto",
    num_bins: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join optional uncertainty_features rows (by ``user_id``) and emit per-row diagnostics."""
    confidence_field = confidence_field.lower().strip()
    rows_out: list[dict[str, Any]] = []
    chosen_primary: str | None = None

    for row in predictions:
        uid = str(row.get("user_id", ""))
        feat = (features_by_user or {}).get(uid, {})
        care = feat.get("care_features") if isinstance(feat.get("care_features"), dict) else {}

        ranking = list(row.get("predicted_ranking") or row.get("reranked_item_ids") or [])
        target = str(row.get("target_item_id", "") or "")
        pred = str(row.get("predicted_item_id", "") or (ranking[0] if ranking else "") or "")
        is_correct = bool(row.get("correctness", pred == target and bool(target)))
        if ranking and target:
            try:
                tr = ranking.index(target) + 1
            except ValueError:
                tr = None
        else:
            tr = None

        raw_c = row.get("raw_confidence")
        cal_c = row.get("calibrated_confidence")
        try:
            raw_cf = float(raw_c) if raw_c is not None and np.isfinite(float(raw_c)) else None
        except (TypeError, ValueError):
            raw_cf = None
        try:
            cal_cf = float(cal_c) if cal_c is not None and np.isfinite(float(cal_c)) else None
        except (TypeError, ValueError):
            cal_cf = None

        probs = row.get("candidate_probabilities")
        if not isinstance(probs, list):
            probs = None
        ent_c = _entropy_confidence_from_probs(probs) if probs else None
        margin_c = _margin_confidence_from_probs(probs) if probs else None
        sc_c = _self_consistency_score(row)
        parsed_top1p = row.get("top1_probability") or row.get("parsed_top1_probability")
        try:
            parsed_pf = float(parsed_top1p) if parsed_top1p is not None and np.isfinite(float(parsed_top1p)) else None
        except (TypeError, ValueError):
            parsed_pf = None

        rank_score = row.get("top1_score") or row.get("rank_score")
        try:
            rank_sf = float(rank_score) if rank_score is not None and np.isfinite(float(rank_score)) else None
        except (TypeError, ValueError):
            rank_sf = None
        if rank_sf is not None:
            rank_sf = float(np.clip(rank_sf, 0.0, 1.0))

        verbal = care.get("verbalized_raw_confidence", raw_cf)

        def pick_primary() -> tuple[float | None, str]:
            if confidence_field == "verbalized" or confidence_field == "raw_confidence":
                return raw_cf, "verbalized_raw_confidence"
            if confidence_field in ("confidence",):
                return raw_cf if raw_cf is not None else cal_cf, "confidence"
            if confidence_field == "calibrated" or confidence_field == "calibrated_confidence":
                return cal_cf, "calibrated_confidence"
            if confidence_field == "score":
                return rank_sf, "rank_score"
            if confidence_field == "margin":
                return margin_c, "score_margin_normalized"
            if confidence_field == "entropy":
                return ent_c, "entropy_confidence"
            if confidence_field == "self_consistency":
                return sc_c, "self_consistency"
            # auto
            if raw_cf is not None:
                return raw_cf, "verbalized_raw_confidence"
            if cal_cf is not None:
                return cal_cf, "calibrated_confidence"
            if margin_c is not None:
                return margin_c, "score_margin_normalized"
            if ent_c is not None:
                return ent_c, "entropy_confidence"
            if sc_c is not None:
                return sc_c, "self_consistency"
            if rank_sf is not None:
                return rank_sf, "rank_score"
            return 0.5, "default_0.5_missing_all_sources"

        conf, source = pick_primary()
        if chosen_primary is None:
            chosen_primary = source

        conf_f = float(conf) if conf is not None and np.isfinite(float(conf)) else float("nan")
        conf_clip = float(np.clip(conf_f, 0.0, 1.0)) if np.isfinite(conf_f) else float("nan")

        pop = row.get("item_popularity_bucket") or care.get("target_popularity_bucket")
        pop_s = str(pop) if pop is not None else "unknown"
        head = pop_s.lower() == "head"
        tail = pop_s.lower() == "tail"

        hc_wrong = bool(np.isfinite(conf_clip) and conf_clip >= 0.7 and not is_correct)
        lc_right = bool(np.isfinite(conf_clip) and conf_clip <= 0.4 and is_correct)

        cbin = None
        if np.isfinite(conf_clip):
            cbin = int(min(num_bins - 1, max(0, int(conf_clip * num_bins))))

        out_row: dict[str, Any] = {
            "user_id": uid,
            "dataset": row.get("dataset"),
            "domain": row.get("domain"),
            "split": row.get("split"),
            "is_correct_at_1": is_correct,
            "target_rank": tr,
            "confidence": conf_clip if np.isfinite(conf_clip) else None,
            "confidence_source": source,
            "popularity_bucket": pop_s,
            "is_head": head,
            "is_tail": tail,
            "high_confidence_wrong": hc_wrong,
            "low_confidence_correct": lc_right,
            "confidence_bin": cbin,
            "verbalized_raw_confidence": raw_cf,
            "calibrated_confidence": cal_cf,
            "entropy_confidence": ent_c,
            "margin_confidence": margin_c,
            "rank_score": rank_sf,
            "parsed_top1_probability": parsed_pf,
            "self_consistency_score": sc_c,
        }
        rows_out.append(out_row)

    meta = {
        "confidence_field_request": confidence_field,
        "primary_confidence_source_observed": chosen_primary,
        "num_bins": int(num_bins),
        "n_predictions": len(predictions),
        "n_feature_users": len(features_by_user) if features_by_user else 0,
    }
    return rows_out, meta


def summarize_calibration_diagnostics(
    rows: list[dict[str, Any]],
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """JSON summary for ``calibration_summary.json``."""
    used = [r for r in rows if r.get("confidence") is not None and np.isfinite(float(r["confidence"]))]
    if not used:
        return {"n_used": 0, "note": "no_finite_confidence"}

    def _fcol(key: str) -> np.ndarray:
        out: list[float] = []
        for r in used:
            x = r.get(key)
            try:
                xf = float(x) if x is not None else float("nan")
            except (TypeError, ValueError):
                xf = float("nan")
            out.append(xf if np.isfinite(xf) else float("nan"))
        return np.asarray(out, dtype=float)

    c_arr = np.clip(_fcol("confidence"), 0.0, 1.0)
    y_arr = np.asarray([float(bool(r.get("is_correct_at_1"))) for r in used], dtype=float)
    ece, mce, _ = expected_calibration_error(y_arr, c_arr, n_bins=n_bins)
    src = compare_confidence_sources(
        y_arr,
        {
            "primary": c_arr,
            "verbalized": np.clip(_fcol("verbalized_raw_confidence"), 0.0, 1.0),
            "entropy": np.clip(_fcol("entropy_confidence"), 0.0, 1.0),
            "margin": np.clip(_fcol("margin_confidence"), 0.0, 1.0),
            "self_consistency": np.clip(_fcol("self_consistency_score"), 0.0, 1.0),
            "parsed_top1_probability": np.clip(_fcol("parsed_top1_probability"), 0.0, 1.0),
            "rank_score": np.clip(_fcol("rank_score"), 0.0, 1.0),
        },
        n_bins=n_bins,
    )
    return {
        "n_used": len(used),
        "ece": float(ece),
        "mce": float(mce),
        "brier": float(brier_score(y_arr, c_arr)),
        "adaptive_ece": float(adaptive_ece(y_arr, c_arr, n_bins=n_bins)),
        "nll": float(negative_log_likelihood(y_arr, c_arr)),
        "auroc": float(confidence_correctness_auc(y_arr, c_arr)) if len(np.unique(y_arr)) > 1 else float("nan"),
        "high_confidence_wrong_rate": float(high_confidence_wrong_rate(c_arr, y_arr.astype(bool))),
        "low_confidence_correct_rate": float(low_confidence_correct_rate(c_arr, y_arr.astype(bool))),
        "compare_confidence_sources": src,
    }
