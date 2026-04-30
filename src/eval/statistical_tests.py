from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


RANKING_METRICS = ("HR@10", "NDCG@10", "MRR")


@dataclass(frozen=True)
class PairedTestResult:
    baseline: str
    method: str
    metric: str
    baseline_mean: float
    method_mean: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    holm_p_value: float | None = None
    significant: bool | None = None
    result_label: str = "observed_best"


def _normalize_item_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass
    return [text]


def _event_id(record: dict[str, Any], fallback_idx: int) -> str:
    for key in ("source_event_id", "event_id", "user_id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return str(fallback_idx)


def _ndcg_from_rank(rank: int, k: int) -> float:
    return float(1.0 / np.log2(rank + 1)) if 0 < rank <= k else 0.0


def _mrr_from_rank(rank: int) -> float:
    return float(1.0 / rank) if rank > 0 else 0.0


def _positive_rank_from_record(record: dict[str, Any], k: int) -> int:
    if "positive_rank" in record and not pd.isna(record.get("positive_rank")):
        return int(record["positive_rank"])

    positive_item_id = str(record.get("positive_item_id", "")).strip()
    ranked_ids = _normalize_item_list(record.get("pred_ranked_item_ids"))
    topk_ids = _normalize_item_list(record.get("topk_item_ids"))
    if positive_item_id and positive_item_id in ranked_ids:
        return ranked_ids.index(positive_item_id) + 1
    if positive_item_id and positive_item_id in topk_ids:
        return topk_ids.index(positive_item_id) + 1
    candidate_count = len(_normalize_item_list(record.get("candidate_item_ids")))
    ranked_count = len(ranked_ids) or len(topk_ids)
    return max(candidate_count, ranked_count, k) + 1


def build_event_metric_frame(
    df: pd.DataFrame,
    *,
    method: str,
    k: int = 10,
) -> pd.DataFrame:
    """Normalize ranking outputs to one row per event with paired metrics."""

    if df.empty:
        return pd.DataFrame()

    if {"rank", "label"}.issubset(df.columns):
        id_col = "source_event_id" if "source_event_id" in df.columns else "user_id"
        rows: list[dict[str, Any]] = []
        for event_id, group in df.groupby(id_col, dropna=False):
            ranked = group.sort_values("rank")
            positives = ranked[ranked["label"].astype(int) == 1]
            positive_rank = int(positives["rank"].min()) if not positives.empty else len(ranked) + 1
            topk_items = ranked.head(k)["item_id"].astype(str).tolist() if "item_id" in ranked.columns else []
            candidate_items = ranked["item_id"].astype(str).tolist() if "item_id" in ranked.columns else []
            rows.append(_metric_row(method, str(event_id), positive_rank, topk_items, candidate_items, k, ranked))
        return pd.DataFrame(rows)

    rows = []
    for idx, record in enumerate(df.to_dict(orient="records")):
        positive_rank = _positive_rank_from_record(record, k)
        topk_items = _normalize_item_list(record.get("topk_item_ids"))[:k]
        if not topk_items:
            topk_items = _normalize_item_list(record.get("pred_ranked_item_ids"))[:k]
        candidate_items = _normalize_item_list(record.get("candidate_item_ids"))
        rows.append(
            _metric_row(
                method,
                _event_id(record, idx),
                positive_rank,
                topk_items,
                candidate_items,
                k,
                pd.DataFrame([record]),
            )
        )
    return pd.DataFrame(rows)


def _metric_row(
    method: str,
    event_id: str,
    positive_rank: int,
    topk_items: list[str],
    candidate_items: list[str],
    k: int,
    source_df: pd.DataFrame,
) -> dict[str, Any]:
    parse_col = source_df["parse_success"] if "parse_success" in source_df.columns else pd.Series(dtype=bool)
    ooc_col = (
        source_df["contains_out_of_candidate_item"]
        if "contains_out_of_candidate_item" in source_df.columns
        else pd.Series(dtype=bool)
    )
    return {
        "method": method,
        "event_id": event_id,
        f"HR@{k}": float(positive_rank <= k),
        f"NDCG@{k}": _ndcg_from_rank(positive_rank, k),
        "MRR": _mrr_from_rank(positive_rank),
        "topk_item_ids": topk_items,
        "candidate_item_ids": candidate_items,
        "parse_success": float(parse_col.astype(float).mean()) if len(parse_col) else float("nan"),
        "out_of_candidate": float(ooc_col.astype(float).mean()) if len(ooc_col) else float("nan"),
    }


def aggregate_event_metric(df: pd.DataFrame, metric: str, *, k: int = 10) -> float:
    if df.empty:
        return float("nan")
    if metric == f"coverage@{k}":
        exposed: set[str] = set()
        candidates: set[str] = set()
        for record in df.to_dict(orient="records"):
            exposed.update(_normalize_item_list(record.get("topk_item_ids")))
            candidates.update(_normalize_item_list(record.get("candidate_item_ids")))
        return float(len(exposed) / len(candidates)) if candidates else float("nan")
    if metric in df.columns:
        return float(df[metric].mean())
    raise ValueError(f"Metric not found: {metric}")


def paired_bootstrap_delta(
    baseline_df: pd.DataFrame,
    method_df: pd.DataFrame,
    *,
    metric: str,
    k: int = 10,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> tuple[float, float, float]:
    baseline = baseline_df.set_index("event_id", drop=False)
    method = method_df.set_index("event_id", drop=False)
    common_ids = sorted(set(baseline.index) & set(method.index))
    if not common_ids:
        raise ValueError("No paired event ids shared by baseline and method.")

    baseline = baseline.loc[common_ids]
    method = method.loc[common_ids]
    observed = aggregate_event_metric(method, metric, k=k) - aggregate_event_metric(baseline, metric, k=k)
    rng = np.random.default_rng(random_state)
    deltas: list[float] = []
    for _ in range(n_bootstrap):
        sampled_ids = rng.choice(common_ids, size=len(common_ids), replace=True)
        sample_base = baseline.loc[sampled_ids].reset_index(drop=True)
        sample_method = method.loc[sampled_ids].reset_index(drop=True)
        deltas.append(
            aggregate_event_metric(sample_method, metric, k=k)
            - aggregate_event_metric(sample_base, metric, k=k)
        )
    return (
        float(observed),
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
    )


def paired_permutation_test(
    baseline_df: pd.DataFrame,
    method_df: pd.DataFrame,
    *,
    metric: str,
    n_permutations: int = 2000,
    random_state: int = 42,
) -> float:
    if metric not in baseline_df.columns or metric not in method_df.columns:
        return float("nan")
    baseline = baseline_df.set_index("event_id")
    method = method_df.set_index("event_id")
    common_ids = sorted(set(baseline.index) & set(method.index))
    if not common_ids:
        raise ValueError("No paired event ids shared by baseline and method.")

    diffs = method.loc[common_ids, metric].to_numpy(dtype=float) - baseline.loc[common_ids, metric].to_numpy(dtype=float)
    observed = abs(float(np.mean(diffs)))
    rng = np.random.default_rng(random_state)
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=len(diffs), replace=True)
        if abs(float(np.mean(diffs * signs))) >= observed:
            count += 1
    return float((count + 1) / (n_permutations + 1))


def holm_bonferroni(p_values: list[float], *, alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    indexed = [(idx, p) for idx, p in enumerate(p_values)]
    finite = [(idx, p) for idx, p in indexed if not pd.isna(p)]
    adjusted = [float("nan")] * len(p_values)
    rejected = [False] * len(p_values)
    m = len(finite)
    running_max = 0.0
    for rank, (idx, p_value) in enumerate(sorted(finite, key=lambda item: item[1]), start=1):
        adj = min(1.0, (m - rank + 1) * float(p_value))
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
        rejected[idx] = running_max < alpha
    return adjusted, rejected


def compare_method_frames(
    method_frames: dict[str, pd.DataFrame],
    *,
    baselines: tuple[str, ...] = ("direct", "structured_risk"),
    k: int = 10,
    n_bootstrap: int = 2000,
    n_permutations: int = 2000,
    random_state: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    metrics = (f"HR@{k}", f"NDCG@{k}", "MRR", f"coverage@{k}")
    rows: list[dict[str, Any]] = []
    for baseline_name in baselines:
        if baseline_name not in method_frames:
            continue
        baseline_df = method_frames[baseline_name]
        for method_name, method_df in method_frames.items():
            if method_name == baseline_name:
                continue
            for metric in metrics:
                delta, ci_low, ci_high = paired_bootstrap_delta(
                    baseline_df,
                    method_df,
                    metric=metric,
                    k=k,
                    n_bootstrap=n_bootstrap,
                    random_state=random_state,
                )
                p_value = paired_permutation_test(
                    baseline_df,
                    method_df,
                    metric=metric,
                    n_permutations=n_permutations,
                    random_state=random_state,
                )
                rows.append(
                    {
                        "baseline": baseline_name,
                        "method": method_name,
                        "metric": metric,
                        "baseline_mean": aggregate_event_metric(baseline_df, metric, k=k),
                        "method_mean": aggregate_event_metric(method_df, metric, k=k),
                        "delta": delta,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_value": p_value,
                    }
                )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    adjusted, rejected = holm_bonferroni(result["p_value"].astype(float).tolist(), alpha=alpha)
    result["holm_p_value"] = adjusted
    result["significant"] = rejected
    result["result_label"] = result.apply(
        lambda row: "winner"
        if bool(row["significant"]) and float(row["ci_low"]) > 0.0 and float(row["delta"]) > 0.0
        else "observed_best",
        axis=1,
    )
    result["status_label"] = "completed_result"
    return result


def build_main_table_with_ci(
    method_frames: dict[str, pd.DataFrame],
    significance_df: pd.DataFrame,
    *,
    direct_name: str = "direct",
    k: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method_name, frame in method_frames.items():
        direct_rows = significance_df[
            (significance_df["baseline"] == direct_name)
            & (significance_df["method"] == method_name)
        ] if not significance_df.empty else pd.DataFrame()
        metric_lookup = {row["metric"]: row for _, row in direct_rows.iterrows()}
        ndcg_key = f"NDCG@{k}"
        hr_key = f"HR@{k}"
        coverage_key = f"coverage@{k}"
        ndcg_delta = metric_lookup.get(ndcg_key, {}).get("delta", 0.0) if method_name != direct_name else 0.0
        ndcg_ci_low = metric_lookup.get(ndcg_key, {}).get("ci_low", 0.0) if method_name != direct_name else 0.0
        ndcg_ci_high = metric_lookup.get(ndcg_key, {}).get("ci_high", 0.0) if method_name != direct_name else 0.0
        rows.append(
            {
                "Method": method_name,
                ndcg_key: aggregate_event_metric(frame, ndcg_key, k=k),
                "delta_vs_direct_ndcg": ndcg_delta,
                "delta_vs_direct_95ci": f"[{ndcg_ci_low:.6f}, {ndcg_ci_high:.6f}]",
                "p_value_vs_direct_ndcg": metric_lookup.get(ndcg_key, {}).get("holm_p_value", 0.0)
                if method_name != direct_name
                else 0.0,
                "MRR": aggregate_event_metric(frame, "MRR", k=k),
                hr_key: aggregate_event_metric(frame, hr_key, k=k),
                coverage_key: aggregate_event_metric(frame, coverage_key, k=k),
                "OOC": float(frame["out_of_candidate"].mean()) if "out_of_candidate" in frame.columns else float("nan"),
                "Parse": float(frame["parse_success"].mean()) if "parse_success" in frame.columns else float("nan"),
                "winner_label": metric_lookup.get(ndcg_key, {}).get("result_label", "baseline")
                if method_name != direct_name
                else "baseline",
                "status_label": "completed_result",
            }
        )
    return pd.DataFrame(rows)
