from __future__ import annotations

import argparse
import csv
import glob
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.ranking_task_metrics import (
    build_ranking_eval_frame,
    compute_ranking_task_metrics,
)


DOMAIN_SUFFIX = {
    "beauty": "full973",
    "books": "small500",
    "electronics": "small500",
    "movies": "small500",
}

DEFAULT_EXTERNAL_METHODS = (
    "llm2rec_style_qwen3_sasrec",
    "llmesr_style_qwen3_sasrec",
    "llmemb_style_qwen3_sasrec",
    "rlmrec_style_qwen3_graphcl",
    "irllrec_style_qwen3_intent",
    "setrec_style_qwen3_identifier",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run external-only diagnostics: no structured-risk/SRPD method is "
            "used as a candidate method. The optional direct row is only a base "
            "reference for binning and deltas."
        )
    )
    parser.add_argument(
        "--week77_root",
        default="~/projects/uncertainty-llm4rec/export/week7_7_four_domain_final",
        help="Week7.7 export root containing the direct base predictions.",
    )
    parser.add_argument(
        "--external_summary_glob",
        default="outputs/*/tables/same_candidate_external_baseline_summary.csv",
        help="Glob over completed same-candidate external baseline summaries.",
    )
    parser.add_argument("--domains", default="beauty,books,electronics,movies")
    parser.add_argument(
        "--external_methods",
        default=",".join(DEFAULT_EXTERNAL_METHODS),
        help="Comma-separated external methods to include in the external-only set.",
    )
    parser.add_argument("--output_dir", default="outputs/summary/week8_external_only_phenomenon")
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig").to_dict(orient="records")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
            return str(value).strip()
    return str(fallback_idx)


def _parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _week77_direct_path(week77_root: Path, domain: str) -> Path:
    suffix = DOMAIN_SUFFIX[domain]
    structured_exp = week77_root / f"{domain}_qwen3_local_structured_risk_{suffix}"
    return structured_exp / "predictions" / "rank_predictions.jsonl"


def _external_method_paths(summary_glob: str) -> dict[str, dict[str, Path]]:
    paths: dict[str, dict[str, Path]] = {}
    for path_text in sorted(glob.glob(summary_glob)):
        summary_path = Path(path_text).expanduser()
        if not summary_path.exists():
            continue
        for row in _read_csv_rows(summary_path):
            if row.get("status_label") != "same_schema_external_baseline":
                continue
            if row.get("artifact_class") != "completed_result":
                continue
            domain = row.get("domain", "").strip()
            method = row.get("baseline_name", "").strip()
            prediction_path = row.get("prediction_path", "").strip()
            if not domain or not method or not prediction_path:
                continue
            paths.setdefault(domain, {})[method] = Path(prediction_path).expanduser()
    return paths


def _resolve_path(path: Path, *, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root / path


def _positive_rank(record: dict[str, Any]) -> int:
    if "positive_rank" in record and not pd.isna(record.get("positive_rank")):
        return int(record["positive_rank"])
    positive = str(record.get("positive_item_id", "")).strip()
    ranked = _normalize_item_list(record.get("pred_ranked_item_ids"))
    if not ranked:
        ranked = _normalize_item_list(record.get("topk_item_ids"))
    if positive and positive in ranked:
        return ranked.index(positive) + 1
    candidates = _normalize_item_list(record.get("candidate_item_ids"))
    return max(len(candidates), len(ranked)) + 1


def _ndcg(rank: int, k: int) -> float:
    return float(1.0 / np.log2(rank + 1)) if 0 < rank <= k else 0.0


def _mrr(rank: int) -> float:
    return float(1.0 / rank) if rank > 0 else 0.0


def _rank_bin(rank: int) -> str:
    if rank <= 1:
        return "rank_1"
    if rank <= 3:
        return "rank_2_3"
    if rank <= 6:
        return "rank_4_6"
    return "rank_gt_6"


def _positive_popularity(record: dict[str, Any]) -> str:
    positive = str(record.get("positive_item_id", "")).strip()
    candidates = _normalize_item_list(record.get("candidate_item_ids"))
    groups = _normalize_item_list(record.get("candidate_popularity_groups"))
    for idx, item_id in enumerate(candidates):
        if item_id == positive:
            if idx < len(groups) and groups[idx]:
                return str(groups[idx]).strip().lower()
            return "unknown"
    return "not_in_candidate"


def _index_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_event_id(record, idx): record for idx, record in enumerate(records)}


def _rank_lookup(record: dict[str, Any]) -> dict[str, int]:
    ranked = _normalize_item_list(record.get("pred_ranked_item_ids"))
    if not ranked:
        ranked = _normalize_item_list(record.get("topk_item_ids"))
    return {item_id: idx + 1 for idx, item_id in enumerate(ranked)}


def _mean_pairwise_rank_gap(records: list[dict[str, Any]], candidates: list[str]) -> float:
    if len(records) < 2 or not candidates:
        return float("nan")
    rank_maps = [_rank_lookup(record) for record in records]
    fallback = len(candidates) + 1
    gaps = []
    for left, right in combinations(rank_maps, 2):
        item_gaps = [abs(left.get(item, fallback) - right.get(item, fallback)) for item in candidates]
        gaps.append(float(np.mean(item_gaps)))
    return float(np.mean(gaps)) if gaps else float("nan")


def build_external_only_event_rows(
    *,
    domain: str,
    base_records: list[dict[str, Any]],
    external_records: dict[str, list[dict[str, Any]]],
    k: int,
) -> list[dict[str, Any]]:
    base_by_event = _index_records(base_records) if base_records else {}
    external_by_method = {method: _index_records(records) for method, records in external_records.items()}
    common_ids: set[str] = set()
    for rows in external_by_method.values():
        common_ids.update(rows)
    if base_by_event:
        common_ids &= set(base_by_event)

    event_rows: list[dict[str, Any]] = []
    for event_id in sorted(common_ids):
        method_records = {
            method: rows[event_id]
            for method, rows in external_by_method.items()
            if event_id in rows
        }
        if not method_records:
            continue
        base_record = base_by_event.get(event_id) or next(iter(method_records.values()))
        base_rank = _positive_rank(base_record)
        candidates = _normalize_item_list(base_record.get("candidate_item_ids"))

        method_metric_rows = []
        for method, record in method_records.items():
            rank = _positive_rank(record)
            method_metric_rows.append(
                {
                    "method": method,
                    "rank": rank,
                    "ndcg": _ndcg(rank, k),
                    "mrr": _mrr(rank),
                }
            )
        best_event = max(method_metric_rows, key=lambda row: (row["ndcg"], row["mrr"]))
        rank_values = [row["rank"] for row in method_metric_rows]
        top1_values = []
        for record in method_records.values():
            ranked = _normalize_item_list(record.get("pred_ranked_item_ids"))
            if ranked:
                top1_values.append(ranked[0])

        best_single_placeholder = float("nan")
        row = {
            "domain": domain,
            "event_id": event_id,
            "base_rank": base_rank,
            "base_rank_bin": _rank_bin(base_rank),
            f"base_NDCG@{k}": _ndcg(base_rank, k),
            "base_MRR": _mrr(base_rank),
            "positive_popularity_group": _positive_popularity(base_record),
            "external_oracle_method": best_event["method"],
            "external_oracle_rank": best_event["rank"],
            f"external_oracle_NDCG@{k}": best_event["ndcg"],
            "external_oracle_MRR": best_event["mrr"],
            "external_method_count": len(method_records),
            "external_positive_rank_std": float(np.std(rank_values)) if len(rank_values) > 1 else 0.0,
            "external_unique_top1_count": len(set(top1_values)),
            "external_mean_pairwise_rank_gap": _mean_pairwise_rank_gap(list(method_records.values()), candidates),
            f"external_best_single_NDCG@{k}": best_single_placeholder,
            "external_best_single_MRR": best_single_placeholder,
            f"oracle_gain_vs_best_single_NDCG@{k}": best_single_placeholder,
            f"best_single_delta_vs_base_NDCG@{k}": best_single_placeholder,
            f"oracle_delta_vs_base_NDCG@{k}": best_event["ndcg"] - _ndcg(base_rank, k),
        }
        for metric_row in method_metric_rows:
            method = metric_row["method"]
            row[f"{method}_rank"] = metric_row["rank"]
            row[f"{method}_NDCG@{k}"] = metric_row["ndcg"]
            row[f"{method}_MRR"] = metric_row["mrr"]
        event_rows.append(row)
    return event_rows


def _method_metrics(external_records: dict[str, list[dict[str, Any]]], *, domain: str, k: int) -> list[dict[str, Any]]:
    rows = []
    for method, records in sorted(external_records.items()):
        eval_df = build_ranking_eval_frame(pd.DataFrame(records))
        metrics = compute_ranking_task_metrics(eval_df, k=k)
        rows.append({"domain": domain, "method": method, **metrics})
    return rows


def _attach_best_single(event_rows: list[dict[str, Any]], method_metric_rows: list[dict[str, Any]], *, k: int) -> None:
    if not event_rows or not method_metric_rows:
        return

    by_domain: dict[str, str] = {}
    for domain, group in pd.DataFrame(method_metric_rows).groupby("domain"):
        best = group.sort_values([f"NDCG@{k}", "MRR"], ascending=False).iloc[0]
        by_domain[str(domain)] = str(best["method"])

    for row in event_rows:
        method = by_domain.get(str(row["domain"]))
        if not method:
            continue
        best_ndcg = float(row.get(f"{method}_NDCG@{k}", float("nan")))
        best_mrr = float(row.get(f"{method}_MRR", float("nan")))
        base_ndcg = float(row.get(f"base_NDCG@{k}", float("nan")))
        oracle_ndcg = float(row.get(f"external_oracle_NDCG@{k}", float("nan")))
        row["external_best_single_method"] = method
        row[f"external_best_single_NDCG@{k}"] = best_ndcg
        row["external_best_single_MRR"] = best_mrr
        row[f"best_single_delta_vs_base_NDCG@{k}"] = best_ndcg - base_ndcg
        row[f"oracle_gain_vs_best_single_NDCG@{k}"] = oracle_ndcg - best_ndcg


def _summarize_group(df: pd.DataFrame, group_col: str, *, k: int) -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    agg = (
        df.groupby(["domain", group_col], dropna=False)
        .agg(
            event_count=("event_id", "count"),
            base_ndcg=(f"base_NDCG@{k}", "mean"),
            best_single_ndcg=(f"external_best_single_NDCG@{k}", "mean"),
            oracle_ndcg=(f"external_oracle_NDCG@{k}", "mean"),
            best_single_delta_vs_base=(f"best_single_delta_vs_base_NDCG@{k}", "mean"),
            oracle_delta_vs_base=(f"oracle_delta_vs_base_NDCG@{k}", "mean"),
            oracle_gain_vs_best_single=(f"oracle_gain_vs_best_single_NDCG@{k}", "mean"),
            mean_pairwise_rank_gap=("external_mean_pairwise_rank_gap", "mean"),
            positive_rank_std=("external_positive_rank_std", "mean"),
            unique_top1_count=("external_unique_top1_count", "mean"),
        )
        .reset_index()
    )
    return agg


def _add_disagreement_bins(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["external_disagreement_bin"] = "single_or_missing"
    for domain, idx in out.groupby("domain").groups.items():
        values = out.loc[idx, "external_mean_pairwise_rank_gap"].astype(float)
        if values.nunique(dropna=True) < 3:
            out.loc[idx, "external_disagreement_bin"] = "flat"
            continue
        out.loc[idx, "external_disagreement_bin"] = pd.qcut(
            values.rank(method="first"),
            q=3,
            labels=["low_disagreement", "mid_disagreement", "high_disagreement"],
        ).astype(str)
    return out


def _write_csv(df_or_rows: pd.DataFrame | list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df_or_rows, pd.DataFrame):
        df_or_rows.to_csv(path, index=False)
    else:
        pd.DataFrame(df_or_rows).to_csv(path, index=False)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    week77_root = Path(args.week77_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    domains = _parse_csv_list(args.domains)
    requested_external = set(_parse_csv_list(args.external_methods))
    external_paths = _external_method_paths(args.external_summary_glob)

    all_method_metrics: list[dict[str, Any]] = []
    all_event_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    config = {
        "week77_root": str(week77_root),
        "external_summary_glob": args.external_summary_glob,
        "domains": domains,
        "external_methods": sorted(requested_external),
        "k": args.k,
        "note": (
            "This diagnostic excludes structured-risk and SRPD as candidate "
            "methods. The direct row is only a base reference."
        ),
    }
    _write_json(output_dir / "run_config.json", config)

    for domain in domains:
        direct_path = _resolve_path(_week77_direct_path(week77_root, domain), repo_root=repo_root)
        base_records: list[dict[str, Any]] = []
        manifest_rows.append(
            {
                "domain": domain,
                "role": "base_reference_only",
                "method": "direct",
                "path": str(direct_path),
                "exists": direct_path.exists(),
                "rows": "",
            }
        )
        if direct_path.exists():
            base_records = _load_records(direct_path)
            manifest_rows[-1]["rows"] = len(base_records)

        domain_external_paths = {
            method: path
            for method, path in external_paths.get(domain, {}).items()
            if method in requested_external
        }
        external_records: dict[str, list[dict[str, Any]]] = {}
        for method, raw_path in sorted(domain_external_paths.items()):
            path = _resolve_path(raw_path, repo_root=repo_root)
            exists = path.exists()
            manifest_rows.append(
                {
                    "domain": domain,
                    "role": "external_candidate_method",
                    "method": method,
                    "path": str(path),
                    "exists": exists,
                    "rows": "",
                }
            )
            if not exists:
                continue
            records = _load_records(path)
            external_records[method] = records
            manifest_rows[-1]["rows"] = len(records)

        method_rows = _method_metrics(external_records, domain=domain, k=args.k)
        all_method_metrics.extend(method_rows)
        event_rows = build_external_only_event_rows(
            domain=domain,
            base_records=base_records,
            external_records=external_records,
            k=args.k,
        )
        _attach_best_single(event_rows, method_rows, k=args.k)
        all_event_rows.extend(event_rows)

    event_df = _add_disagreement_bins(pd.DataFrame(all_event_rows))
    method_df = pd.DataFrame(all_method_metrics)

    oracle_rows = []
    if not event_df.empty and not method_df.empty:
        for domain, group in event_df.groupby("domain"):
            method_group = method_df[method_df["domain"] == domain]
            best_single = method_group.sort_values([f"NDCG@{args.k}", "MRR"], ascending=False).iloc[0]
            oracle_rows.append(
                {
                    "domain": domain,
                    "external_best_single_method": best_single["method"],
                    f"external_best_single_NDCG@{args.k}": float(best_single[f"NDCG@{args.k}"]),
                    "external_best_single_MRR": float(best_single["MRR"]),
                    f"base_NDCG@{args.k}": float(group[f"base_NDCG@{args.k}"].mean()),
                    f"external_oracle_NDCG@{args.k}": float(group[f"external_oracle_NDCG@{args.k}"].mean()),
                    "external_oracle_MRR": float(group["external_oracle_MRR"].mean()),
                    f"best_single_delta_vs_base_NDCG@{args.k}": float(
                        group[f"best_single_delta_vs_base_NDCG@{args.k}"].mean()
                    ),
                    f"oracle_delta_vs_base_NDCG@{args.k}": float(
                        group[f"oracle_delta_vs_base_NDCG@{args.k}"].mean()
                    ),
                    f"oracle_gain_vs_best_single_NDCG@{args.k}": float(
                        group[f"oracle_gain_vs_best_single_NDCG@{args.k}"].mean()
                    ),
                    "event_count": int(len(group)),
                }
            )

    _write_csv(manifest_rows, output_dir / "input_manifest.csv")
    _write_csv(method_df, output_dir / "external_only_method_metrics.csv")
    _write_csv(event_df, output_dir / "external_only_event_details.csv")
    _write_csv(oracle_rows, output_dir / "external_only_oracle_summary.csv")
    _write_csv(_summarize_group(event_df, "base_rank_bin", k=args.k), output_dir / "external_only_base_rank_bins.csv")
    _write_csv(
        _summarize_group(event_df, "positive_popularity_group", k=args.k),
        output_dir / "external_only_popularity_bins.csv",
    )
    _write_csv(
        _summarize_group(event_df, "external_disagreement_bin", k=args.k),
        output_dir / "external_only_disagreement_bins.csv",
    )

    print(f"Saved external-only method metrics: {output_dir / 'external_only_method_metrics.csv'}")
    print(f"Saved external-only oracle summary: {output_dir / 'external_only_oracle_summary.csv'}")
    print(f"Saved external-only bin diagnostics: {output_dir}")


if __name__ == "__main__":
    main()
