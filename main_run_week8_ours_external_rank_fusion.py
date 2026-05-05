from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.eval.ranking_task_metrics import (
    build_ranking_eval_frame,
    compute_ranking_exposure_distribution,
    compute_ranking_task_metrics,
)
from src.utils.io import save_jsonl


DOMAIN_SUFFIX = {
    "beauty": "full973",
    "books": "small500",
    "electronics": "small500",
    "movies": "small500",
}

DEFAULT_SRPD_BEST = {
    "beauty": "srpd_v2",
    "books": "srpd_v2",
    "electronics": "srpd_v5",
    "movies": "srpd_v4",
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
            "Build diagnostic rank-fusion rows that combine our Week7.7 "
            "framework predictions with completed same-candidate external "
            "paper-project baselines."
        )
    )
    parser.add_argument(
        "--week77_root",
        default="~/projects/uncertainty-llm4rec/export/week7_7_four_domain_final",
        help="Week7.7 export root containing direct, structured-risk, and SRPD predictions.",
    )
    parser.add_argument(
        "--external_summary_glob",
        default="outputs/*/tables/same_candidate_external_baseline_summary.csv",
        help="Glob over completed same-candidate external baseline summaries.",
    )
    parser.add_argument("--domains", default="beauty,books,electronics,movies")
    parser.add_argument(
        "--ours_methods",
        default="structured_risk,srpd_best",
        help="Comma-separated methods from {direct,structured_risk,srpd_best}.",
    )
    parser.add_argument(
        "--external_methods",
        default=",".join(DEFAULT_EXTERNAL_METHODS),
        help="Comma-separated external baseline names to fuse with ours.",
    )
    parser.add_argument(
        "--weights",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help=(
            "Comma-separated weights on our method. 0.0 is external-only, "
            "1.0 is ours-only. Sweeping weights on test is diagnostic."
        ),
    )
    parser.add_argument("--output_dir", default="outputs/summary/week8_ours_external_rank_fusion")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--save_all_predictions",
        action="store_true",
        help="Save predictions for every pair/weight instead of only each pair's best diagnostic row.",
    )
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


def _slug(value: str) -> str:
    return (
        str(value)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(".", "_")
        .replace("-", "_")
    )


def _weight_slug(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _parse_weights(text: str) -> list[float]:
    weights = []
    for item in _parse_csv_list(text):
        value = float(item)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Fusion weight must be in [0, 1], got {value}")
        weights.append(value)
    if not weights:
        raise ValueError("At least one fusion weight is required.")
    return weights


def _week77_method_paths(week77_root: Path, domain: str) -> dict[str, Path]:
    suffix = DOMAIN_SUFFIX[domain]
    structured_exp = week77_root / f"{domain}_qwen3_local_structured_risk_{suffix}"
    srpd_variant = DEFAULT_SRPD_BEST[domain]
    srpd_exp = week77_root / f"{domain}_qwen3_rank_{srpd_variant}_{suffix}"
    return {
        "direct": structured_exp / "predictions" / "rank_predictions.jsonl",
        "structured_risk": structured_exp / "reranked" / "rank_reranked.jsonl",
        f"srpd_best_{srpd_variant}": srpd_exp / "predictions" / "rank_predictions.jsonl",
    }


def _resolve_ours_aliases(methods: list[str], domain: str) -> list[str]:
    resolved = []
    srpd_name = f"srpd_best_{DEFAULT_SRPD_BEST[domain]}"
    for method in methods:
        if method == "srpd_best":
            resolved.append(srpd_name)
        else:
            resolved.append(method)
    return resolved


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


def _index_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_event_id(record, idx): record for idx, record in enumerate(records)}


def _rank_to_borda_scores(record: dict[str, Any]) -> dict[str, float]:
    candidates = _normalize_item_list(record.get("candidate_item_ids"))
    ranked = _normalize_item_list(record.get("pred_ranked_item_ids"))
    if not ranked:
        ranked = _normalize_item_list(record.get("topk_item_ids"))

    n = len(candidates) or len(ranked)
    if n <= 1:
        return {item_id: 1.0 for item_id in candidates or ranked}

    fallback_rank = n
    rank_lookup = {item_id: idx for idx, item_id in enumerate(ranked)}
    scores: dict[str, float] = {}
    for item_id in candidates:
        rank = rank_lookup.get(item_id, fallback_rank)
        scores[item_id] = float((n - rank) / n)
    return scores


def fuse_ranked_item_ids(
    ours_record: dict[str, Any],
    external_record: dict[str, Any],
    *,
    ours_weight: float,
) -> list[str]:
    """Fuse two complete candidate rankings with normalized Borda scores."""

    candidates = _normalize_item_list(ours_record.get("candidate_item_ids"))
    if not candidates:
        candidates = _normalize_item_list(external_record.get("candidate_item_ids"))
    ours_scores = _rank_to_borda_scores(ours_record)
    external_scores = _rank_to_borda_scores(external_record)
    original_index = {item_id: idx for idx, item_id in enumerate(candidates)}

    external_weight = 1.0 - ours_weight
    combined = []
    for item_id in candidates:
        score = ours_weight * ours_scores.get(item_id, 0.0) + external_weight * external_scores.get(item_id, 0.0)
        combined.append((item_id, score, original_index.get(item_id, len(original_index))))
    return [item_id for item_id, _, _ in sorted(combined, key=lambda item: (-item[1], item[2]))]


def build_fusion_predictions(
    ours_records: list[dict[str, Any]],
    external_records: list[dict[str, Any]],
    *,
    ours_method: str,
    external_method: str,
    ours_weight: float,
    k: int,
) -> list[dict[str, Any]]:
    ours_by_event = _index_records(ours_records)
    external_by_event = _index_records(external_records)
    common_ids = sorted(set(ours_by_event) & set(external_by_event))
    predictions: list[dict[str, Any]] = []

    for event_id in common_ids:
        ours_record = ours_by_event[event_id]
        external_record = external_by_event[event_id]
        ranked_ids = fuse_ranked_item_ids(ours_record, external_record, ours_weight=ours_weight)
        prediction = dict(ours_record)
        prediction["pred_ranked_item_ids"] = ranked_ids
        prediction["topk_item_ids"] = ranked_ids[:k]
        prediction["parse_success"] = True
        prediction["latency"] = 0.0
        prediction["confidence"] = -1.0
        prediction["contains_out_of_candidate_item"] = False
        prediction["raw_response"] = (
            f"rank_fusion:{ours_method}+{external_method}:ours_weight={ours_weight:.2f}"
        )
        prediction["fusion_ours_method"] = ours_method
        prediction["fusion_external_method"] = external_method
        prediction["fusion_ours_weight"] = ours_weight
        predictions.append(prediction)
    return predictions


def _metrics_for_predictions(predictions: list[dict[str, Any]], *, k: int) -> tuple[dict[str, float], pd.DataFrame]:
    eval_df = build_ranking_eval_frame(pd.DataFrame(predictions))
    metrics = compute_ranking_task_metrics(eval_df, k=k)
    return metrics, eval_df


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _method_metric_rows(
    records_by_method: dict[str, list[dict[str, Any]]],
    *,
    domain: str,
    role: str,
    k: int,
) -> list[dict[str, Any]]:
    rows = []
    for method, records in sorted(records_by_method.items()):
        metrics, _ = _metrics_for_predictions(records, k=k)
        rows.append(
            {
                "domain": domain,
                "role": role,
                "method": method,
                "method_a": "",
                "method_b": "",
                "ours_weight": "",
                "is_test_sweep_best": False,
                **metrics,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    week77_root = Path(args.week77_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    domains = _parse_csv_list(args.domains)
    requested_ours = _parse_csv_list(args.ours_methods)
    requested_external = set(_parse_csv_list(args.external_methods))
    weights = _parse_weights(args.weights)
    external_paths = _external_method_paths(args.external_summary_glob)

    all_metric_rows: list[dict[str, Any]] = []
    pair_summary_rows: list[dict[str, Any]] = []
    best_domain_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    config = {
        "week77_root": str(week77_root),
        "external_summary_glob": args.external_summary_glob,
        "domains": domains,
        "ours_methods": requested_ours,
        "external_methods": sorted(requested_external),
        "weights": weights,
        "k": args.k,
        "note": "Sweeping weights on the test set is a diagnostic/upper-bound search, not a main paper claim.",
    }
    _write_json(output_dir / "run_config.json", config)

    for domain in domains:
        ours_paths = _week77_method_paths(week77_root, domain)
        ours_methods = _resolve_ours_aliases(requested_ours, domain)
        domain_external_paths = {
            method: path
            for method, path in external_paths.get(domain, {}).items()
            if method in requested_external
        }

        ours_records: dict[str, list[dict[str, Any]]] = {}
        external_records: dict[str, list[dict[str, Any]]] = {}

        for role, method_paths, target in [
            ("ours", {m: ours_paths[m] for m in ours_methods if m in ours_paths}, ours_records),
            ("external", domain_external_paths, external_records),
        ]:
            for method, raw_path in sorted(method_paths.items()):
                path = _resolve_path(raw_path, repo_root=repo_root)
                exists = path.exists()
                manifest_rows.append(
                    {
                        "domain": domain,
                        "role": role,
                        "method": method,
                        "path": str(path),
                        "exists": exists,
                        "rows": "",
                    }
                )
                if not exists:
                    continue
                records = _load_records(path)
                target[method] = records
                manifest_rows[-1]["rows"] = len(records)

        all_metric_rows.extend(_method_metric_rows(ours_records, domain=domain, role="ours", k=args.k))
        all_metric_rows.extend(_method_metric_rows(external_records, domain=domain, role="external", k=args.k))

        domain_pair_best: list[dict[str, Any]] = []
        for ours_method, ours_method_records in sorted(ours_records.items()):
            ours_metrics, _ = _metrics_for_predictions(ours_method_records, k=args.k)
            ours_ndcg = float(ours_metrics.get(f"NDCG@{args.k}", float("nan")))
            for external_method, external_method_records in sorted(external_records.items()):
                external_metrics, _ = _metrics_for_predictions(external_method_records, k=args.k)
                external_ndcg = float(external_metrics.get(f"NDCG@{args.k}", float("nan")))
                best_constituent_ndcg = max(ours_ndcg, external_ndcg)

                weight_rows: list[dict[str, Any]] = []
                weight_predictions: dict[float, list[dict[str, Any]]] = {}
                for weight in weights:
                    predictions = build_fusion_predictions(
                        ours_method_records,
                        external_method_records,
                        ours_method=ours_method,
                        external_method=external_method,
                        ours_weight=weight,
                        k=args.k,
                    )
                    if not predictions:
                        continue
                    metrics, eval_df = _metrics_for_predictions(predictions, k=args.k)
                    ndcg = float(metrics.get(f"NDCG@{args.k}", float("nan")))
                    row = {
                        "domain": domain,
                        "role": "ours_external_rank_fusion",
                        "method": (
                            f"fusion_rank_{ours_method}__{external_method}__ours_w{_weight_slug(weight)}"
                        ),
                        "method_a": ours_method,
                        "method_b": external_method,
                        "ours_weight": weight,
                        "is_test_sweep_best": False,
                        "ours_NDCG@10": ours_ndcg,
                        "external_NDCG@10": external_ndcg,
                        "best_constituent_NDCG@10": best_constituent_ndcg,
                        "delta_vs_ours_NDCG@10": ndcg - ours_ndcg,
                        "delta_vs_external_NDCG@10": ndcg - external_ndcg,
                        "delta_vs_best_constituent_NDCG@10": ndcg - best_constituent_ndcg,
                        **metrics,
                    }
                    weight_rows.append(row)
                    weight_predictions[weight] = predictions
                    all_metric_rows.append(row)

                    if args.save_all_predictions:
                        pred_path = (
                            output_dir
                            / domain
                            / "predictions"
                            / f"{_slug(row['method'])}.jsonl"
                        )
                        all_weight_table_dir = output_dir / domain / "tables"
                        all_weight_table_dir.mkdir(parents=True, exist_ok=True)
                        save_jsonl(predictions, pred_path)
                        eval_df.to_csv(
                            all_weight_table_dir / f"{_slug(row['method'])}_eval_records.csv",
                            index=False,
                        )

                if not weight_rows:
                    continue
                best_row = max(weight_rows, key=lambda row: float(row.get(f"NDCG@{args.k}", float("-inf"))))
                best_row = dict(best_row)
                best_row["is_test_sweep_best"] = True
                pair_summary_rows.append(best_row)
                domain_pair_best.append(best_row)

                best_weight = float(best_row["ours_weight"])
                best_predictions = weight_predictions[best_weight]
                pair_slug = (
                    f"best_fusion_rank_{ours_method}__{external_method}"
                    f"__ours_w{_weight_slug(best_weight)}"
                )
                pair_pred_path = output_dir / domain / "predictions" / f"{_slug(pair_slug)}.jsonl"
                pair_table_dir = output_dir / domain / "tables"
                pair_table_dir.mkdir(parents=True, exist_ok=True)
                save_jsonl(best_predictions, pair_pred_path)
                best_metrics, best_eval_df = _metrics_for_predictions(best_predictions, k=args.k)
                best_eval_df.to_csv(
                    pair_table_dir / f"{_slug(pair_slug)}_eval_records.csv",
                    index=False,
                )
                compute_ranking_exposure_distribution(best_eval_df, k=args.k).to_csv(
                    pair_table_dir / f"{_slug(pair_slug)}_exposure_distribution.csv",
                    index=False,
                )
                pd.DataFrame([best_metrics]).to_csv(
                    pair_table_dir / f"{_slug(pair_slug)}_ranking_metrics.csv",
                    index=False,
                )

        if domain_pair_best:
            best_domain_rows.append(
                max(domain_pair_best, key=lambda row: float(row.get(f"NDCG@{args.k}", float("-inf"))))
            )

    _write_csv(manifest_rows, output_dir / "input_manifest.csv")
    _write_csv(all_metric_rows, output_dir / "fusion_all_metrics.csv")
    _write_csv(pair_summary_rows, output_dir / "fusion_pair_best_metrics.csv")
    _write_csv(best_domain_rows, output_dir / "fusion_best_by_domain.csv")

    print(f"Saved fusion metrics: {output_dir / 'fusion_all_metrics.csv'}")
    print(f"Saved pair best metrics: {output_dir / 'fusion_pair_best_metrics.csv'}")
    print(f"Saved domain best metrics: {output_dir / 'fusion_best_by_domain.csv'}")
    print("Note: best-over-weight rows are diagnostic if weights were selected on the test set.")


if __name__ == "__main__":
    main()
