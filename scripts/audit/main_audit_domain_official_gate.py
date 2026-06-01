from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_METRICS = ("HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR")
REQUIRED_SCORE_SCHEMA = ["source_event_id", "user_id", "item_id", "score"]
PRIMARY_VARIANT = "official_code_qwen3base_default_hparams_declared_adaptation"

OFFICIAL_METHOD_DIRS = {
    "llmemb": "{exp}_llmemb_official_qwen3base_same_candidate",
    "proex_profile": "{exp}_proex_profile_official_qwen3base_same_candidate",
    "promax_profile": "{exp}_promax_profile_official_qwen3base_same_candidate",
    "elmrec_graph": "{exp}_elmrec_graph_official_qwen3base_same_candidate",
    "irllrec_intent": "{exp}_irllrec_intent_official_qwen3base_same_candidate",
    "rlmrec_graphcl": "{exp}_rlmrec_graphcl_official_qwen3base_same_candidate",
    "llm2rec_sasrec": "{exp}_llm2rec_sasrec_official_qwen3base_same_candidate",
    "llmesr_sasrec": "{exp}_llmesr_sasrec_official_qwen3base_same_candidate",
}

OFFICIAL_REQUIRED_FILES = (
    "fairness_provenance.json",
    "scores.csv",
    "predictions/rank_predictions.jsonl",
    "tables/ranking_metrics.csv",
    "tables/same_candidate_external_baseline_summary.csv",
    "tables/external_score_coverage.csv",
    "tables/ranking_exposure_distribution.csv",
    "tables/ranking_eval_records.csv",
)

CCRP_REQUIRED_IMPORTED_FILES = (
    "predictions/rank_predictions.jsonl",
    "tables/ranking_metrics.csv",
    "tables/same_candidate_external_baseline_summary.csv",
    "tables/external_score_coverage.csv",
    "tables/ranking_exposure_distribution.csv",
    "tables/ranking_eval_records.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit one large10000_100neg domain for the eight official same-candidate "
            "baselines plus imported C-CRP v3 artifacts. This is a read-only gate."
        )
    )
    parser.add_argument("--root", default=".", help="Project root containing outputs/.")
    parser.add_argument("--domain", required=True, help="Domain name, e.g. sports.")
    parser.add_argument("--expected_users", type=int, default=10000)
    parser.add_argument("--expected_candidates_per_user", type=int, default=101)
    parser.add_argument("--output_json", default="", help="Optional gate JSON path.")
    parser.add_argument("--output_csv", default="", help="Optional compact metrics CSV path.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _line_count(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _read_single_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1:
        raise ValueError(f"expected one row, got {len(rows)}")
    return rows[0]


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _finite_metric(row: dict[str, Any], metric: str) -> bool:
    value = _as_float(row.get(metric))
    return math.isfinite(value)


def _first_existing(base_dir: Path, patterns: tuple[str, ...]) -> str:
    for pattern in patterns:
        for path in sorted(base_dir.glob(pattern)):
            if path.is_file() and path.stat().st_size > 0:
                return str(path)
    return ""


def _basic_file_row(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "present": path.exists() and path.is_file() and path.stat().st_size > 0,
        "size": path.stat().st_size if path.exists() else 0,
    }
    if path.suffix in {".csv", ".jsonl"} and row["present"]:
        row["lines"] = _line_count(path)
    return row


def _metric_gate(
    metrics_path: Path,
    failures: list[str],
    expected_users: int,
    expected_candidates_per_user: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if not metrics_path.exists():
        failures.append("missing_metrics_csv")
        return metrics
    try:
        metrics = _read_single_csv(metrics_path)
    except Exception as exc:
        failures.append(f"metrics_read_error:{exc}")
        return {}

    for metric in REQUIRED_METRICS:
        if metric not in metrics or not _finite_metric(metrics, metric):
            failures.append(f"missing_or_nonfinite_metric:{metric}")
    if int(_as_float(metrics.get("sample_count"))) != expected_users:
        failures.append("sample_count_mismatch")
    if _as_float(metrics.get("avg_candidates")) != float(expected_candidates_per_user):
        failures.append("avg_candidates_mismatch")
    return metrics


def _summary_gate(
    summary_path: Path,
    failures: list[str],
    expected_score_rows: int,
    expected_users: int,
    expected_candidates_per_user: int,
    *,
    expected_status_label: str,
    expected_artifact_class: str = "completed_result",
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if not summary_path.exists():
        failures.append("missing_summary_csv")
        return summary
    try:
        summary = _read_single_csv(summary_path)
    except Exception as exc:
        failures.append(f"summary_read_error:{exc}")
        return {}

    if summary.get("status_label") != expected_status_label:
        failures.append("summary_status_label_mismatch")
    if summary.get("artifact_class") != expected_artifact_class:
        failures.append("summary_artifact_class_mismatch")
    if int(_as_float(summary.get("sample_count"))) != expected_users:
        failures.append("summary_sample_count_mismatch")
    if _as_float(summary.get("avg_candidates")) != float(expected_candidates_per_user):
        failures.append("summary_avg_candidates_mismatch")
    if _as_float(summary.get("score_coverage_rate")) != 1.0:
        failures.append("summary_score_coverage_not_1")
    if int(_as_float(summary.get("ranking_events"))) != expected_users:
        failures.append("summary_ranking_events_mismatch")
    if int(_as_float(summary.get("total_candidates"))) != expected_score_rows:
        failures.append("summary_total_candidates_mismatch")
    if int(_as_float(summary.get("matched_candidates"))) != expected_score_rows:
        failures.append("summary_matched_candidates_mismatch")
    return summary


def _coverage_gate(path: Path, failures: list[str], expected_score_rows: int) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    if not path.exists():
        failures.append("missing_external_score_coverage_csv")
        return coverage
    try:
        coverage = _read_single_csv(path)
    except Exception as exc:
        failures.append(f"coverage_read_error:{exc}")
        return {}
    if _as_float(coverage.get("score_coverage_rate")) != 1.0:
        failures.append("coverage_score_coverage_not_1")
    for key in ("candidate_key_count", "score_key_count"):
        if key in coverage and int(_as_float(coverage.get(key))) != expected_score_rows:
            failures.append(f"coverage_{key}_mismatch")
    return coverage


def audit_official_dir(
    method: str,
    evidence_dir: Path,
    *,
    expected_users: int,
    expected_candidates_per_user: int,
) -> dict[str, Any]:
    expected_score_rows = expected_users * expected_candidates_per_user
    failures: list[str] = []
    warnings: list[str] = []
    files: dict[str, Any] = {}

    for rel_path in OFFICIAL_REQUIRED_FILES:
        row = _basic_file_row(evidence_dir / rel_path)
        files[rel_path] = row
        if not row["present"]:
            failures.append(f"missing_required_file:{rel_path}")

    if files.get("scores.csv", {}).get("lines") != expected_score_rows + 1:
        failures.append("scores_csv_line_count_mismatch")
    if files.get("predictions/rank_predictions.jsonl", {}).get("lines") != expected_users:
        failures.append("prediction_jsonl_line_count_mismatch")
    if files.get("tables/ranking_eval_records.csv", {}).get("lines") != expected_users + 1:
        failures.append("ranking_eval_records_line_count_mismatch")
    if files.get("tables/same_candidate_external_baseline_summary.csv", {}).get("lines") != 2:
        failures.append("summary_csv_line_count_mismatch")

    score_audit_json = _first_existing(evidence_dir, ("*score*audit*.json",))
    score_audit_txt = _first_existing(evidence_dir, ("*same_candidate_score_audit.txt",))
    run_summary = _first_existing(evidence_dir, ("*run_summary.json",))
    if not score_audit_json:
        failures.append("missing_score_audit_json")
    if not score_audit_txt:
        failures.append("missing_same_candidate_score_audit_txt")
    if not run_summary:
        failures.append("missing_run_summary_json")

    provenance: dict[str, Any] = {}
    prov_path = evidence_dir / "fairness_provenance.json"
    if prov_path.exists():
        provenance = _load_json(prov_path)
        blockers = provenance.get("blockers") or []
        if provenance.get("implementation_status") != "official_completed":
            failures.append("provenance_not_official_completed")
        if blockers != []:
            failures.append("provenance_blockers_not_empty")
        if _as_float(provenance.get("score_coverage_rate")) != 1.0:
            failures.append("provenance_score_coverage_not_1")
        if provenance.get("comparison_variant") != PRIMARY_VARIANT:
            failures.append("provenance_comparison_variant_mismatch")
        if provenance.get("score_schema") != REQUIRED_SCORE_SCHEMA:
            failures.append("provenance_score_schema_mismatch")
        if provenance.get("test_set_model_selection_allowed") is not False:
            failures.append("provenance_allows_test_set_model_selection")
        if provenance.get("baseline_extra_tuning_allowed") is not False:
            failures.append("provenance_allows_extra_baseline_tuning")
    else:
        failures.append("missing_fairness_provenance")

    metrics = _metric_gate(
        evidence_dir / "tables/ranking_metrics.csv",
        failures,
        expected_users,
        expected_candidates_per_user,
    )
    summary = _summary_gate(
        evidence_dir / "tables/same_candidate_external_baseline_summary.csv",
        failures,
        expected_score_rows,
        expected_users,
        expected_candidates_per_user,
        expected_status_label="same_schema_external_baseline",
    )
    coverage = _coverage_gate(evidence_dir / "tables/external_score_coverage.csv", failures, expected_score_rows)

    return {
        "method": method,
        "kind": "official_baseline",
        "dir": str(evidence_dir),
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "metrics": {metric: metrics.get(metric, "") for metric in REQUIRED_METRICS}
        | {
            "sample_count": metrics.get("sample_count", ""),
            "avg_candidates": metrics.get("avg_candidates", ""),
        },
        "provenance": {
            "implementation_status": provenance.get("implementation_status", ""),
            "blockers": provenance.get("blockers", ""),
            "score_coverage_rate": provenance.get("score_coverage_rate", ""),
            "comparison_variant": provenance.get("comparison_variant", ""),
        },
        "coverage": coverage,
        "summary": {
            key: summary.get(key, "")
            for key in (
                "baseline_name",
                "domain",
                "status_label",
                "artifact_class",
                "ranking_events",
                "total_candidates",
                "matched_candidates",
                "score_coverage_rate",
                "sample_count",
                "avg_candidates",
            )
        },
        "row_counts": {
            "scores_csv_lines": files.get("scores.csv", {}).get("lines"),
            "predictions_jsonl_lines": files.get("predictions/rank_predictions.jsonl", {}).get("lines"),
            "ranking_eval_records_csv_lines": files.get("tables/ranking_eval_records.csv", {}).get("lines"),
            "summary_csv_lines": files.get("tables/same_candidate_external_baseline_summary.csv", {}).get("lines"),
        },
        "dynamic_files": {
            "score_audit_json": score_audit_json,
            "score_audit_txt": score_audit_txt,
            "run_summary_json": run_summary,
        },
    }


def audit_ccrp(
    root: Path,
    domain: str,
    *,
    expected_users: int,
    expected_candidates_per_user: int,
) -> dict[str, Any]:
    exp = f"{domain}_large10000_100neg"
    raw_dir = root / "outputs" / f"{exp}_ccrp_v3"
    imported_dir = root / "outputs" / f"{exp}_ccrp_v3_qwen3base_pointwise_same_candidate"
    expected_score_rows = expected_users * expected_candidates_per_user
    failures: list[str] = []
    files: dict[str, Any] = {}

    for rel_path in ("report.json", "scores.csv", "user_ranks.jsonl"):
        row = _basic_file_row(raw_dir / rel_path)
        files[f"raw/{rel_path}"] = row
        if not row["present"]:
            failures.append(f"missing_raw_file:{rel_path}")
    for rel_path in CCRP_REQUIRED_IMPORTED_FILES:
        row = _basic_file_row(imported_dir / rel_path)
        files[f"imported/{rel_path}"] = row
        if not row["present"]:
            failures.append(f"missing_imported_file:{rel_path}")

    if files.get("raw/scores.csv", {}).get("lines") != expected_score_rows + 1:
        failures.append("raw_scores_csv_line_count_mismatch")
    if files.get("raw/user_ranks.jsonl", {}).get("lines") != expected_users:
        failures.append("raw_user_ranks_jsonl_line_count_mismatch")
    if files.get("imported/predictions/rank_predictions.jsonl", {}).get("lines") != expected_users:
        failures.append("imported_prediction_jsonl_line_count_mismatch")
    if files.get("imported/tables/ranking_eval_records.csv", {}).get("lines") != expected_users + 1:
        failures.append("imported_ranking_eval_records_line_count_mismatch")
    if files.get("imported/tables/same_candidate_external_baseline_summary.csv", {}).get("lines") != 2:
        failures.append("imported_summary_csv_line_count_mismatch")

    metrics = _metric_gate(
        imported_dir / "tables/ranking_metrics.csv",
        failures,
        expected_users,
        expected_candidates_per_user,
    )
    summary = _summary_gate(
        imported_dir / "tables/same_candidate_external_baseline_summary.csv",
        failures,
        expected_score_rows,
        expected_users,
        expected_candidates_per_user,
        expected_status_label="same_schema_internal_method",
    )
    coverage = _coverage_gate(imported_dir / "tables/external_score_coverage.csv", failures, expected_score_rows)

    report: dict[str, Any] = {}
    if (raw_dir / "report.json").exists():
        report = _load_json(raw_dir / "report.json")

    return {
        "method": "ccrp_v3_qwen3base_pointwise",
        "kind": "internal_method",
        "raw_dir": str(raw_dir),
        "imported_dir": str(imported_dir),
        "ok": not failures,
        "failures": failures,
        "warnings": [],
        "metrics": {metric: metrics.get(metric, "") for metric in REQUIRED_METRICS}
        | {
            "sample_count": metrics.get("sample_count", ""),
            "avg_candidates": metrics.get("avg_candidates", ""),
        },
        "coverage": coverage,
        "summary": {
            key: summary.get(key, "")
            for key in (
                "baseline_name",
                "domain",
                "status_label",
                "artifact_class",
                "ranking_events",
                "total_candidates",
                "matched_candidates",
                "score_coverage_rate",
                "sample_count",
                "avg_candidates",
            )
        },
        "row_counts": {
            "raw_scores_csv_lines": files.get("raw/scores.csv", {}).get("lines"),
            "raw_user_ranks_jsonl_lines": files.get("raw/user_ranks.jsonl", {}).get("lines"),
            "predictions_jsonl_lines": files.get("imported/predictions/rank_predictions.jsonl", {}).get("lines"),
            "ranking_eval_records_csv_lines": files.get("imported/tables/ranking_eval_records.csv", {}).get("lines"),
            "summary_csv_lines": files.get("imported/tables/same_candidate_external_baseline_summary.csv", {}).get("lines"),
        },
        "raw_report_keys": sorted(report.keys()),
    }


def find_stray_official_dirs(root: Path, domain: str, expected_dirs: set[str]) -> list[dict[str, Any]]:
    pattern = f"{domain}_large10000_100neg_*official*qwen3base*same_candidate"
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "outputs").glob(pattern)):
        if not path.is_dir() or path.name in expected_dirs:
            continue
        file_count = 0
        total_bytes = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                file_count += 1
                total_bytes += file_path.stat().st_size
        rows.append(
            {
                "dir": str(path),
                "file_count": file_count,
                "total_bytes": total_bytes,
                "empty": file_count == 0,
            }
        )
    return rows


def write_compact_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "kind",
        "ok",
        "sample_count",
        "avg_candidates",
        *REQUIRED_METRICS,
        "score_coverage_rate",
        "scores_csv_lines",
        "predictions_jsonl_lines",
        "ranking_eval_records_csv_lines",
        "failures",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row.get("metrics", {})
            compact = {
                "method": row.get("method", ""),
                "kind": row.get("kind", ""),
                "ok": row.get("ok", False),
                "sample_count": metrics.get("sample_count", ""),
                "avg_candidates": metrics.get("avg_candidates", ""),
                "score_coverage_rate": row.get("summary", {}).get("score_coverage_rate")
                or row.get("provenance", {}).get("score_coverage_rate", ""),
                "scores_csv_lines": row.get("row_counts", {}).get("scores_csv_lines")
                or row.get("row_counts", {}).get("raw_scores_csv_lines", ""),
                "predictions_jsonl_lines": row.get("row_counts", {}).get("predictions_jsonl_lines", ""),
                "ranking_eval_records_csv_lines": row.get("row_counts", {}).get("ranking_eval_records_csv_lines", ""),
                "failures": ";".join(row.get("failures", [])),
            }
            for metric in REQUIRED_METRICS:
                compact[metric] = metrics.get(metric, "")
            writer.writerow(compact)


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    exp = f"{args.domain}_large10000_100neg"
    official_rows = []
    for method, dir_template in OFFICIAL_METHOD_DIRS.items():
        evidence_dir = root / "outputs" / dir_template.format(exp=exp)
        official_rows.append(
            audit_official_dir(
                method,
                evidence_dir,
                expected_users=args.expected_users,
                expected_candidates_per_user=args.expected_candidates_per_user,
            )
        )

    ccrp_row = audit_ccrp(
        root,
        args.domain,
        expected_users=args.expected_users,
        expected_candidates_per_user=args.expected_candidates_per_user,
    )
    expected_dir_names = {Path(row["dir"]).name for row in official_rows}
    stray_dirs = find_stray_official_dirs(root, args.domain, expected_dir_names)

    official_ok = sum(1 for row in official_rows if row["ok"])
    result = {
        "domain": args.domain,
        "root": str(root),
        "expected_users": args.expected_users,
        "expected_candidates_per_user": args.expected_candidates_per_user,
        "official_expected_count": len(OFFICIAL_METHOD_DIRS),
        "official_ok_count": official_ok,
        "official_all_ok": official_ok == len(OFFICIAL_METHOD_DIRS),
        "ccrp_ok": ccrp_row["ok"],
        "gate_ok": official_ok == len(OFFICIAL_METHOD_DIRS) and ccrp_row["ok"],
        "official_rows": official_rows,
        "ccrp": ccrp_row,
        "stray_official_like_dirs": stray_dirs,
    }

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_csv:
        write_compact_csv(Path(args.output_csv).expanduser(), [ccrp_row, *official_rows])
    if not args.quiet:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["gate_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
