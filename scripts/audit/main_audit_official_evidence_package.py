from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PRIMARY_VARIANT = "official_code_qwen3base_default_hparams_declared_adaptation"
REQUIRED_SCORE_SCHEMA = ["source_event_id", "user_id", "item_id", "score"]
REQUIRED_METRIC_COLUMNS = [
    "HR@5",
    "HR@10",
    "HR@20",
    "NDCG@5",
    "NDCG@10",
    "NDCG@20",
    "MRR",
]
COMMON_REQUIRED_FILES = [
    "fairness_provenance.json",
    "tables/ranking_metrics.csv",
    "tables/same_candidate_external_baseline_summary.csv",
    "tables/external_score_coverage.csv",
    "tables/ranking_exposure_distribution.csv",
    "tables/ranking_eval_records.csv",
]
SERVER_REQUIRED_FILES = [
    "scores.csv",
    "predictions/rank_predictions.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a completed official baseline evidence directory. "
            "Use local_light for copied local evidence packages and server_final "
            "for full server output directories."
        )
    )
    parser.add_argument("--evidence_dir", required=True, help="Directory to audit.")
    parser.add_argument(
        "--mode",
        choices=["local_light", "server_final"],
        default="local_light",
        help="local_light permits scores/predictions to stay server-only; server_final requires them.",
    )
    parser.add_argument("--expected_users", type=int, default=10000)
    parser.add_argument("--expected_candidates_per_user", type=int, default=101)
    parser.add_argument("--output_json", default="", help="Optional path for the audit JSON.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _line_count(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _read_single_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, got {len(rows)}")
    return rows[0]


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _present_file(path: Path) -> dict[str, Any]:
    exists = path.exists() and path.is_file() and path.stat().st_size > 0
    row: dict[str, Any] = {
        "path": str(path),
        "present": exists,
        "size": path.stat().st_size if path.exists() else 0,
    }
    if exists and path.suffix in {".csv", ".jsonl"}:
        row["lines"] = _line_count(path)
    return row


def _first_glob(base_dir: Path, patterns: list[str]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(base_dir.glob(pattern)))
    for path in matches:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _audit(evidence_dir: Path, mode: str, expected_users: int, expected_candidates_per_user: int) -> dict[str, Any]:
    expected_score_rows = expected_users * expected_candidates_per_user
    expected_score_csv_lines = expected_score_rows + 1
    expected_eval_csv_lines = expected_users + 1

    files: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    warnings: list[str] = []

    required_files = list(COMMON_REQUIRED_FILES)
    if mode == "server_final":
        required_files.extend(SERVER_REQUIRED_FILES)

    for rel_path in required_files:
        file_row = _present_file(evidence_dir / rel_path)
        files[rel_path] = file_row
        if not file_row["present"]:
            failures.append(f"missing_required_file:{rel_path}")

    optional_files = ["inspect_fairness_provenance.json", *SERVER_REQUIRED_FILES]
    for rel_path in optional_files:
        if rel_path not in files:
            files[rel_path] = _present_file(evidence_dir / rel_path)

    score_audit_json = _first_glob(evidence_dir, ["*score*audit*.json"])
    score_audit_txt = _first_glob(evidence_dir, ["*same_candidate_score_audit.txt"])
    run_summary_json = _first_glob(evidence_dir, ["*run_summary.json"])
    if score_audit_json is None:
        failures.append("missing_score_audit_json")
    if score_audit_txt is None:
        failures.append("missing_same_candidate_score_audit_txt")
    if run_summary_json is None:
        failures.append("missing_run_summary_json")

    dynamic_files = {
        "score_audit_json": str(score_audit_json) if score_audit_json else "",
        "score_audit_txt": str(score_audit_txt) if score_audit_txt else "",
        "run_summary_json": str(run_summary_json) if run_summary_json else "",
    }

    provenance_summary: dict[str, Any] = {}
    prov_path = evidence_dir / "fairness_provenance.json"
    if prov_path.exists() and prov_path.stat().st_size > 0:
        try:
            provenance = _load_json(prov_path)
            blockers = provenance.get("blockers") or []
            score_schema = provenance.get("score_schema") or []
            provenance_summary = {
                "implementation_status": provenance.get("implementation_status"),
                "blocker_count": len(blockers) if isinstance(blockers, list) else "invalid",
                "score_coverage_rate": provenance.get("score_coverage_rate"),
                "comparison_variant": provenance.get("comparison_variant"),
                "score_schema": score_schema,
                "test_set_model_selection_allowed": provenance.get("test_set_model_selection_allowed"),
                "baseline_extra_tuning_allowed": provenance.get("baseline_extra_tuning_allowed"),
            }
            if provenance.get("implementation_status") != "official_completed":
                failures.append("provenance_not_official_completed")
            if blockers != []:
                failures.append("provenance_blockers_not_empty")
            if _as_float(provenance.get("score_coverage_rate")) != 1.0:
                failures.append("provenance_score_coverage_not_1")
            if provenance.get("comparison_variant") != PRIMARY_VARIANT:
                failures.append("provenance_comparison_variant_mismatch")
            if score_schema != REQUIRED_SCORE_SCHEMA:
                failures.append("provenance_score_schema_mismatch")
            if provenance.get("test_set_model_selection_allowed") is not False:
                failures.append("provenance_allows_test_set_selection")
            if provenance.get("baseline_extra_tuning_allowed") is not False:
                failures.append("provenance_allows_extra_baseline_tuning")
        except Exception as exc:
            failures.append(f"provenance_read_error:{exc}")

    metric_summary: dict[str, Any] = {}
    metrics_path = evidence_dir / "tables/ranking_metrics.csv"
    if metrics_path.exists() and metrics_path.stat().st_size > 0:
        try:
            metrics = _read_single_csv(metrics_path)
            metric_summary = {key: metrics.get(key, "") for key in REQUIRED_METRIC_COLUMNS}
            metric_summary["sample_count"] = metrics.get("sample_count", "")
            metric_summary["avg_candidates"] = metrics.get("avg_candidates", "")
            for metric_name in REQUIRED_METRIC_COLUMNS:
                if metrics.get(metric_name, "") == "":
                    failures.append(f"missing_metric_column:{metric_name}")
            if int(_as_float(metrics.get("sample_count"), -1)) != expected_users:
                failures.append("ranking_metrics_sample_count_mismatch")
            if _as_float(metrics.get("avg_candidates"), -1.0) != float(expected_candidates_per_user):
                failures.append("ranking_metrics_avg_candidates_mismatch")
        except Exception as exc:
            failures.append(f"ranking_metrics_read_error:{exc}")

    coverage_summary: dict[str, Any] = {}
    coverage_path = evidence_dir / "tables/external_score_coverage.csv"
    if coverage_path.exists() and coverage_path.stat().st_size > 0:
        try:
            coverage = _read_single_csv(coverage_path)
            coverage_summary = dict(coverage)
            if _as_float(coverage.get("score_coverage_rate"), -1.0) != 1.0:
                failures.append("external_score_coverage_not_1")
        except Exception as exc:
            failures.append(f"external_score_coverage_read_error:{exc}")

    eval_records = files.get("tables/ranking_eval_records.csv", {})
    if eval_records.get("present") and eval_records.get("lines") != expected_eval_csv_lines:
        failures.append("ranking_eval_records_line_count_mismatch")

    scores_file = files.get("scores.csv", {})
    if scores_file.get("present") and scores_file.get("lines") != expected_score_csv_lines:
        failures.append("scores_csv_line_count_mismatch")
    if mode == "local_light" and scores_file.get("present"):
        warnings.append("local_light_contains_scores_csv")

    predictions_file = files.get("predictions/rank_predictions.jsonl", {})
    if predictions_file.get("present") and predictions_file.get("lines") != expected_users:
        failures.append("rank_predictions_line_count_mismatch")
    if mode == "local_light" and predictions_file.get("present"):
        warnings.append("local_light_contains_predictions")

    return {
        "evidence_dir": str(evidence_dir),
        "mode": mode,
        "expected_users": expected_users,
        "expected_candidates_per_user": expected_candidates_per_user,
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "files": files,
        "dynamic_files": dynamic_files,
        "provenance": provenance_summary,
        "metrics": metric_summary,
        "coverage": coverage_summary,
    }


def main() -> int:
    args = parse_args()
    evidence_dir = Path(args.evidence_dir).expanduser()
    result = _audit(
        evidence_dir=evidence_dir,
        mode=args.mode,
        expected_users=args.expected_users,
        expected_candidates_per_user=args.expected_candidates_per_user,
    )
    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.quiet:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
