from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any


FULL_METRICS = ("MRR", "HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20")
DEFAULT_ABLATIONS = (
    "full",
    "without_boundary_uncertainty",
    "without_calibration_gap",
    "without_evidence_support",
    "without_counterevidence",
    "without_risk_penalty",
)
COMMAND_FILES = ("commands.md", "commands.txt", "run_commands.md", "run_command.txt")
LOG_FILES = ("log_snippets.md", "log_snippets.txt", "key_logs.md", "run_log_excerpt.txt")
SERVER_COMPARISON_FILES = (
    "local_server_manifest_comparison.json",
    "local_server_evidence_consistency.json",
    "manifest_comparison.json",
)
CONFIG_FILES = ("config.json", "run_config.json", "selected_valid_config.json", "selected_hyperparameters.json")
DEFAULT_CONTROLS = ("eta", "confidence_weight", "weight_grid_label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a completed Phase 2.5 paper-critical module package without "
            "running experiments. This enforces the lightweight evidence package "
            "standard for observation, component ablation, and hyperparameter modules."
        )
    )
    parser.add_argument(
        "--module",
        required=True,
        choices=("observation_motivation", "component_ablation", "hyperparameter_analysis"),
    )
    parser.add_argument("--package_dir", required=True)
    parser.add_argument("--expected_events", type=int, default=10000)
    parser.add_argument("--expected_candidates_per_event", type=int, default=101)
    parser.add_argument("--min_join_rate", type=float, default=0.999)
    parser.add_argument("--expected_ablation", action="append", default=[])
    parser.add_argument("--expected_control", action="append", default=[])
    parser.add_argument("--min_plot_files", type=int, default=2)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_md", default="")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames or []), rows


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_int(value: Any, default: int = -1) -> int:
    try:
        parsed = float(str(value).strip())
    except Exception:
        return default
    if not math.isfinite(parsed) or abs(parsed - round(parsed)) > 1e-9:
        return default
    return int(round(parsed))


def _present(base: Path, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if (base / name).exists() and (base / name).stat().st_size > 0]


def _require_file(base: Path, rel_path: str, failures: list[str]) -> Path:
    path = base / rel_path
    if not path.exists():
        failures.append(f"missing_required_file:{rel_path}")
    elif path.stat().st_size <= 0:
        failures.append(f"empty_required_file:{rel_path}")
    return path


def _load_required_json(base: Path, rel_path: str, failures: list[str]) -> dict[str, Any]:
    path = _require_file(base, rel_path, failures)
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    try:
        return _read_json(path)
    except Exception as exc:
        failures.append(f"json_read_error:{rel_path}:{exc}")
        return {}


def _load_required_csv(base: Path, rel_path: str, failures: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    path = _require_file(base, rel_path, failures)
    if not path.exists() or path.stat().st_size <= 0:
        return [], []
    try:
        return _csv_rows(path)
    except Exception as exc:
        failures.append(f"csv_read_error:{rel_path}:{exc}")
        return [], []


def _has_nonempty_command(provenance: dict[str, Any], base: Path) -> bool:
    return bool(str(provenance.get("command", "")).strip()) or bool(_present(base, COMMAND_FILES))


def _has_input_manifest(provenance: dict[str, Any], base: Path) -> bool:
    if provenance.get("input_sha256") or provenance.get("sweep_sha256") or provenance.get("test_sweep_sha256"):
        return True
    return bool(
        _present(
            base,
            (
                "input_path_manifest.sha256",
                "input_manifest.sha256",
                "input_path_manifest.json",
                "input_manifest.json",
            ),
        )
    )


def _has_config_or_selected_hparams(provenance: dict[str, Any], base: Path) -> bool:
    if _present(base, CONFIG_FILES):
        return True
    for key in (
        "filters",
        "controls",
        "score_modes_grid",
        "ablations_grid",
        "etas_grid",
        "confidence_weights_grid",
        "weight_grid",
        "selected_config",
    ):
        if provenance.get(key):
            return True
    return False


def _comparison_has_substance(comparison: dict[str, Any]) -> bool:
    checked_files = comparison.get("checked_files")
    if isinstance(checked_files, list) and any(row.get("sha256") or row.get("sha256_ok") for row in checked_files if isinstance(row, dict)):
        return True
    files = comparison.get("files")
    if isinstance(files, dict) and files:
        return True
    manifest_checks = comparison.get("manifest_checks")
    if isinstance(manifest_checks, dict) and any(isinstance(row, dict) and row.get("ok") for row in manifest_checks.values()):
        return True
    rows = comparison.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("files") or row.get("light_sync_manifest") or row.get("server_large_artifact_manifest"):
                return True
    if int(comparison.get("row_count") or 0) > 0 and int(comparison.get("ok_count") or 0) > 0 and files:
        return True
    return False


def _check_general_package(base: Path, provenance: dict[str, Any], failures: list[str], warnings: list[str]) -> None:
    if not _has_nonempty_command(provenance, base):
        failures.append("missing_command_record")
    if not _present(base, LOG_FILES):
        failures.append("missing_log_snippets")
    if not str(provenance.get("git_commit", "")).strip():
        failures.append("missing_git_commit")
    if not _has_config_or_selected_hparams(provenance, base):
        failures.append("missing_config_or_selected_hyperparameters")
    if not _has_input_manifest(provenance, base):
        failures.append("missing_input_path_manifest_or_hashes")
    comparisons = _present(base, SERVER_COMPARISON_FILES)
    if not comparisons:
        failures.append("missing_local_server_manifest_comparison")
    else:
        for name in comparisons:
            try:
                comparison = _read_json(base / name)
            except Exception as exc:
                failures.append(f"local_server_manifest_comparison_read_error:{name}:{exc}")
                continue
            if comparison.get("ok") is not True:
                failures.append(f"local_server_manifest_comparison_not_ok:{name}")
            if not _comparison_has_substance(comparison):
                failures.append(f"local_server_manifest_comparison_lacks_evidence:{name}")

    for path in base.iterdir():
        if not path.is_file():
            continue
        if path.name == "scores.csv":
            failures.append("disallowed_bulk_scores_csv")
        if path.name == "rank_predictions.jsonl":
            failures.append("disallowed_bulk_prediction_jsonl")
        if path.suffix.lower() in {".pt", ".pth", ".npy", ".pkl"}:
            warnings.append(f"large_or_binary_artifact_in_package:{path.name}")


def _check_metrics(header: list[str], failures: list[str], *, context: str) -> None:
    missing = [metric for metric in FULL_METRICS if metric not in header]
    if missing:
        failures.append(f"{context}:missing_full_metrics:{','.join(missing)}")


def _path_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _resolve_figure(base: Path, value: str) -> Path:
    path = Path(value)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([Path.cwd() / path, base / path, base / path.name])
    for candidate in candidates:
        if candidate.exists() and _path_inside(candidate, base):
            return candidate
    return base / path.name


def _check_figures(base: Path, provenance: dict[str, Any], failures: list[str], *, min_plot_files: int) -> list[str]:
    figure_values = [str(item) for item in provenance.get("figure_paths", []) if str(item).strip()]
    if not figure_values:
        figure_values = [str(path) for path in sorted(base.glob("fig_*.png")) + sorted(base.glob("fig_*.pdf"))]
    present: list[str] = []
    for value in figure_values:
        path = _resolve_figure(base, value)
        if path.exists() and path.stat().st_size > 0:
            if not _path_inside(path, base):
                failures.append(f"figure_outside_package:{value}")
                continue
            present.append(path.name)
        else:
            failures.append(f"missing_figure_file:{value}")
    if len(present) < min_plot_files:
        failures.append(f"insufficient_plot_files:{len(present)}<{min_plot_files}")
    return present


def _event_count(row: dict[str, Any]) -> int:
    for key in ("n_events", "n_users", "users", "user_count", "event_count"):
        if key in row and str(row.get(key, "")).strip():
            return int(float(str(row[key])))
    return -1


def _check_observation_event_rows(
    event_rows: list[dict[str, str]],
    failures: list[str],
    *,
    expected_candidates_per_event: int,
) -> None:
    seen: set[tuple[str, str]] = set()
    for idx, row in enumerate(event_rows):
        method = str(row.get("method", "")).strip()
        event_id = str(row.get("event_id", "")).strip()
        if not method or not event_id:
            failures.append(f"observation_blank_event_key:{idx}")
            continue
        key = (method, event_id)
        if key in seen:
            failures.append(f"observation_duplicate_method_event:{method}:{event_id}")
        seen.add(key)
        rank = _as_int(row.get("positive_rank"))
        if rank < 1:
            failures.append(f"observation_invalid_positive_rank:{method}:{event_id}:{row.get('positive_rank')}")
        if "num_candidates" in row and str(row.get("num_candidates", "")).strip():
            num_candidates = _as_int(row.get("num_candidates"))
            if expected_candidates_per_event > 0 and num_candidates != expected_candidates_per_event:
                failures.append(
                    f"observation_num_candidates_mismatch:{method}:{event_id}:{num_candidates}!={expected_candidates_per_event}"
                )
            if rank > 0 and num_candidates > 0 and rank > num_candidates:
                failures.append(f"observation_positive_rank_beyond_candidates:{method}:{event_id}:{rank}>{num_candidates}")


def _audit_observation(
    base: Path,
    *,
    expected_events: int,
    expected_candidates_per_event: int,
    min_join_rate: float,
    min_plot_files: int,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    event_header, event_rows = _load_required_csv(base, "observation_event_bins.csv", failures)
    summary_header, summary_rows = _load_required_csv(base, "observation_summary.csv", failures)
    summary_json = _load_required_json(base, "observation_summary.json", failures)
    provenance = _load_required_json(base, "observation_provenance.json", failures)

    _check_general_package(base, provenance, failures, warnings)
    _check_metrics(summary_header, failures, context="observation_summary")
    if provenance.get("artifact_class") != "paper_critical_observation_motivation":
        failures.append("observation:unexpected_artifact_class")
    if provenance.get("status_label") != "paper_critical_observation_ready":
        failures.append("observation:unexpected_status_label")
    if provenance.get("paper_claim_scope") != "motivation_only_not_main_table_sota":
        failures.append("observation:unexpected_paper_claim_scope")
    if int(provenance.get("expected_candidates_per_event") or 0) != expected_candidates_per_event:
        failures.append(
            "observation:expected_candidates_per_event_mismatch:"
            f"{provenance.get('expected_candidates_per_event')}!={expected_candidates_per_event}"
        )
    _check_observation_event_rows(
        event_rows,
        failures,
        expected_candidates_per_event=expected_candidates_per_event,
    )

    methods = sorted({str(row.get("method", "")) for row in summary_rows if str(row.get("uncertainty_bin")) == "ALL"})
    if expected_events > 0 and methods:
        expected_event_rows = expected_events * len(methods)
        if len(event_rows) != expected_event_rows:
            failures.append(f"observation_event_bins_row_count:{len(event_rows)}!={expected_event_rows}")
        for row in summary_rows:
            if str(row.get("uncertainty_bin")) == "ALL" and _event_count(row) != expected_events:
                failures.append(f"observation_summary_all_count:{row.get('method')}:{_event_count(row)}!={expected_events}")
    if not methods:
        failures.append("observation:no_all_summary_rows")
    if not isinstance(summary_json.get("rows"), list) or len(summary_json.get("rows", [])) != len(summary_rows):
        failures.append("observation_summary_json_row_mismatch")

    join_report = provenance.get("join_report", [])
    if not isinstance(join_report, list) or not join_report:
        failures.append("observation:missing_join_report")
    for row in join_report if isinstance(join_report, list) else []:
        if _as_float(row.get("join_rate"), 0.0) < min_join_rate:
            failures.append(f"observation_join_rate_below_floor:{row.get('method')}:{row.get('join_rate')}")
        if row.get("exact_event_match") is not True:
            failures.append(f"observation_event_match_not_exact:{row.get('method')}")
    figures = _check_figures(base, provenance, failures, min_plot_files=min_plot_files)

    return {
        "module": "observation_motivation",
        "ok": not failures,
        "paper_claim_ready": not failures,
        "status_label": provenance.get("status_label"),
        "event_rows": len(event_rows),
        "summary_rows": len(summary_rows),
        "methods": methods,
        "figures": figures,
        "failures": failures,
        "warnings": warnings,
    }


def _audit_component_ablation(
    base: Path,
    *,
    expected_events: int,
    expected_candidates_per_event: int,
    expected_ablations: tuple[str, ...],
    min_plot_files: int,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    summary_header, summary_rows = _load_required_csv(base, "component_ablation_summary.csv", failures)
    sweep_header, sweep_rows = _load_required_csv(base, "valid_ccrp_sweep.csv", failures)
    provenance = _load_required_json(base, "component_ablation_provenance.json", failures)

    _check_general_package(base, provenance, failures, warnings)
    _check_metrics(summary_header, failures, context="component_ablation_summary")
    for column in ("audit_ok", "degeneracy_audit_ok", "score_coverage_rate"):
        if column not in summary_header:
            failures.append(f"component_ablation_summary_missing_column:{column}")
    selected_header, selected_rows = _load_required_csv(base, "selected_test_metrics.csv", failures)
    _check_metrics(selected_header, failures, context="selected_test_metrics")
    for column in ("audit_ok", "degeneracy_audit_ok", "score_coverage_rate", "candidate_key_count"):
        if column not in selected_header:
            failures.append(f"selected_test_metrics_missing_column:{column}")
    _load_required_json(base, "selected_valid_config.json", failures)
    internal_provenance = _load_required_json(base, "ccrp_internal_provenance.json", failures)
    _load_required_csv(base, "tables/ranking_metrics.csv", failures)
    coverage_header, coverage_rows = _load_required_csv(base, "tables/external_score_coverage.csv", failures)
    _load_required_csv(base, "tables/same_candidate_external_baseline_summary.csv", failures)
    _load_required_csv(base, "tables/ranking_eval_records.csv", failures)
    if provenance.get("artifact_class") != "paper_critical_component_ablation":
        failures.append("component_ablation:unexpected_artifact_class")
    if provenance.get("status_label") != "paper_critical_component_ablation_ready":
        failures.append("component_ablation:unexpected_status_label")
    if not internal_provenance:
        failures.append("component_ablation:missing_internal_provenance")

    summary_abls = {str(row.get("ablation", "")).strip() for row in summary_rows}
    sweep_abls = {str(row.get("ablation", "")).strip() for row in sweep_rows}
    for ablation in expected_ablations:
        if ablation not in summary_abls:
            failures.append(f"component_ablation_summary_missing:{ablation}")
        if ablation not in sweep_abls:
            failures.append(f"valid_sweep_missing_ablation:{ablation}")
    if expected_events > 0:
        for row in summary_rows:
            count = _event_count(row)
            if count != expected_events:
                failures.append(f"component_ablation_event_count:{row.get('ablation')}:{count}!={expected_events}")
    for row in summary_rows:
        if str(row.get("status_label", "")).strip() not in {"paper_critical_component_ablation_row", "same_schema_internal_ablation"}:
            failures.append(f"component_ablation_bad_row_status:{row.get('ablation')}:{row.get('status_label')}")
        if _as_bool(row.get("selected_on_test", False)):
            failures.append(f"component_ablation_test_selected_row:{row.get('ablation')}")
        if _as_float(row.get("score_coverage_rate"), 0.0) != 1.0:
            failures.append(f"component_ablation_score_coverage_not_one:{row.get('ablation')}:{row.get('score_coverage_rate')}")
        for column in ("audit_ok", "degeneracy_audit_ok"):
            if not _as_bool(row.get(column)):
                failures.append(f"component_ablation_{column}_false:{row.get('ablation')}")
    if len(selected_rows) != 1:
        failures.append(f"selected_test_metrics_row_count:{len(selected_rows)}!=1")
    for row in selected_rows:
        if _as_bool(row.get("selected_on_test", False)):
            failures.append("selected_test_metrics_selected_on_test")
        if _as_float(row.get("score_coverage_rate"), 0.0) != 1.0:
            failures.append(f"selected_test_metrics_score_coverage_not_one:{row.get('score_coverage_rate')}")
        for column in ("audit_ok", "degeneracy_audit_ok"):
            if not _as_bool(row.get(column)):
                failures.append(f"selected_test_metrics_{column}_false")
        expected_score_keys = expected_events * expected_candidates_per_event if expected_events > 0 else 0
        if expected_score_keys and _as_int(row.get("candidate_key_count")) != expected_score_keys:
            failures.append(f"selected_test_metrics_candidate_key_count:{row.get('candidate_key_count')}!={expected_score_keys}")
    if "score_coverage_rate" not in coverage_header:
        failures.append("external_score_coverage_missing_score_coverage_rate")
    for row in coverage_rows:
        if _as_float(row.get("score_coverage_rate"), 0.0) != 1.0:
            failures.append(f"external_score_coverage_not_one:{row.get('baseline_name')}:{row.get('score_coverage_rate')}")
    figures = _check_figures(base, provenance, failures, min_plot_files=min_plot_files)

    return {
        "module": "component_ablation",
        "ok": not failures,
        "paper_claim_ready": not failures,
        "status_label": provenance.get("status_label"),
        "summary_rows": len(summary_rows),
        "valid_sweep_rows": len(sweep_rows),
        "selected_test_metric_rows": len(selected_rows),
        "summary_ablations": sorted(summary_abls),
        "expected_ablations": list(expected_ablations),
        "figures": figures,
        "failures": failures,
        "warnings": warnings,
    }


def _audit_hyperparameter(base: Path, *, expected_controls: tuple[str, ...], min_plot_files: int) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    summary_header, summary_rows = _load_required_csv(base, "ccrp_hyperparameter_curve_summary.csv", failures)
    provenance = _load_required_json(base, "ccrp_hyperparameter_curve_provenance.json", failures)

    _check_general_package(base, provenance, failures, warnings)
    for column in ("split", "control", "control_value", "metric_name", "metric_value"):
        if column not in summary_header:
            failures.append(f"hyperparameter_summary_missing_column:{column}")
    if provenance.get("artifact_class") != "paper_critical_hyperparameter_analysis":
        failures.append("hyperparameter:unexpected_artifact_class")
    if provenance.get("status_label") != "paper_critical_hyperparameter_curve_ready":
        failures.append("hyperparameter:unexpected_status_label")
    if provenance.get("paper_claim_scope") != "valid_and_test_stability_curve_candidate":
        failures.append("hyperparameter:unexpected_paper_claim_scope")
    if provenance.get("reporting_mode") != "valid_and_test":
        failures.append("hyperparameter:not_valid_and_test")
    if not str(provenance.get("test_sweep_sha256", "")).strip():
        failures.append("hyperparameter:missing_test_sweep_sha256")
    audit_summary = provenance.get("audit_summary", {})
    if not isinstance(audit_summary, dict):
        failures.append("hyperparameter:missing_audit_summary")
        audit_summary = {}
    if audit_summary.get("require_audit_ok") is not True:
        failures.append("hyperparameter:audit_ok_not_required")
    if audit_summary.get("missing_audit_columns"):
        failures.append(f"hyperparameter:missing_audit_columns:{audit_summary.get('missing_audit_columns')}")
    if int(audit_summary.get("audited_rows") or 0) <= 0:
        failures.append("hyperparameter:no_audited_rows")
    if int(audit_summary.get("dropped_audit_rows") or 0) != 0:
        failures.append(f"hyperparameter:dropped_audit_rows:{audit_summary.get('dropped_audit_rows')}")
    controls = [str(control) for control in provenance.get("controls", [])]
    for control in expected_controls:
        if control not in controls:
            failures.append(f"hyperparameter_missing_expected_control:{control}")
        if control not in {str(row.get("control", "")) for row in summary_rows}:
            failures.append(f"hyperparameter_summary_missing_control:{control}")
    reports = provenance.get("control_reports", [])
    if not isinstance(reports, list):
        failures.append("hyperparameter:control_reports_not_list")
        reports = []
    report_keys = {(str(row.get("split", "")), str(row.get("control", ""))) for row in reports}
    for split in ("valid", "test"):
        for control in expected_controls:
            if (split, control) not in report_keys:
                failures.append(f"hyperparameter_missing_control_report:{split}:{control}")
    for row in provenance.get("control_reports", []):
        if row.get("meets_min_values") is not True:
            failures.append(f"hyperparameter_control_too_short:{row.get('split')}:{row.get('control')}")
    figures = _check_figures(
        base,
        provenance,
        failures,
        min_plot_files=max(min_plot_files, 2 * len(expected_controls)),
    )

    return {
        "module": "hyperparameter_analysis",
        "ok": not failures,
        "paper_claim_ready": not failures,
        "status_label": provenance.get("status_label"),
        "summary_rows": len(summary_rows),
        "controls": provenance.get("controls", []),
        "expected_controls": list(expected_controls),
        "reporting_mode": provenance.get("reporting_mode"),
        "figures": figures,
        "failures": failures,
        "warnings": warnings,
    }


def build_audit(
    *,
    module: str,
    package_dir: str | Path,
    expected_events: int = 10000,
    expected_candidates_per_event: int = 101,
    min_join_rate: float = 0.999,
    expected_ablations: tuple[str, ...] = DEFAULT_ABLATIONS,
    expected_controls: tuple[str, ...] = DEFAULT_CONTROLS,
    min_plot_files: int = 2,
) -> dict[str, Any]:
    base = Path(package_dir)
    if not base.exists() or not base.is_dir():
        return {
            "schema_version": "2026-06-06.phase2_5_module_package_audit.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": "local_read_only_phase2_5_module_package_audit",
            "read_only": True,
            "will_start_experiment": False,
            "will_delete": False,
            "package_dir": str(base),
            "module": module,
            "ok": False,
            "paper_claim_ready": False,
            "failures": [f"missing_package_dir:{base}"],
            "warnings": [],
        }

    if module == "observation_motivation":
        module_audit = _audit_observation(
            base,
            expected_events=expected_events,
            expected_candidates_per_event=expected_candidates_per_event,
            min_join_rate=min_join_rate,
            min_plot_files=min_plot_files,
        )
    elif module == "component_ablation":
        module_audit = _audit_component_ablation(
            base,
            expected_events=expected_events,
            expected_candidates_per_event=expected_candidates_per_event,
            expected_ablations=expected_ablations,
            min_plot_files=min_plot_files,
        )
    elif module == "hyperparameter_analysis":
        module_audit = _audit_hyperparameter(
            base,
            expected_controls=expected_controls,
            min_plot_files=min_plot_files,
        )
    else:
        raise ValueError(f"unsupported module: {module}")

    failures = list(module_audit.get("failures", []))
    warnings = list(module_audit.get("warnings", []))
    return {
        "schema_version": "2026-06-06.phase2_5_module_package_audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "local_read_only_phase2_5_module_package_audit",
        "read_only": True,
        "will_start_experiment": False,
        "will_delete": False,
        "package_dir": str(base),
        "module": module,
        "expected_events": expected_events,
        "expected_candidates_per_event": expected_candidates_per_event,
        "ok": not failures,
        "paper_claim_ready": not failures,
        "module_audit": module_audit,
        "failures": failures,
        "warnings": warnings,
    }


def write_markdown(path: str | Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Phase 2.5 Module Package Audit",
        "",
        f"- Generated UTC: `{audit['created_at_utc']}`",
        f"- Module: `{audit['module']}`",
        f"- Package: `{audit['package_dir']}`",
        f"- OK: `{audit['ok']}`",
        f"- Paper claim ready: `{audit['paper_claim_ready']}`",
        f"- Read only: `{audit['read_only']}`",
        f"- Will start experiment: `{audit['will_start_experiment']}`",
        f"- Will delete: `{audit['will_delete']}`",
        "",
        "## Failures",
        "",
    ]
    lines.extend(f"- {failure}" for failure in audit["failures"] or ["none"])
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in audit["warnings"] or ["none"])
    lines.append("")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    expected_ablations = tuple(args.expected_ablation or DEFAULT_ABLATIONS)
    expected_controls = tuple(args.expected_control or DEFAULT_CONTROLS)
    audit = build_audit(
        module=args.module,
        package_dir=args.package_dir,
        expected_events=args.expected_events,
        expected_candidates_per_event=args.expected_candidates_per_event,
        min_join_rate=args.min_join_rate,
        expected_ablations=expected_ablations,
        expected_controls=expected_controls,
        min_plot_files=args.min_plot_files,
    )
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(args.output_md, audit)
    print(json.dumps({"ok": audit["ok"], "paper_claim_ready": audit["paper_claim_ready"], "failures": audit["failures"]}, indent=2))
    if not audit["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
