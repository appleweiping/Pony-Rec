from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
import math
from pathlib import Path
import re
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
LOG_FAILURE_MARKERS = (
    "Traceback",
    "CUDA out of memory",
    "No space left on device",
    "OOM",
    "FAILED",
)
SERVER_COMPARISON_FILES = (
    "local_server_manifest_comparison.json",
    "local_server_evidence_consistency.json",
    "manifest_comparison.json",
)
CONFIG_FILES = ("config.json", "run_config.json", "selected_valid_config.json", "selected_hyperparameters.json")
DEFAULT_CONTROLS = ("eta", "confidence_weight", "weight_grid_label")
SEED_KEYS = ("seed", "seeds", "random_seed", "random_seeds", "seed_list")
HYPERPARAMETER_SUMMARY_AUDIT_COLUMNS = ("audit_ok", "degeneracy_audit_ok", "score_coverage_rate", "candidate_key_count")
MAX_HYPERPARAMETER_RELATIVE_DROP = 0.05
SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")
HASH_EQUALITY_PAIRS = (
    ("local_sha256", "server_sha256"),
    ("local_hash", "server_hash"),
    ("local_digest", "server_digest"),
    ("expected_sha256", "actual_sha256"),
    ("expected_hash", "actual_hash"),
    ("expected_digest", "actual_digest"),
    ("expected", "actual"),
)
FILE_IDENTITY_KEYS = (
    "path",
    "rel_path",
    "relative_path",
    "file",
    "file_path",
    "name",
    "local_path",
    "server_path",
)
HASH_EVIDENCE_CONTAINER_KEYS = {
    "checked_files",
    "files",
    "manifest_checks",
    "rows",
    "artifacts",
    "light_sync_manifest",
    "server_large_artifact_manifest",
}


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


def _has_seed_record(provenance: dict[str, Any], base: Path) -> bool:
    for key in SEED_KEYS:
        value = provenance.get(key)
        if isinstance(value, list) and value:
            return True
        if str(value or "").strip():
            return True
    for name in CONFIG_FILES:
        path = base / name
        if not path.exists() or path.stat().st_size <= 0:
            continue
        try:
            payload = _read_json(path)
        except Exception:
            continue
        for key in SEED_KEYS:
            value = payload.get(key)
            if isinstance(value, list) and value:
                return True
            if str(value or "").strip():
                return True
    return False


def _scan_log_snippets(base: Path, failures: list[str]) -> None:
    for name in _present(base, LOG_FILES):
        text = (base / name).read_text(encoding="utf-8", errors="replace")
        for marker in LOG_FAILURE_MARKERS:
            if marker in text:
                failures.append(f"log_snippet_contains_failure_marker:{name}:{marker}")


def _clean_sha256(value: Any) -> str:
    text = str(value or "").strip()
    if SHA256_HEX_RE.fullmatch(text):
        return text.lower()
    return ""


def _record_status_allows(record: dict[str, Any]) -> bool:
    for key in ("ok", "sha256_ok", "hash_ok", "size_ok", "present", "exists", "local_present", "server_present"):
        if key in record and not _as_bool(record.get(key)):
            return False
    return True


def _has_file_identity(record: dict[str, Any]) -> bool:
    return any(str(record.get(key, "")).strip() for key in FILE_IDENTITY_KEYS)


def _record_has_hash_evidence(record: dict[str, Any]) -> bool:
    if not _record_status_allows(record) or not _has_file_identity(record):
        return False
    for left_key, right_key in HASH_EQUALITY_PAIRS:
        left_hash = _clean_sha256(record.get(left_key))
        right_hash = _clean_sha256(record.get(right_key))
        if left_hash and left_hash == right_hash:
            return True
    sha256 = _clean_sha256(record.get("sha256"))
    if sha256 and (_as_bool(record.get("sha256_ok")) or _as_bool(record.get("hash_ok"))):
        return True
    return False


def _node_has_hash_evidence(value: Any, *, inherited_path: str = "") -> bool:
    if isinstance(value, list):
        return any(_node_has_hash_evidence(item, inherited_path=inherited_path) for item in value)
    if not isinstance(value, dict):
        return False

    record = dict(value)
    if inherited_path and not _has_file_identity(record):
        record["rel_path"] = inherited_path
    if _record_has_hash_evidence(record):
        return True

    for key, child in value.items():
        key_text = str(key)
        child_path = "" if key_text in HASH_EVIDENCE_CONTAINER_KEYS else key_text
        if _node_has_hash_evidence(child, inherited_path=child_path):
            return True
    return False


def _comparison_has_substance(comparison: dict[str, Any]) -> bool:
    return _node_has_hash_evidence(comparison)


def _check_general_package(base: Path, provenance: dict[str, Any], failures: list[str], warnings: list[str]) -> None:
    if not _has_nonempty_command(provenance, base):
        failures.append("missing_command_record")
    if not _present(base, LOG_FILES):
        failures.append("missing_log_snippets")
    else:
        _scan_log_snippets(base, failures)
    if not str(provenance.get("git_commit", "")).strip():
        failures.append("missing_git_commit")
    if not _has_seed_record(provenance, base):
        failures.append("missing_seed_record")
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

    for path in base.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(base).as_posix()
        if path.name == "scores.csv":
            failures.append(f"disallowed_bulk_scores_csv:{rel_path}")
        if path.name == "rank_predictions.jsonl":
            failures.append(f"disallowed_bulk_prediction_jsonl:{rel_path}")
        if path.suffix.lower() in {".pt", ".pth", ".npy", ".pkl", ".npz", ".ckpt", ".safetensors", ".bin"}:
            warnings.append(f"large_or_binary_artifact_in_package:{rel_path}")


def _check_metrics(header: list[str], failures: list[str], *, context: str) -> None:
    missing = [metric for metric in FULL_METRICS if metric not in header]
    if missing:
        failures.append(f"{context}:missing_full_metrics:{','.join(missing)}")


def _check_metric_values(rows: list[dict[str, str]], failures: list[str], *, context: str) -> None:
    for idx, row in enumerate(rows):
        for metric in FULL_METRICS:
            if metric not in row:
                continue
            raw = row.get(metric, "")
            value = _as_float(raw)
            if not math.isfinite(value):
                failures.append(f"{context}:nonfinite_metric:{idx}:{metric}:{raw}")
            elif value < 0.0 or value > 1.0:
                failures.append(f"{context}:metric_out_of_range:{idx}:{metric}:{raw}")


def _check_scalar_metric_values(rows: list[dict[str, str]], failures: list[str], *, context: str, column: str) -> None:
    for idx, row in enumerate(rows):
        raw = row.get(column, "")
        value = _as_float(raw)
        if not math.isfinite(value):
            failures.append(f"{context}:nonfinite_metric:{idx}:{column}:{raw}")
        elif value < 0.0 or value > 1.0:
            failures.append(f"{context}:metric_out_of_range:{idx}:{column}:{raw}")


def _hyperparameter_summary_value_counts(
    rows: list[dict[str, str]],
    *,
    expected_controls: tuple[str, ...],
    expected_metric: str,
    min_values: int,
    failures: list[str],
) -> dict[tuple[str, str], int]:
    allowed_splits = {"valid", "test"}
    grouped: dict[tuple[str, str], set[str]] = {}
    expected_control_set = set(expected_controls)
    seen_points: set[tuple[str, str, str]] = set()
    for idx, row in enumerate(rows):
        split = str(row.get("split", "")).strip()
        control = str(row.get("control", "")).strip()
        control_value = str(row.get("control_value", "")).strip()
        metric_name = str(row.get("metric_name", "")).strip()
        if split not in allowed_splits:
            failures.append(f"hyperparameter_summary_unexpected_split:{idx}:{split}")
        if not control_value:
            failures.append(f"hyperparameter_summary_empty_control_value:{idx}:{split}:{control}")
        if expected_metric and metric_name != expected_metric:
            failures.append(f"hyperparameter_summary_unexpected_metric:{idx}:{metric_name}!={expected_metric}")
        if split in allowed_splits and control in expected_control_set and control_value:
            point_key = (split, control, control_value)
            if point_key in seen_points:
                failures.append(f"hyperparameter_summary_duplicate_curve_point:{split}:{control}:{control_value}")
            seen_points.add(point_key)
            grouped.setdefault((split, control), set()).add(control_value)
        candidate_rows_for_value = _as_int(row.get("candidate_rows_for_value"), default=-1)
        if candidate_rows_for_value != 1:
            failures.append(f"hyperparameter_summary_candidate_rows_for_value:{idx}:{candidate_rows_for_value}!=1")

    counts = {key: len(values) for key, values in grouped.items()}
    for split in ("valid", "test"):
        for control in expected_controls:
            count = counts.get((split, control), 0)
            if count == 0:
                failures.append(f"hyperparameter_summary_missing_split_control:{split}:{control}")
            elif count < min_values:
                failures.append(f"hyperparameter_summary_control_too_short:{split}:{control}:{count}<{min_values}")
    return counts


def _check_hyperparameter_summary_audit_columns(
    header: list[str],
    rows: list[dict[str, str]],
    failures: list[str],
    *,
    expected_candidate_key_count: int,
) -> None:
    for column in HYPERPARAMETER_SUMMARY_AUDIT_COLUMNS:
        if column not in header:
            failures.append(f"hyperparameter_summary_missing_audit_column:{column}")
    if any(column not in header for column in HYPERPARAMETER_SUMMARY_AUDIT_COLUMNS):
        return
    for idx, row in enumerate(rows):
        if not _as_bool(row.get("audit_ok")):
            failures.append(f"hyperparameter_summary_audit_false:{idx}:audit_ok")
        if not _as_bool(row.get("degeneracy_audit_ok")):
            failures.append(f"hyperparameter_summary_audit_false:{idx}:degeneracy_audit_ok")
        coverage = _as_float(row.get("score_coverage_rate"))
        if not math.isfinite(coverage) or abs(coverage - 1.0) > 1e-12:
            failures.append(f"hyperparameter_summary_score_coverage_not_one:{idx}:{row.get('score_coverage_rate')}")
        key_count = _as_int(row.get("candidate_key_count"), default=-1)
        if expected_candidate_key_count > 0 and key_count != expected_candidate_key_count:
            failures.append(
                f"hyperparameter_summary_candidate_key_count:{idx}:{key_count}!={expected_candidate_key_count}"
            )


def _check_hyperparameter_stability_report(
    provenance: dict[str, Any],
    *,
    expected_controls: tuple[str, ...],
    expected_metric: str,
    summary_rows: list[dict[str, str]],
    failures: list[str],
) -> None:
    report = provenance.get("stability_report")
    if not isinstance(report, list):
        failures.append("hyperparameter:missing_stability_report")
        return
    report_by_control: dict[str, dict[str, Any]] = {}
    for row in report:
        if not isinstance(row, dict):
            failures.append("hyperparameter_stability_report_bad_row")
            continue
        control = str(row.get("control", "")).strip()
        if not control:
            failures.append("hyperparameter_stability_report_empty_control")
            continue
        if control in report_by_control:
            failures.append(f"hyperparameter_stability_duplicate_control:{control}")
        if control not in set(expected_controls):
            failures.append(f"hyperparameter_stability_unexpected_control:{control}")
        report_by_control[control] = row
    for control in expected_controls:
        row = report_by_control.get(control)
        if not row:
            failures.append(f"hyperparameter_missing_stability_report:{control}")
            continue
        expected = _expected_hyperparameter_stability(summary_rows, control=control)
        if expected:
            for key in ("valid_best_value", "test_best_value"):
                actual_text = str(row.get(key, "")).strip()
                if actual_text != str(expected[key]):
                    failures.append(f"hyperparameter_stability_report_mismatch:{control}:{key}:{actual_text}!={expected[key]}")
            for key in ("valid_metric_at_best", "test_best_metric", "test_metric_at_valid_best", "relative_drop_from_test_best"):
                actual_value = _as_float(row.get(key))
                expected_value = _as_float(expected[key])
                if not math.isfinite(actual_value) or abs(actual_value - expected_value) > 1e-9:
                    failures.append(
                        f"hyperparameter_stability_report_mismatch:{control}:{key}:{row.get(key)}!={expected_value}"
                    )
            actual_rank = _as_int(row.get("test_rank_of_valid_best"), default=-1)
            if actual_rank != int(expected["test_rank_of_valid_best"]):
                failures.append(
                    f"hyperparameter_stability_report_mismatch:{control}:test_rank_of_valid_best:{actual_rank}!={expected['test_rank_of_valid_best']}"
                )
        if str(row.get("metric", "")).strip() != expected_metric:
            failures.append(f"hyperparameter_stability_metric_mismatch:{control}:{row.get('metric')}!={expected_metric}")
        for key in ("valid_best_value", "test_best_value"):
            if not str(row.get(key, "")).strip():
                failures.append(f"hyperparameter_stability_missing_field:{control}:{key}")
        for key in ("valid_metric_at_best", "test_best_metric", "test_metric_at_valid_best", "relative_drop_from_test_best"):
            value = _as_float(row.get(key))
            if not math.isfinite(value):
                failures.append(f"hyperparameter_stability_nonfinite:{control}:{key}:{row.get(key)}")
        test_rank = _as_int(row.get("test_rank_of_valid_best"), default=-1)
        if test_rank <= 0:
            failures.append(f"hyperparameter_stability_bad_test_rank:{control}:{row.get('test_rank_of_valid_best')}")
        drop = _as_float(row.get("relative_drop_from_test_best"))
        tolerance = _as_float(row.get("relative_drop_tolerance"), default=MAX_HYPERPARAMETER_RELATIVE_DROP)
        if not math.isfinite(tolerance) or tolerance < 0.0 or tolerance > MAX_HYPERPARAMETER_RELATIVE_DROP:
            failures.append(f"hyperparameter_stability_bad_tolerance:{control}:{row.get('relative_drop_tolerance')}")
            tolerance = MAX_HYPERPARAMETER_RELATIVE_DROP
        if not _as_bool(row.get("has_valid_and_test")):
            failures.append(f"hyperparameter_stability_missing_valid_or_test:{control}")
        if not _as_bool(row.get("stable_within_tolerance")):
            failures.append(f"hyperparameter_stability_report_not_stable:{control}:{row.get('relative_drop_from_test_best')}")
        if math.isfinite(drop) and drop > tolerance + 1e-12:
            failures.append(f"hyperparameter_stability_drop_exceeds_tolerance:{control}:{drop}>{tolerance}")


def _expected_hyperparameter_stability(rows: list[dict[str, str]], *, control: str) -> dict[str, Any]:
    valid = [row for row in rows if str(row.get("split", "")).strip() == "valid" and str(row.get("control", "")).strip() == control]
    test = [row for row in rows if str(row.get("split", "")).strip() == "test" and str(row.get("control", "")).strip() == control]
    if not valid or not test:
        return {}

    def best(rows_for_split: list[dict[str, str]]) -> dict[str, str]:
        return max(rows_for_split, key=lambda row: _as_float(row.get("metric_value")))

    valid_best = best(valid)
    test_best = best(test)
    valid_best_value = str(valid_best.get("control_value", "")).strip()
    test_best_value = str(test_best.get("control_value", "")).strip()
    test_at_valid = [row for row in test if str(row.get("control_value", "")).strip() == valid_best_value]
    if not test_at_valid:
        return {
            "valid_best_value": valid_best_value,
            "test_best_value": test_best_value,
            "valid_metric_at_best": _as_float(valid_best.get("metric_value")),
            "test_best_metric": _as_float(test_best.get("metric_value")),
            "test_metric_at_valid_best": float("nan"),
            "relative_drop_from_test_best": float("nan"),
            "test_rank_of_valid_best": -1,
        }
    valid_metric_at_best = _as_float(valid_best.get("metric_value"))
    test_best_metric = _as_float(test_best.get("metric_value"))
    test_metric_at_valid_best = _as_float(test_at_valid[0].get("metric_value"))
    denominator = abs(test_best_metric)
    if denominator <= 1e-12:
        relative_drop = 0.0 if abs(test_best_metric - test_metric_at_valid_best) <= 1e-12 else float("inf")
    else:
        relative_drop = max(0.0, (test_best_metric - test_metric_at_valid_best) / denominator)
    ranked_test = sorted(test, key=lambda row: _as_float(row.get("metric_value")), reverse=True)
    rank_map = {str(row.get("control_value", "")).strip(): idx + 1 for idx, row in enumerate(ranked_test)}
    return {
        "valid_best_value": valid_best_value,
        "test_best_value": test_best_value,
        "valid_metric_at_best": valid_metric_at_best,
        "test_best_metric": test_best_metric,
        "test_metric_at_valid_best": test_metric_at_valid_best,
        "relative_drop_from_test_best": relative_drop,
        "test_rank_of_valid_best": rank_map.get(valid_best_value, -1),
    }


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


def _check_component_config_matches_main(
    row: dict[str, str],
    main_provenance: dict[str, Any],
    failures: list[str],
) -> None:
    if not main_provenance:
        return
    ablation = str(row.get("ablation", "")).strip() or "unknown"
    for key in ("score_mode", "weight_grid_label"):
        expected = str(main_provenance.get(key, "")).strip()
        if not expected or key not in row:
            continue
        actual = str(row.get(key, "")).strip()
        if actual != expected:
            failures.append(f"component_ablation_config_mismatch:{ablation}:{key}:{actual}!={expected}")
    for key in ("eta", "confidence_weight"):
        if key not in main_provenance or key not in row:
            continue
        expected = _as_float(main_provenance.get(key))
        actual = _as_float(row.get(key))
        if not math.isfinite(expected) or not math.isfinite(actual):
            failures.append(f"component_ablation_config_nonfinite:{ablation}:{key}:{row.get(key)}!={main_provenance.get(key)}")
        elif abs(actual - expected) > 1e-12:
            failures.append(f"component_ablation_config_mismatch:{ablation}:{key}:{actual}!={expected}")


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
    _check_metric_values(summary_rows, failures, context="observation_summary")
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
    _check_metric_values(summary_rows, failures, context="component_ablation_summary")
    for column in ("audit_ok", "degeneracy_audit_ok", "score_coverage_rate"):
        if column not in summary_header:
            failures.append(f"component_ablation_summary_missing_column:{column}")
    selected_header, selected_rows = _load_required_csv(base, "selected_test_metrics.csv", failures)
    _check_metrics(selected_header, failures, context="selected_test_metrics")
    _check_metric_values(selected_rows, failures, context="selected_test_metrics")
    for column in ("audit_ok", "degeneracy_audit_ok", "score_coverage_rate", "candidate_key_count"):
        if column not in selected_header:
            failures.append(f"selected_test_metrics_missing_column:{column}")
    _load_required_json(base, "selected_valid_config.json", failures)
    internal_provenance = _load_required_json(base, "ccrp_internal_provenance.json", failures)
    ranking_header, ranking_rows = _load_required_csv(base, "tables/ranking_metrics.csv", failures)
    _check_metrics(ranking_header, failures, context="ranking_metrics_table")
    _check_metric_values(ranking_rows, failures, context="ranking_metrics_table")
    coverage_header, coverage_rows = _load_required_csv(base, "tables/external_score_coverage.csv", failures)
    baseline_summary_header, baseline_summary_rows = _load_required_csv(
        base, "tables/same_candidate_external_baseline_summary.csv", failures
    )
    ranking_eval_header, ranking_eval_rows = _load_required_csv(base, "tables/ranking_eval_records.csv", failures)
    if provenance.get("artifact_class") != "paper_critical_component_ablation":
        failures.append("component_ablation:unexpected_artifact_class")
    if provenance.get("status_label") != "paper_critical_component_ablation_ready":
        failures.append("component_ablation:unexpected_status_label")
    if provenance.get("ok") is not True:
        failures.append("component_ablation:provenance_not_ok")
    if not internal_provenance:
        failures.append("component_ablation:missing_internal_provenance")

    summary_abls = {str(row.get("ablation", "")).strip() for row in summary_rows}
    sweep_abls = {str(row.get("ablation", "")).strip() for row in sweep_rows}
    for ablation in expected_ablations:
        if ablation not in summary_abls:
            failures.append(f"component_ablation_summary_missing:{ablation}")
    if "full" not in sweep_abls:
        failures.append("valid_sweep_missing_main_ablation:full")
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
        _check_component_config_matches_main(row, internal_provenance, failures)
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
    for column in ("ranking_events", "total_candidates", "matched_candidates"):
        if column not in coverage_header:
            failures.append(f"external_score_coverage_missing_column:{column}")
    expected_score_keys = expected_events * expected_candidates_per_event if expected_events > 0 else 0
    for row in coverage_rows:
        if _as_float(row.get("score_coverage_rate"), 0.0) != 1.0:
            failures.append(f"external_score_coverage_not_one:{row.get('baseline_name')}:{row.get('score_coverage_rate')}")
        if expected_events > 0 and _as_int(row.get("ranking_events")) != expected_events:
            failures.append(f"external_score_coverage_ranking_events:{row.get('baseline_name')}:{row.get('ranking_events')}!={expected_events}")
        if expected_score_keys and _as_int(row.get("total_candidates")) != expected_score_keys:
            failures.append(
                f"external_score_coverage_total_candidates:{row.get('baseline_name')}:{row.get('total_candidates')}!={expected_score_keys}"
            )
        if expected_score_keys and _as_int(row.get("matched_candidates")) != expected_score_keys:
            failures.append(
                f"external_score_coverage_matched_candidates:{row.get('baseline_name')}:{row.get('matched_candidates')}!={expected_score_keys}"
            )
    if expected_events > 0 and len(ranking_eval_rows) != expected_events:
        failures.append(f"ranking_eval_records_row_count:{len(ranking_eval_rows)}!={expected_events}")
    if "status_label" not in baseline_summary_header:
        failures.append("same_candidate_summary_missing_status_label")
    if not baseline_summary_rows:
        failures.append("same_candidate_summary_empty")
    for row in baseline_summary_rows:
        status = str(row.get("status_label", "")).strip()
        if status not in {"paper_critical_component_ablation_row", "same_schema_internal_ablation"}:
            failures.append(f"same_candidate_summary_bad_status:{row.get('baseline_name')}:{status}")
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


def _audit_hyperparameter(
    base: Path,
    *,
    expected_events: int,
    expected_candidates_per_event: int,
    expected_controls: tuple[str, ...],
    min_plot_files: int,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    summary_header, summary_rows = _load_required_csv(base, "ccrp_hyperparameter_curve_summary.csv", failures)
    provenance = _load_required_json(base, "ccrp_hyperparameter_curve_provenance.json", failures)

    _check_general_package(base, provenance, failures, warnings)
    for column in ("split", "control", "control_value", "metric_name", "metric_value"):
        if column not in summary_header:
            failures.append(f"hyperparameter_summary_missing_column:{column}")
    _check_scalar_metric_values(summary_rows, failures, context="hyperparameter_summary", column="metric_value")
    _check_hyperparameter_summary_audit_columns(
        summary_header,
        summary_rows,
        failures,
        expected_candidate_key_count=expected_events * expected_candidates_per_event,
    )
    if provenance.get("artifact_class") != "paper_critical_hyperparameter_analysis":
        failures.append("hyperparameter:unexpected_artifact_class")
    if provenance.get("status_label") != "paper_critical_hyperparameter_curve_ready":
        failures.append("hyperparameter:unexpected_status_label")
    if provenance.get("paper_claim_scope") != "valid_and_test_stability_curve_candidate":
        failures.append("hyperparameter:unexpected_paper_claim_scope")
    if provenance.get("reporting_mode") != "valid_and_test":
        failures.append("hyperparameter:not_valid_and_test")
    sweep_sha_raw = str(provenance.get("sweep_sha256", "")).strip()
    test_sweep_sha_raw = str(provenance.get("test_sweep_sha256", "")).strip()
    sweep_sha = _clean_sha256(sweep_sha_raw)
    test_sweep_sha = _clean_sha256(test_sweep_sha_raw)
    if not sweep_sha_raw:
        failures.append("hyperparameter:missing_sweep_sha256")
    elif not sweep_sha:
        failures.append(f"hyperparameter:invalid_sweep_sha256:{sweep_sha_raw}")
    if not test_sweep_sha_raw:
        failures.append("hyperparameter:missing_test_sweep_sha256")
    elif not test_sweep_sha:
        failures.append(f"hyperparameter:invalid_test_sweep_sha256:{test_sweep_sha_raw}")
    if sweep_sha and test_sweep_sha and sweep_sha == test_sweep_sha:
        failures.append("hyperparameter:valid_test_sweep_hash_equal")
    expected_metric = str(provenance.get("metric", "")).strip()
    if not expected_metric:
        failures.append("hyperparameter:missing_metric")
    elif expected_metric not in FULL_METRICS:
        failures.append(f"hyperparameter:unsupported_metric:{expected_metric}")
    min_values = _as_int(provenance.get("min_values"), default=-1)
    if min_values < 3:
        failures.append(f"hyperparameter:min_values_too_low:{provenance.get('min_values')}")
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
    if 0 < int(audit_summary.get("audited_rows") or 0) < len(summary_rows):
        failures.append(f"hyperparameter:audited_rows_less_than_summary:{audit_summary.get('audited_rows')}<{len(summary_rows)}")
    controls = [str(control) for control in provenance.get("controls", [])]
    summary_value_counts = _hyperparameter_summary_value_counts(
        summary_rows,
        expected_controls=expected_controls,
        expected_metric=expected_metric,
        min_values=max(min_values, 3),
        failures=failures,
    )
    _check_hyperparameter_stability_report(
        provenance,
        expected_controls=expected_controls,
        expected_metric=expected_metric,
        summary_rows=summary_rows,
        failures=failures,
    )
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
        split = str(row.get("split", "")).strip()
        control = str(row.get("control", "")).strip()
        if split in {"valid", "test"} and control in set(expected_controls):
            reported_values = _as_int(row.get("curve_values"), default=-1)
            summary_values = summary_value_counts.get((split, control), 0)
            if reported_values != summary_values:
                failures.append(
                    f"hyperparameter_control_report_value_mismatch:{split}:{control}:{reported_values}!={summary_values}"
                )
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
            expected_events=expected_events,
            expected_candidates_per_event=expected_candidates_per_event,
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
