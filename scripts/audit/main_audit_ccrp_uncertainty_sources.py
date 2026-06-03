from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


EVENT_COLUMNS = ("source_event_id", "event_id", "user_id")
USER_COLUMNS = ("user_id",)
ITEM_COLUMNS = ("candidate_item_id", "item_id")
UNCERTAINTY_COLUMNS = ("ccrp_uncertainty", "uncertainty", "shadow_uncertainty", "risk_uncertainty")
RAW_PROBABILITY_COLUMNS = ("relevance_probability", "shadow_primary_score", "shadow_score", "confidence")
CALIBRATED_PROBABILITY_COLUMNS = (
    "calibrated_relevance_probability",
    "shadow_calibrated_score",
    "calibrated_confidence",
)
EVIDENCE_COLUMNS = ("evidence_support", "evidence")
COUNTEREVIDENCE_COLUMNS = ("counterevidence_strength", "counterevidence")
CCRP_COMPONENT_COLUMNS = (
    "ccrp_boundary_uncertainty",
    "ccrp_calibration_gap",
    "ccrp_evidence_uncertainty",
    "ccrp_risk_adjusted_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether candidate C-CRP/signal artifacts contain real uncertainty "
            "fields or only final score rows. This is read-only and is intended before "
            "paper-critical observation/ablation launches."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Source artifact to audit. Repeat for multiple files. LABEL= is optional.",
    )
    parser.add_argument("--candidate_items_path", default="", help="Optional same-candidate item CSV for key coverage.")
    parser.add_argument("--expected_events", type=int, default=10000)
    parser.add_argument("--expected_candidates_per_event", type=int, default=101)
    parser.add_argument("--min_candidate_key_coverage", type=float, default=0.999)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _finite_float(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def _pick(record: dict[str, Any], columns: Iterable[str]) -> str:
    for col in columns:
        value = _text(record.get(col))
        if value:
            return value
    return ""


def _has_any(columns: set[str], candidates: Iterable[str]) -> bool:
    return any(col in columns for col in candidates)


def _first_present(columns: set[str], candidates: Iterable[str]) -> str:
    for col in candidates:
        if col in columns:
            return col
    return ""


def _iter_records(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if text:
                    row = json.loads(text)
                    if isinstance(row, dict):
                        yield row
        return
    with path.open(newline="", encoding="utf-8") as fh:
        yield from csv.DictReader(fh)


def _load_candidate_keys(path: str | Path) -> dict[str, Any]:
    candidate_path = Path(path)
    keys: set[tuple[str, str, str]] = set()
    events: set[str] = set()
    row_count = 0
    for row in _iter_records(candidate_path):
        row_count += 1
        event_id = _pick(row, EVENT_COLUMNS)
        user_id = _pick(row, USER_COLUMNS)
        item_id = _pick(row, ITEM_COLUMNS)
        if event_id:
            events.add(event_id)
        if event_id and user_id and item_id:
            keys.add((event_id, user_id, item_id))
    return {
        "path": str(candidate_path),
        "row_count": row_count,
        "event_count": len(events),
        "key_count": len(keys),
        "keys": keys,
    }


def _source_label_and_path(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        label = label.strip()
        source_path = Path(path.strip())
        return label or source_path.stem, source_path
    source_path = Path(spec.strip())
    return source_path.stem, source_path


def audit_source(
    *,
    label: str,
    path: str | Path,
    candidate_keys: set[tuple[str, str, str]] | None = None,
    expected_events: int = 10000,
    expected_candidates_per_event: int = 101,
    min_candidate_key_coverage: float = 0.999,
) -> dict[str, Any]:
    source_path = Path(path)
    if not source_path.exists() or not source_path.is_file():
        return {
            "label": label,
            "path": str(source_path),
            "exists": False,
            "status": "missing",
            "paper_ready_uncertainty_rows": False,
            "recomputable_signal_rows": False,
            "failures": ["missing_source_file"],
        }

    columns: set[str] = set()
    events: set[str] = set()
    keys: set[tuple[str, str, str]] = set()
    duplicate_keys = 0
    row_count = 0
    finite_uncertainty_rows = 0
    uncertainty_rows = 0

    for row in _iter_records(source_path):
        row_count += 1
        columns.update(str(col) for col in row.keys())
        event_id = _pick(row, EVENT_COLUMNS)
        user_id = _pick(row, USER_COLUMNS)
        item_id = _pick(row, ITEM_COLUMNS)
        if event_id:
            events.add(event_id)
        if event_id and user_id and item_id:
            key = (event_id, user_id, item_id)
            if key in keys:
                duplicate_keys += 1
            keys.add(key)
        for col in UNCERTAINTY_COLUMNS:
            if col in row:
                uncertainty_rows += 1
                if math.isfinite(_finite_float(row.get(col))):
                    finite_uncertainty_rows += 1
                break

    expected_score_rows = expected_events * expected_candidates_per_event
    has_event_key = _has_any(columns, EVENT_COLUMNS)
    has_user_id = _has_any(columns, USER_COLUMNS)
    has_item_key = _has_any(columns, ITEM_COLUMNS)
    uncertainty_col = _first_present(columns, UNCERTAINTY_COLUMNS)
    has_uncertainty = bool(uncertainty_col)
    has_raw_probability = _has_any(columns, RAW_PROBABILITY_COLUMNS)
    has_calibrated_probability = _has_any(columns, CALIBRATED_PROBABILITY_COLUMNS)
    has_evidence = _has_any(columns, EVIDENCE_COLUMNS)
    has_counterevidence = _has_any(columns, COUNTEREVIDENCE_COLUMNS)
    has_ccrp_components = all(col in columns for col in CCRP_COMPONENT_COLUMNS)
    has_identity = has_event_key and has_user_id and has_item_key
    recomputable_signal_rows = has_identity and has_raw_probability and has_evidence and has_counterevidence

    candidate_key_count = len(candidate_keys) if candidate_keys is not None else 0
    matched_candidate_keys = len(keys & candidate_keys) if candidate_keys is not None else 0
    extra_source_keys = len(keys - candidate_keys) if candidate_keys is not None else 0
    missing_candidate_keys = len(candidate_keys - keys) if candidate_keys is not None else 0
    coverage_rate = (matched_candidate_keys / candidate_key_count) if candidate_key_count else None

    event_count_ok = len(events) == expected_events
    score_row_count_ok = row_count == expected_score_rows
    uncertainty_rows_ok = has_uncertainty and finite_uncertainty_rows > 0
    candidate_coverage_ok = coverage_rate is None or coverage_rate >= min_candidate_key_coverage
    paper_ready_uncertainty_rows = (
        has_identity
        and uncertainty_rows_ok
        and event_count_ok
        and candidate_coverage_ok
        and (candidate_keys is None or matched_candidate_keys == candidate_key_count)
    )

    failures: list[str] = []
    warnings: list[str] = []
    if not has_event_key:
        failures.append("missing_event_key_column")
    if not has_user_id:
        failures.append("missing_user_id_column")
    if not has_item_key:
        warnings.append("missing_item_key_column")
    if not has_uncertainty:
        failures.append("missing_uncertainty_column")
    if has_uncertainty and finite_uncertainty_rows == 0:
        failures.append("no_finite_uncertainty_rows")
    if len(events) != expected_events:
        warnings.append("event_count_mismatch")
    if row_count != expected_score_rows:
        warnings.append("row_count_not_full_candidate_grid")
    if coverage_rate is not None and coverage_rate < min_candidate_key_coverage:
        failures.append("candidate_key_coverage_below_threshold")
    if candidate_keys is not None and missing_candidate_keys:
        warnings.append("missing_candidate_keys")
    if candidate_keys is not None and extra_source_keys:
        warnings.append("extra_source_keys")
    if duplicate_keys:
        warnings.append("duplicate_source_keys")

    if paper_ready_uncertainty_rows:
        status = "paper_ready_uncertainty_rows"
    elif recomputable_signal_rows:
        status = "recomputable_signal_rows"
    elif "score" in columns and not has_uncertainty and not recomputable_signal_rows:
        status = "score_only_not_uncertainty"
    elif has_uncertainty:
        status = "uncertainty_rows_incomplete"
    else:
        status = "insufficient"

    return {
        "label": label,
        "path": str(source_path),
        "exists": True,
        "status": status,
        "paper_ready_uncertainty_rows": paper_ready_uncertainty_rows,
        "recomputable_signal_rows": recomputable_signal_rows,
        "row_count": row_count,
        "event_count": len(events),
        "expected_events": expected_events,
        "expected_score_rows": expected_score_rows,
        "key_count": len(keys),
        "duplicate_keys": duplicate_keys,
        "candidate_key_count": candidate_key_count,
        "matched_candidate_keys": matched_candidate_keys,
        "missing_candidate_keys": missing_candidate_keys,
        "extra_source_keys": extra_source_keys,
        "candidate_key_coverage_rate": coverage_rate,
        "uncertainty_col": uncertainty_col,
        "finite_uncertainty_rows": finite_uncertainty_rows,
        "uncertainty_rows": uncertainty_rows,
        "has_event_key": has_event_key,
        "has_user_id": has_user_id,
        "has_item_key": has_item_key,
        "has_raw_probability": has_raw_probability,
        "has_calibrated_probability": has_calibrated_probability,
        "has_evidence": has_evidence,
        "has_counterevidence": has_counterevidence,
        "has_ccrp_components": has_ccrp_components,
        "columns": sorted(columns),
        "failures": failures,
        "warnings": warnings,
    }


def audit_sources(
    *,
    sources: list[str],
    candidate_items_path: str = "",
    expected_events: int = 10000,
    expected_candidates_per_event: int = 101,
    min_candidate_key_coverage: float = 0.999,
) -> dict[str, Any]:
    candidate_summary: dict[str, Any] | None = None
    candidate_keys: set[tuple[str, str, str]] | None = None
    if candidate_items_path:
        loaded = _load_candidate_keys(candidate_items_path)
        candidate_keys = loaded.pop("keys")
        candidate_summary = loaded
    rows = [
        audit_source(
            label=label,
            path=path,
            candidate_keys=candidate_keys,
            expected_events=expected_events,
            expected_candidates_per_event=expected_candidates_per_event,
            min_candidate_key_coverage=min_candidate_key_coverage,
        )
        for label, path in (_source_label_and_path(spec) for spec in sources)
    ]
    return {
        "expected_events": expected_events,
        "expected_candidates_per_event": expected_candidates_per_event,
        "min_candidate_key_coverage": min_candidate_key_coverage,
        "candidate_items": candidate_summary,
        "sources": rows,
        "paper_ready_count": sum(1 for row in rows if row.get("paper_ready_uncertainty_rows")),
        "recomputable_signal_count": sum(1 for row in rows if row.get("recomputable_signal_rows")),
    }


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "path",
        "status",
        "paper_ready_uncertainty_rows",
        "recomputable_signal_rows",
        "row_count",
        "event_count",
        "key_count",
        "candidate_key_count",
        "candidate_key_coverage_rate",
        "uncertainty_col",
        "finite_uncertainty_rows",
        "failures",
        "warnings",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in fieldnames}
            out["failures"] = ";".join(row.get("failures") or [])
            out["warnings"] = ";".join(row.get("warnings") or [])
            writer.writerow(out)


def main() -> None:
    args = parse_args()
    if not args.source:
        raise SystemExit("At least one --source is required.")
    payload = audit_sources(
        sources=args.source,
        candidate_items_path=args.candidate_items_path,
        expected_events=args.expected_events,
        expected_candidates_per_event=args.expected_candidates_per_event,
        min_candidate_key_coverage=args.min_candidate_key_coverage,
    )
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_csv:
        _write_csv(args.output_csv, payload["sources"])
    if not args.quiet:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
