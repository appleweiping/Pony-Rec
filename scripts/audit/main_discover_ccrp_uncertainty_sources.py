from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from scripts.audit.main_audit_ccrp_uncertainty_sources import (
    COUNTEREVIDENCE_COLUMNS,
    EVIDENCE_COLUMNS,
    EVENT_COLUMNS,
    ITEM_COLUMNS,
    RAW_PROBABILITY_COLUMNS,
    UNCERTAINTY_COLUMNS,
    USER_COLUMNS,
    _load_candidate_keys,
    audit_source,
)


DEFAULT_NAME_TOKENS = ("ccrp", "shadow", "signal", "calibr", "pointwise", "scored", "selected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover candidate C-CRP uncertainty/signal artifacts by inspecting CSV/JSONL headers. "
            "Use --full_audit to run the stricter row/key audit on discovered candidates."
        )
    )
    parser.add_argument("--root", action="append", default=[], help="Directory to scan. Repeatable. Defaults to outputs.")
    parser.add_argument("--domain", action="append", default=[], help="Domain token to require in path, e.g. sports.")
    parser.add_argument("--name_token", action="append", default=[], help="Filename/path token filter. Defaults to C-CRP/signal tokens.")
    parser.add_argument("--candidate_items_path", default="", help="Optional candidate_items.csv for --full_audit.")
    parser.add_argument("--expected_events", type=int, default=10000)
    parser.add_argument("--expected_candidates_per_event", type=int, default=101)
    parser.add_argument("--max_file_mb", type=float, default=600.0)
    parser.add_argument("--full_audit", action="store_true", help="Read matching files fully and run source audit.")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _header_from_csv(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            return next(csv.reader(fh), [])
    except Exception:
        return []


def _header_from_jsonl(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                row = json.loads(text)
                return list(row.keys()) if isinstance(row, dict) else []
    except Exception:
        return []
    return []


def _has_any(columns: set[str], candidates: Iterable[str]) -> bool:
    return any(col in columns for col in candidates)


def classify_header(columns: Iterable[str]) -> dict[str, Any]:
    column_set = set(columns)
    has_identity = (
        _has_any(column_set, EVENT_COLUMNS)
        and _has_any(column_set, USER_COLUMNS)
        and _has_any(column_set, ITEM_COLUMNS)
    )
    has_uncertainty = _has_any(column_set, UNCERTAINTY_COLUMNS)
    has_recomputable_signal = (
        _has_any(column_set, RAW_PROBABILITY_COLUMNS)
        and _has_any(column_set, EVIDENCE_COLUMNS)
        and _has_any(column_set, COUNTEREVIDENCE_COLUMNS)
    )
    has_score_only = "score" in column_set and not has_uncertainty and not has_recomputable_signal
    if has_identity and has_uncertainty:
        status = "header_paper_ready_candidate"
    elif has_identity and has_recomputable_signal:
        status = "header_recomputable_signal_candidate"
    elif has_identity and has_score_only:
        status = "header_score_only_candidate"
    else:
        status = "header_insufficient"
    return {
        "header_status": status,
        "has_identity": has_identity,
        "has_uncertainty": has_uncertainty,
        "has_recomputable_signal": has_recomputable_signal,
        "has_score_only": has_score_only,
        "columns": sorted(column_set),
    }


def _path_matches(path: Path, *, root: Path, domains: list[str], name_tokens: list[str]) -> bool:
    try:
        text = path.relative_to(root).as_posix().lower()
    except ValueError:
        text = path.as_posix().lower()
    if domains and not any(domain.lower() in text for domain in domains):
        return False
    if name_tokens and not any(token.lower() in text for token in name_tokens):
        return False
    return True


def discover_sources(
    *,
    roots: list[str] | None = None,
    domains: list[str] | None = None,
    name_tokens: list[str] | None = None,
    candidate_items_path: str = "",
    expected_events: int = 10000,
    expected_candidates_per_event: int = 101,
    max_file_mb: float = 600.0,
    full_audit: bool = False,
) -> dict[str, Any]:
    scan_roots = [Path(root) for root in (roots or ["outputs"])]
    domain_filters = domains or []
    token_filters = name_tokens or list(DEFAULT_NAME_TOKENS)
    max_bytes = int(max_file_mb * 1024 * 1024)
    rows: list[dict[str, Any]] = []
    skipped = {"missing_root": 0, "too_large": 0, "unsupported_suffix": 0, "filtered": 0, "empty_header": 0}
    candidate_keys = None
    if full_audit and candidate_items_path:
        loaded = _load_candidate_keys(candidate_items_path)
        candidate_keys = loaded["keys"]

    for root in scan_roots:
        if not root.exists():
            skipped["missing_root"] += 1
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".csv", ".jsonl"}:
                skipped["unsupported_suffix"] += 1
                continue
            if not _path_matches(path, root=root, domains=domain_filters, name_tokens=token_filters):
                skipped["filtered"] += 1
                continue
            size = path.stat().st_size
            if size > max_bytes:
                skipped["too_large"] += 1
                continue
            columns = _header_from_csv(path) if path.suffix.lower() == ".csv" else _header_from_jsonl(path)
            if not columns:
                skipped["empty_header"] += 1
                continue
            row: dict[str, Any] = {
                "path": str(path),
                "size": size,
                **classify_header(columns),
            }
            if row["header_status"] != "header_insufficient":
                if full_audit:
                    audited = audit_source(
                        label=path.stem,
                        path=path,
                        candidate_keys=candidate_keys,
                        expected_events=expected_events,
                        expected_candidates_per_event=expected_candidates_per_event,
                    )
                    row.update({f"audit_{key}": value for key, value in audited.items() if key != "columns"})
                rows.append(row)

    return {
        "roots": [str(root) for root in scan_roots],
        "domains": domain_filters,
        "name_tokens": token_filters,
        "max_file_mb": max_file_mb,
        "full_audit": full_audit,
        "candidate_items_path": candidate_items_path,
        "candidate_count": len(rows),
        "skipped": skipped,
        "sources": rows,
    }


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "size",
        "header_status",
        "has_identity",
        "has_uncertainty",
        "has_recomputable_signal",
        "has_score_only",
        "audit_status",
        "audit_candidate_key_coverage_rate",
        "audit_failures",
        "audit_warnings",
    ]
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in fieldnames}
            out["audit_failures"] = ";".join(row.get("audit_failures") or [])
            out["audit_warnings"] = ";".join(row.get("audit_warnings") or [])
            writer.writerow(out)


def main() -> None:
    args = parse_args()
    payload = discover_sources(
        roots=args.root or ["outputs"],
        domains=args.domain,
        name_tokens=args.name_token or list(DEFAULT_NAME_TOKENS),
        candidate_items_path=args.candidate_items_path,
        expected_events=args.expected_events,
        expected_candidates_per_event=args.expected_candidates_per_event,
        max_file_mb=args.max_file_mb,
        full_audit=args.full_audit,
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
