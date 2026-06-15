from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs/summary/paper_critical")
DEFAULT_STAMP = "20260613"


def _consistency_audit_json_for_stamp(stamp: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"final_blocker_consistency_audit_{stamp}.json"


DEFAULT_CONSISTENCY_AUDIT = _consistency_audit_json_for_stamp(DEFAULT_STAMP)
DEFAULT_DOCS = [
    Path("docs/active_todo_pony_uncertainty.md"),
    Path("docs/paper_claims_and_status.md"),
    Path("docs/milestones/README.md"),
    Path("docs/server_runbook.md"),
]

CURRENT_SECTION_BOUNDS = {
    "docs/active_todo_pony_uncertainty.md": ("## Current Checkpoint", "## Hard Invariants"),
    "docs/paper_claims_and_status.md": ("## Paper-critical readiness modules", "## Not primary claims"),
    "docs/milestones/README.md": (
        "## Current Evidence Integrity",
        "## Current Evidence Integrity (updated 2026-06-06)",
    ),
    "docs/server_runbook.md": ("## Current Priority Order", "## Key Scripts"),
}

NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _repo_relative_key(path: Path, root: Path) -> str:
    return _repo_relative(path, root).replace("\\", "/")


def _path_state(path: Path, root: Path) -> dict[str, Any]:
    state = {
        "path": _repo_relative(path, root),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        "sha256": "",
    }
    if path.exists() and path.is_file():
        state["sha256"] = _sha256_file(path)
    return state


def _read_json(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.exists():
        return {}, [f"missing_json:{path}"]
    if not path.is_file():
        return {}, [f"not_a_file:{path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, [f"invalid_json:{path}:{exc.msg}"]
    if not isinstance(payload, dict):
        return {}, [f"json_not_object:{path}"]
    return payload, []


def _read_text(path: Path) -> tuple[str, list[str]]:
    if not path.exists():
        return "", [f"missing_doc:{path}"]
    if not path.is_file():
        return "", [f"not_a_file:{path}"]
    return path.read_text(encoding="utf-8", errors="replace"), []


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _current_section(text: str, start_marker: str, end_marker: str | None) -> tuple[str, int, int, list[str]]:
    lines = text.splitlines()
    start_index: int | None = None
    end_index = len(lines)
    for idx, line in enumerate(lines):
        if line.strip().startswith(start_marker):
            start_index = idx
            break
    if start_index is None:
        return "", 0, 0, [f"missing_current_section:{start_marker}"]
    if end_marker:
        for idx in range(start_index + 1, len(lines)):
            if lines[idx].strip().startswith(end_marker):
                end_index = idx
                break
    return "\n".join(lines[start_index:end_index]), start_index + 1, end_index, []


def _contains_count(text: str, count: int) -> bool:
    word = NUMBER_WORDS.get(count, str(count))
    patterns = [
        rf"failed\s+Claude\s+attempts\s+`?{count}`?",
        rf"failed\s+attempts\s+`?{count}`?",
        rf"{word}\s+failed\s+Claude\s+attempts",
        rf"{word}\s+failed\s+attempts",
        rf"reports\s+{word}\s+failed",
        rf"records\s+{word}\s+failed",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _stale_count_hits(section: str, *, latest_count: int, start_line: int) -> list[dict[str, Any]]:
    if latest_count <= 0:
        return []
    hits: list[dict[str, Any]] = []
    for offset, line in enumerate(section.splitlines()):
        if "at that point" in line.lower() or "superseded" in line.lower():
            continue
        for stale_count in range(1, latest_count):
            stale_word = NUMBER_WORDS.get(stale_count, str(stale_count))
            stale_number = rf"(?<!\d){stale_count}(?!\d)"
            patterns = [
                rf"failed\s+Claude\s+attempts\s+`?{stale_number}`?",
                rf"failed_review_attempts`\s+count\s+`?{stale_number}`?",
                rf"current\s+packet\s+reports\s+{stale_word}\s+failed",
                rf"now\s+records\s+{stale_word}\s+failed",
                rf"now\s+records\s+failed\s+Claude\s+attempts\s+`?{stale_number}`?",
                rf"now\s+reports\s+{stale_word}\s+failed",
                rf"refreshed\s+packet\s+reports.*`?{stale_number}`?",
            ]
            if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in patterns):
                hits.append(
                    {
                        "line": start_line + offset,
                        "stale_count": stale_count,
                        "text": line.strip(),
                    }
                )
                break
    return hits


def _stale_blocker_taxonomy_hits(section: str, *, start_line: int) -> list[dict[str, Any]]:
    patterns = [
        r"two\s+remaining\s+blocker\s+classes",
        r"final\s+two\s+blocker\s+classes",
    ]
    hits: list[dict[str, Any]] = []
    for offset, line in enumerate(section.splitlines()):
        lower = line.lower()
        if "then-two" in lower or "superseded" in lower:
            continue
        for pattern in patterns:
            if re.search(pattern, line, flags=re.IGNORECASE):
                hits.append(
                    {
                        "line": start_line + offset,
                        "pattern": pattern,
                        "text": line.strip(),
                    }
                )
    return hits


def _manual_blocked_present(section: str) -> bool:
    lower = re.sub(r"\s+", " ", section.lower())
    return (
        "manual_confirmation_needed=true" in section
        or "manual_submission_system_ready=false" in section
        or "manual confirmation still needed" in lower
        or ("manual submission-system" in lower and ("remain open" in lower or "blocked" in lower))
    )


def _promax_blocked_present(section: str) -> bool:
    return (
        "promax_public_metadata_ready=false" in section
        or ("ProMax" in section and "404" in section)
        or ("ProMax" in section and "page-range" in section)
    )


def _claude_missing_present(section: str) -> bool:
    lower = re.sub(r"\s+", " ", section.lower())
    return (
        "explicit_claude_opus_present=false" in section
        or "final_panel_coverage_complete=false" in section
        or ("explicit claude opus" in lower and "missing" in lower)
    )


def _recursive_warning_clear_present(section: str) -> bool:
    lower = re.sub(r"\s+", " ", section.lower())
    return (
        "recursive warning regressions `0`" in section
        or "recursive warning-prefix growth" in lower
        or "recursive warning growth" in lower
        or "compact warning lists" in lower
    )


def _doc_observations(section: str, *, latest_count: int) -> dict[str, bool]:
    return {
        "final_submission_ready_false": "final_submission_ready=false" in section,
        "failed_claude_attempt_count_current": _contains_count(section, latest_count),
        "explicit_claude_opus_missing": _claude_missing_present(section),
        "promax_metadata_blocked": _promax_blocked_present(section),
        "manual_submission_blocked": _manual_blocked_present(section),
        "recursive_warning_regression_clear_or_fix_recorded": _recursive_warning_clear_present(section),
    }


def audit_final_blocker_doc_status(
    *,
    root: str | Path = ".",
    consistency_audit_json: str | Path | None = None,
    stamp: str = DEFAULT_STAMP,
    docs: list[str | Path] | None = None,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    if consistency_audit_json is None:
        consistency_audit_json = _consistency_audit_json_for_stamp(stamp)
    consistency_path = (repo / consistency_audit_json).resolve()
    consistency, failures = _read_json(consistency_path)
    summary = consistency.get("summary") or {}
    latest_count = int(summary.get("review_failed_claude_attempt_count") or 0)
    expected = {
        "final_submission_ready": consistency.get("final_submission_ready"),
        "failed_claude_attempts": latest_count,
        "explicit_claude_opus_present": summary.get("explicit_claude_opus_present"),
        "final_panel_coverage_complete": summary.get("final_panel_coverage_complete"),
        "promax_public_metadata_ready": summary.get("promax_public_metadata_ready"),
        "manual_confirmation_needed": summary.get("manual_confirmation_needed"),
        "manual_submission_system_ready": summary.get("manual_submission_system_ready"),
        "recursive_warning_regression_count": summary.get("recursive_warning_regression_count"),
    }
    if consistency and consistency.get("final_blocker_consistency_ok") is not True:
        failures.append("consistency_audit_not_ok")
    if latest_count <= 0:
        failures.append("missing_latest_failed_claude_attempt_count")

    doc_paths = [Path(item) for item in (docs or DEFAULT_DOCS)]
    doc_results: dict[str, Any] = {}
    for rel_path in doc_paths:
        path = (repo / rel_path).resolve()
        rel = _repo_relative_key(path, repo)
        text, doc_failures = _read_text(path)
        bounds = CURRENT_SECTION_BOUNDS.get(rel)
        if bounds is None:
            doc_failures.append(f"missing_current_section_bounds:{rel}")
            section = ""
            start_line = 0
            end_line = 0
        else:
            section, start_line, end_line, section_failures = _current_section(text, bounds[0], bounds[1])
            doc_failures.extend(section_failures)

        observations = _doc_observations(section, latest_count=latest_count)
        stale_hits = _stale_count_hits(section, latest_count=latest_count, start_line=start_line)
        stale_taxonomy_hits = _stale_blocker_taxonomy_hits(section, start_line=start_line)
        missing_observations = [key for key, present in observations.items() if not present]
        doc_failures.extend(f"missing_current_observation:{key}" for key in missing_observations)
        doc_failures.extend(
            f"stale_current_failed_claude_count:line={hit['line']}" for hit in stale_hits
        )
        doc_failures.extend(
            f"stale_blocker_taxonomy:line={hit['line']}" for hit in stale_taxonomy_hits
        )
        doc_results[rel] = {
            "path": _path_state(path, repo),
            "current_section": {
                "start_marker": bounds[0] if bounds else "",
                "end_marker": bounds[1] if bounds else "",
                "start_line": start_line,
                "end_line": end_line,
            },
            "observations": observations,
            "stale_count_hits": stale_hits,
            "stale_blocker_taxonomy_hits": stale_taxonomy_hits,
            "failures": _dedupe(doc_failures),
            "ok": not doc_failures,
        }
        failures.extend(f"{rel}:{failure}" for failure in doc_failures)

    return {
        "schema_version": "2026-06-13.final_blocker_doc_status_audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "local_read_only_final_blocker_doc_status_audit",
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "ok": not failures,
        "final_blocker_doc_status_ok": not failures,
        "final_submission_ready": False,
        "input_paths": {
            "consistency_audit": _path_state(consistency_path, repo),
            "docs": {rel: item["path"] for rel, item in doc_results.items()},
        },
        "expected_current_truth": expected,
        "doc_results": doc_results,
        "failures": _dedupe(failures),
        "next_actions": [
            "Rerun this doc-status audit after any canonical doc, final blocker, review, ProMax, or manual-request refresh.",
            "Treat stale current failed-Claude counts or final-ready wording as blockers before handoff.",
            "Keep final_submission_ready=false until ProMax public metadata, private manual confirmation, and explicit Claude Opus review coverage all close.",
        ],
    }


def render_markdown(audit: dict[str, Any]) -> str:
    expected = audit.get("expected_current_truth") or {}
    lines = [
        "# Final Blocker Doc Status Audit",
        "",
        f"- Created UTC: `{audit['created_at_utc']}`",
        f"- OK: `{str(audit['ok']).lower()}`",
        f"- Final blocker doc status OK: `{str(audit['final_blocker_doc_status_ok']).lower()}`",
        f"- Final submission ready: `{str(audit['final_submission_ready']).lower()}`",
        f"- Expected failed Claude attempts: `{expected.get('failed_claude_attempts')}`",
        f"- Expected explicit Claude Opus present: `{str(expected.get('explicit_claude_opus_present')).lower()}`",
        f"- Expected ProMax public metadata ready: `{str(expected.get('promax_public_metadata_ready')).lower()}`",
        f"- Expected manual submission system ready: `{str(expected.get('manual_submission_system_ready')).lower()}`",
        f"- Expected recursive warning regressions: `{expected.get('recursive_warning_regression_count')}`",
        "",
        "## Doc Results",
        "",
    ]
    for rel, result in (audit.get("doc_results") or {}).items():
        section = result.get("current_section") or {}
        lines.extend(
            [
                f"### `{rel}`",
                f"- OK: `{str(result.get('ok')).lower()}`",
                f"- Current section lines: `{section.get('start_line')}-{section.get('end_line')}`",
            ]
        )
        for key, value in (result.get("observations") or {}).items():
            lines.append(f"- {key}: `{str(value).lower()}`")
        stale_hits = result.get("stale_count_hits") or []
        if stale_hits:
            lines.append("- Stale current failed-Claude count hits:")
            lines.extend(f"  - line `{hit['line']}`: `{hit['text']}`" for hit in stale_hits)
        stale_taxonomy_hits = result.get("stale_blocker_taxonomy_hits") or []
        if stale_taxonomy_hits:
            lines.append("- Stale blocker taxonomy hits:")
            lines.extend(f"  - line `{hit['line']}`: `{hit['text']}`" for hit in stale_taxonomy_hits)
        failures = result.get("failures") or []
        if failures:
            lines.append("- Failures:")
            lines.extend(f"  - `{failure}`" for failure in failures)
        lines.append("")
    lines.extend(["## Failures", ""])
    failures = audit.get("failures") or []
    lines.extend(f"- `{item}`" for item in failures) if failures else lines.append("- None")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in audit.get("next_actions", []))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--stamp", default=DEFAULT_STAMP)
    parser.add_argument("--consistency-audit-json")
    parser.add_argument("--doc", action="append", default=[], help="Canonical status doc to audit.")
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    args = parser.parse_args()

    out_dir = Path(args.root) / DEFAULT_OUTPUT_DIR
    output_json = (
        Path(args.output_json)
        if args.output_json
        else out_dir / f"final_blocker_doc_status_audit_{args.stamp}.json"
    )
    output_md = (
        Path(args.output_md)
        if args.output_md
        else out_dir / f"final_blocker_doc_status_audit_{args.stamp}.md"
    )
    audit = audit_final_blocker_doc_status(
        root=args.root,
        consistency_audit_json=args.consistency_audit_json,
        stamp=args.stamp,
        docs=list(args.doc) if args.doc else None,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(audit), encoding="utf-8")
    print(json.dumps({"ok": audit["ok"], "output_json": str(output_json), "output_md": str(output_md)}, indent=2))
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
