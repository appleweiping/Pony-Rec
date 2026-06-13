from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SUBMISSION_PACKAGE_AUDIT = Path(
    "outputs/summary/paper_critical/submission_package_audit_20260612.json"
)
DEFAULT_SUBMISSION_METADATA_PACKET = Path(
    "outputs/summary/paper_critical/submission_metadata_packet_20260612.json"
)
DEFAULT_SUBMISSION_SOURCE_PACKAGE_REBUILD = Path(
    "outputs/summary/paper_critical/submission_source_package_rebuild_20260612.json"
)
DEFAULT_EXTERNAL_METADATA_AUDIT = Path(
    "outputs/summary/paper_critical/external_proceedings_metadata_recheck_20260612.json"
)
DEFAULT_MANUAL_CHECKLIST = Path(
    "outputs/summary/paper_critical/manual_submission_checklist_20260612.json"
)
DEFAULT_REVIEW_CONTINUATION_PACKET = Path(
    "outputs/summary/paper_critical/review_continuation_packet_20260613.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _path_state(path: Path, root: Path) -> dict[str, Any]:
    try:
        rel = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        rel = str(path)
    return {
        "path": rel,
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


WARNING_PREFIXES = (
    "pre_submission_gate_refresh:",
    "pre_submission_refresh_freshness:",
    "submission_release_candidate:",
    "final_submission_gate:",
    "review_continuation:",
    "submission_package:",
    "submission_source_package:",
    "submission_source_package_rebuild:",
    "submission_metadata_packet:",
    "manual_submission_checklist:",
    "external_proceedings_metadata:",
)


def _normalize_warning(value: Any) -> str:
    text = str(value).strip()
    changed = True
    while changed:
        changed = False
        for prefix in WARNING_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix) :]
                changed = True
    return text


def _gate_record(
    *,
    gate_id: str,
    path: Path,
    root: Path,
    payload: dict[str, Any],
    ready_field: str,
) -> dict[str, Any]:
    failures = list(payload.get("failures") or [])
    warnings = list(payload.get("warnings") or [])
    blockers = list(payload.get("remaining_blockers") or [])
    ready = payload.get(ready_field) is True
    ok = payload.get("ok") is True
    final_ready = payload.get("final_submission_ready") is True
    return {
        "gate_id": gate_id,
        "path": _path_state(path, root),
        "ok": ok,
        "ready_field": ready_field,
        "ready": ready,
        "final_submission_ready": final_ready,
        "schema_version": payload.get("schema_version"),
        "created_at_utc": payload.get("created_at_utc"),
        "failures": failures,
        "warnings": warnings,
        "remaining_blockers": blockers,
    }


def build_final_submission_gate(
    *,
    root: str | Path = ".",
    submission_package_audit_json: str | Path = DEFAULT_SUBMISSION_PACKAGE_AUDIT,
    submission_metadata_packet_json: str | Path = DEFAULT_SUBMISSION_METADATA_PACKET,
    submission_source_package_rebuild_json: str | Path = DEFAULT_SUBMISSION_SOURCE_PACKAGE_REBUILD,
    external_metadata_audit_json: str | Path = DEFAULT_EXTERNAL_METADATA_AUDIT,
    manual_checklist_json: str | Path = DEFAULT_MANUAL_CHECKLIST,
    review_continuation_packet_json: str | Path = DEFAULT_REVIEW_CONTINUATION_PACKET,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    package_path = repo / submission_package_audit_json
    metadata_path = repo / submission_metadata_packet_json
    source_rebuild_path = repo / submission_source_package_rebuild_json
    external_path = repo / external_metadata_audit_json
    manual_path = repo / manual_checklist_json
    review_path = repo / review_continuation_packet_json

    package = _read_json(package_path)
    metadata = _read_json(metadata_path)
    source_rebuild = _read_json(source_rebuild_path)
    external = _read_json(external_path)
    manual = _read_json(manual_path)
    review = _read_json(review_path)

    gates = [
        _gate_record(
            gate_id="submission_package",
            path=package_path,
            root=repo,
            payload=package,
            ready_field="submission_package_ready_for_target_formatting",
        ),
        _gate_record(
            gate_id="submission_metadata_packet",
            path=metadata_path,
            root=repo,
            payload=metadata,
            ready_field="submission_metadata_packet_ready",
        ),
        _gate_record(
            gate_id="submission_source_package_rebuild",
            path=source_rebuild_path,
            root=repo,
            payload=source_rebuild,
            ready_field="submission_source_package_rebuild_ready",
        ),
        _gate_record(
            gate_id="external_proceedings_metadata",
            path=external_path,
            root=repo,
            payload=external,
            ready_field="external_proceedings_metadata_ready",
        ),
        _gate_record(
            gate_id="manual_submission_checklist",
            path=manual_path,
            root=repo,
            payload=manual,
            ready_field="manual_submission_system_ready",
        ),
        _gate_record(
            gate_id="review_continuation",
            path=review_path,
            root=repo,
            payload=review,
            ready_field="review_continuation_ready",
        ),
    ]

    failures: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []
    for gate in gates:
        if not gate["ok"]:
            failures.append(f"{gate['gate_id']}:not_ok")
        if gate["gate_id"] in {
            "submission_package",
            "submission_metadata_packet",
            "submission_source_package_rebuild",
        } and not gate["ready"]:
            failures.append(f"{gate['gate_id']}:not_ready")
        if gate["gate_id"] == "external_proceedings_metadata" and not gate["ready"]:
            blockers.append("external_proceedings_metadata_not_ready")
        if gate["gate_id"] == "manual_submission_checklist" and not gate["ready"]:
            blockers.append("manual_submission_system_not_ready")
        if gate["gate_id"] == "review_continuation":
            if not gate["ready"]:
                failures.append("review_continuation:not_ready")
            if review.get("final_panel_coverage_complete") is not True:
                blockers.append("review_panel_coverage_not_complete")
        if gate["final_submission_ready"]:
            failures.append(f"{gate['gate_id']}:unexpected_local_final_submission_ready")
        warnings.extend(
            f"{gate['gate_id']}:{normalized}"
            for warning in gate["warnings"]
            if (normalized := _normalize_warning(warning))
        )
        blockers.extend(gate["remaining_blockers"])
        failures.extend(f"{gate['gate_id']}:{failure}" for failure in gate["failures"])

    all_local_artifact_gates_ok = (
        package.get("ok") is True
        and package.get("submission_package_ready_for_target_formatting") is True
        and metadata.get("ok") is True
        and metadata.get("submission_metadata_packet_ready") is True
        and source_rebuild.get("ok") is True
        and source_rebuild.get("submission_source_package_rebuild_ready") is True
        and manual.get("ok") is True
        and manual.get("manual_submission_checklist_ready") is True
        and external.get("ok") is True
        and review.get("ok") is True
        and review.get("review_continuation_ready") is True
    )
    external_ready = external.get("external_proceedings_metadata_ready") is True
    manual_ready = manual.get("manual_submission_system_ready") is True
    review_continuation_ready = review.get("review_continuation_ready") is True
    review_panel_coverage_complete = review.get("final_panel_coverage_complete") is True
    final_submission_ready = (
        all_local_artifact_gates_ok
        and external_ready
        and manual_ready
        and review_panel_coverage_complete
        and not failures
        and not blockers
    )

    verdict = (
        "FINAL_SUBMISSION_READY"
        if final_submission_ready
        else "LOCAL_PACKAGE_READY_BUT_EXTERNAL_MANUAL_OR_REVIEW_BLOCKED"
        if all_local_artifact_gates_ok and not failures
        else "FINAL_SUBMISSION_GATE_NEEDS_REPAIR"
    )

    return {
        "schema_version": "2026-06-12.final_submission_gate.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "local_read_only_final_submission_gate",
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "ok": not failures,
        "all_local_artifact_gates_ok": all_local_artifact_gates_ok,
        "external_proceedings_metadata_ready": external_ready,
        "manual_submission_system_ready": manual_ready,
        "review_continuation_ready": review_continuation_ready,
        "review_panel_coverage_complete": review_panel_coverage_complete,
        "final_submission_ready": final_submission_ready,
        "verdict": verdict,
        "input_paths": {
            "submission_package_audit": str(Path(submission_package_audit_json)),
            "submission_metadata_packet": str(Path(submission_metadata_packet_json)),
            "submission_source_package_rebuild": str(Path(submission_source_package_rebuild_json)),
            "external_metadata_audit": str(Path(external_metadata_audit_json)),
            "manual_checklist": str(Path(manual_checklist_json)),
            "review_continuation_packet": str(Path(review_continuation_packet_json)),
        },
        "gates": gates,
        "failures": _dedupe(failures),
        "warnings": _dedupe(warnings),
        "remaining_blockers": _dedupe(blockers),
        "next_actions": [
            "Rerun external proceedings metadata recheck immediately before final submission.",
            "Rerun submission package, source-package staging/rebuild, metadata packet, manual checklist, and this final gate after any paper/source/BibTeX change.",
            "Attach a substantive explicit Claude Opus review and rerun the review-continuation packet before claiming final review-panel coverage.",
            "Complete private author/COI/reviewer/declaration fields only inside the submission system.",
            "Keep final_submission_ready=false until external proceedings metadata, manual submission-system, and final review-panel coverage gates are all ready.",
        ],
    }


def _write_md(path: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# Final Submission Gate",
        "",
        f"Generated: {gate['created_at_utc']}",
        "",
        f"- Verdict: `{gate['verdict']}`",
        f"- OK: `{str(gate['ok']).lower()}`",
        f"- Local artifact gates OK: `{str(gate['all_local_artifact_gates_ok']).lower()}`",
        "- External proceedings metadata ready: "
        f"`{str(gate['external_proceedings_metadata_ready']).lower()}`",
        f"- Manual submission system ready: `{str(gate['manual_submission_system_ready']).lower()}`",
        f"- Review continuation ready: `{str(gate['review_continuation_ready']).lower()}`",
        f"- Review panel coverage complete: `{str(gate['review_panel_coverage_complete']).lower()}`",
        f"- Final submission ready: `{str(gate['final_submission_ready']).lower()}`",
        "",
        "## Gate Summary",
        "",
    ]
    for item in gate["gates"]:
        lines.extend(
            [
                f"- `{item['gate_id']}`: ok=`{str(item['ok']).lower()}`, "
                f"ready=`{str(item['ready']).lower()}`, "
                f"final_ready=`{str(item['final_submission_ready']).lower()}`, "
                f"path=`{item['path']['path']}`",
            ]
        )
    lines.extend(["", "## Remaining Blockers", ""])
    blockers = gate.get("remaining_blockers") or []
    lines.extend(f"- {blocker}" for blocker in blockers) if blockers else lines.append("- None")
    lines.extend(["", "## Failures", ""])
    failures = gate.get("failures") or []
    lines.extend(f"- `{failure}`" for failure in failures) if failures else lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    warnings = gate.get("warnings") or []
    lines.extend(f"- `{warning}`" for warning in warnings) if warnings else lines.append("- None")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {action}" for action in gate.get("next_actions", []))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--submission-package-audit-json", default=str(DEFAULT_SUBMISSION_PACKAGE_AUDIT))
    parser.add_argument("--submission-metadata-packet-json", default=str(DEFAULT_SUBMISSION_METADATA_PACKET))
    parser.add_argument(
        "--submission-source-package-rebuild-json",
        default=str(DEFAULT_SUBMISSION_SOURCE_PACKAGE_REBUILD),
    )
    parser.add_argument("--external-metadata-audit-json", default=str(DEFAULT_EXTERNAL_METADATA_AUDIT))
    parser.add_argument("--manual-checklist-json", default=str(DEFAULT_MANUAL_CHECKLIST))
    parser.add_argument("--review-continuation-packet-json", default=str(DEFAULT_REVIEW_CONTINUATION_PACKET))
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gate = build_final_submission_gate(
        root=args.root,
        submission_package_audit_json=args.submission_package_audit_json,
        submission_metadata_packet_json=args.submission_metadata_packet_json,
        submission_source_package_rebuild_json=args.submission_source_package_rebuild_json,
        external_metadata_audit_json=args.external_metadata_audit_json,
        manual_checklist_json=args.manual_checklist_json,
        review_continuation_packet_json=args.review_continuation_packet_json,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        _write_md(Path(args.output_md), gate)
    if not args.output_json:
        print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if gate["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
