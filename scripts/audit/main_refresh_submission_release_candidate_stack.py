from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from scripts.audit.main_audit_pre_submission_refresh_freshness import (
    build_pre_submission_refresh_freshness_audit,
)
from scripts.audit.main_audit_pre_submission_refresh_freshness import _write_md as write_freshness_md
from scripts.audit.main_build_submission_release_candidate_packet import (
    build_submission_release_candidate_packet,
)
from scripts.audit.main_build_submission_release_candidate_packet import _write_md as write_candidate_md
from scripts.audit.main_refresh_pre_submission_gates import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REVIEW_CONTINUATION_PACKET,
    DEFAULT_STAMP,
    refresh_pre_submission_gates,
)
from scripts.audit.main_refresh_pre_submission_gates import _write_md as write_refresh_md


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _path_state(path: Path, root: Path) -> dict[str, Any]:
    try:
        display_path = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        display_path = str(path)
    return {
        "path": display_path,
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


def _step_record(
    *,
    step_id: str,
    json_path: Path,
    md_path: Path,
    root: Path,
    payload: dict[str, Any],
    ready_field: str,
) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "json": _path_state(json_path, root),
        "md": _path_state(md_path, root),
        "ok": payload.get("ok") is True,
        "ready_field": ready_field,
        "ready": payload.get(ready_field) is True,
        "final_submission_ready": payload.get("final_submission_ready") is True,
        "schema_version": payload.get("schema_version"),
        "created_at_utc": payload.get("created_at_utc"),
        "failures": list(payload.get("failures") or []),
        "warnings": list(payload.get("warnings") or []),
        "remaining_blockers": list(payload.get("remaining_blockers") or []),
    }


def refresh_submission_release_candidate_stack(
    *,
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    stamp: str = DEFAULT_STAMP,
    refresh_kwargs: dict[str, Any] | None = None,
    refresh_runner: Callable[..., dict[str, Any]] = refresh_pre_submission_gates,
    freshness_builder: Callable[..., dict[str, Any]] = build_pre_submission_refresh_freshness_audit,
    candidate_builder: Callable[..., dict[str, Any]] = build_submission_release_candidate_packet,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    out_dir = repo / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    refresh_json = out_dir / f"pre_submission_gate_refresh_{stamp}.json"
    refresh_md = out_dir / f"pre_submission_gate_refresh_{stamp}.md"
    freshness_json = out_dir / f"pre_submission_gate_refresh_freshness_{stamp}.json"
    freshness_md = out_dir / f"pre_submission_gate_refresh_freshness_{stamp}.md"
    candidate_json = out_dir / f"submission_release_candidate_{stamp}.json"
    candidate_md = out_dir / f"submission_release_candidate_{stamp}.md"
    final_gate_json = out_dir / f"final_submission_gate_{stamp}.json"
    source_package_json = out_dir / f"submission_source_package_{stamp}.json"
    source_rebuild_json = out_dir / f"submission_source_package_rebuild_{stamp}.json"
    metadata_packet_json = out_dir / f"submission_metadata_packet_{stamp}.json"
    manual_checklist_json = out_dir / f"manual_submission_checklist_{stamp}.json"
    external_metadata_json = out_dir / f"external_proceedings_metadata_recheck_{stamp}.json"

    refresh_args = dict(refresh_kwargs or {})
    refresh = refresh_runner(
        root=repo,
        output_dir=output_dir,
        stamp=stamp,
        **refresh_args,
    )
    _write_json(refresh_json, refresh)
    write_refresh_md(refresh_md, refresh)

    freshness = freshness_builder(
        root=repo,
        refresh_json=refresh_json.relative_to(repo),
    )
    _write_json(freshness_json, freshness)
    write_freshness_md(freshness_md, freshness)

    candidate = candidate_builder(
        root=repo,
        final_submission_gate_json=final_gate_json.relative_to(repo),
        refresh_freshness_json=freshness_json.relative_to(repo),
        submission_source_package_json=source_package_json.relative_to(repo),
        submission_source_package_rebuild_json=source_rebuild_json.relative_to(repo),
        submission_metadata_packet_json=metadata_packet_json.relative_to(repo),
        manual_checklist_json=manual_checklist_json.relative_to(repo),
        external_metadata_audit_json=external_metadata_json.relative_to(repo),
    )
    _write_json(candidate_json, candidate)
    write_candidate_md(candidate_md, candidate)

    steps = [
        _step_record(
            step_id="pre_submission_gate_refresh",
            json_path=refresh_json,
            md_path=refresh_md,
            root=repo,
            payload=refresh,
            ready_field="ok",
        ),
        _step_record(
            step_id="pre_submission_refresh_freshness",
            json_path=freshness_json,
            md_path=freshness_md,
            root=repo,
            payload=freshness,
            ready_field="refresh_artifact_fresh",
        ),
        _step_record(
            step_id="submission_release_candidate",
            json_path=candidate_json,
            md_path=candidate_md,
            root=repo,
            payload=candidate,
            ready_field="local_release_candidate_ready",
        ),
    ]
    failures = _dedupe([f"{step['step_id']}:{item}" for step in steps for item in step["failures"]])
    warnings = _dedupe([f"{step['step_id']}:{item}" for step in steps for item in step["warnings"]])
    blockers = _dedupe([str(item) for step in steps for item in step["remaining_blockers"]])
    ok = all(step["ok"] for step in steps) and not failures

    return {
        "schema_version": "2026-06-12.submission_release_candidate_stack_refresh.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "local_artifact_submission_release_candidate_stack_refresh",
        "read_only": False,
        "will_ssh": False,
        "will_copy": True,
        "will_delete": True,
        "will_start_experiment": False,
        "stamp": stamp,
        "output_dir": str(Path(output_dir)),
        "ok": ok,
        "local_release_candidate_ready": candidate.get("local_release_candidate_ready") is True,
        "readiness_scope": candidate.get("readiness_scope") or "local_artifacts_only",
        "blocking_status": candidate.get("blocking_status") or "",
        "final_submission_ready": candidate.get("final_submission_ready") is True,
        "final_submission_ready_source": candidate.get("final_submission_ready_source") or "",
        "refresh_ok": refresh.get("ok") is True,
        "freshness_ok": freshness.get("ok") is True,
        "refresh_artifact_fresh": freshness.get("refresh_artifact_fresh") is True,
        "release_candidate_ok": candidate.get("ok") is True,
        "steps": steps,
        "failures": failures,
        "warnings": warnings,
        "remaining_blockers": blockers,
        "next_actions": [
            "Use this stack refresh as the preferred one-command local pre-submission handoff.",
            "Keep final_submission_ready=false until the final submission gate reports true.",
            "Resolve ProMax public page-range/DOI metadata and private manual submission-system confirmation before final submission.",
        ],
    }


def _write_md(path: Path, stack: dict[str, Any]) -> None:
    lines = [
        "# Submission Release-Candidate Stack Refresh",
        "",
        f"Generated: {stack['created_at_utc']}",
        "",
        f"- OK: `{str(stack['ok']).lower()}`",
        f"- Local release candidate ready: `{str(stack['local_release_candidate_ready']).lower()}`",
        f"- Readiness scope: `{stack['readiness_scope']}`",
        f"- Blocking status: `{stack['blocking_status']}`",
        f"- Final submission ready: `{str(stack['final_submission_ready']).lower()}`",
        f"- Stamp: `{stack['stamp']}`",
        "",
        "## Steps",
        "",
    ]
    for step in stack.get("steps", []):
        lines.append(
            f"- `{step['step_id']}`: ok=`{str(step['ok']).lower()}`, "
            f"ready=`{str(step['ready']).lower()}`, json=`{step['json']['path']}`"
        )
    lines.extend(["", "## Remaining Blockers", ""])
    blockers = stack.get("remaining_blockers") or []
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- None")
    lines.extend(["", "## Failures", ""])
    failures = stack.get("failures") or []
    lines.extend(f"- `{item}`" for item in failures) if failures else lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    warnings = stack.get("warnings") or []
    lines.extend(f"- `{item}`" for item in warnings) if warnings else lines.append("- None")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in stack.get("next_actions", []))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stamp", default=DEFAULT_STAMP)
    parser.add_argument("--external-network-mode", choices=["live", "disabled", "fixture"], default="live")
    parser.add_argument("--external-fixture-json")
    parser.add_argument("--external-timeout-seconds", type=int, default=20)
    parser.add_argument("--manual-private-confirmation-json")
    parser.add_argument("--review-continuation-packet-json", default=str(DEFAULT_REVIEW_CONTINUATION_PACKET))
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    refresh_kwargs: dict[str, Any] = {
        "external_network_mode": args.external_network_mode,
        "external_fixture_json": args.external_fixture_json,
        "external_timeout_seconds": args.external_timeout_seconds,
        "manual_private_confirmation_json": args.manual_private_confirmation_json,
        "review_continuation_packet_json": args.review_continuation_packet_json,
    }
    stack = refresh_submission_release_candidate_stack(
        root=args.root,
        output_dir=args.output_dir,
        stamp=args.stamp,
        refresh_kwargs=refresh_kwargs,
    )
    if args.output_json:
        _write_json(Path(args.output_json), stack)
    if args.output_md:
        _write_md(Path(args.output_md), stack)
    if not args.output_json:
        print(json.dumps(stack, indent=2, sort_keys=True))
    return 0 if stack["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
