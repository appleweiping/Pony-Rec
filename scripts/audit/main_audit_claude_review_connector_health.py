from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs/summary/paper_critical")
DEFAULT_STAMP = "20260613"
DEFAULT_GLOB = "claude_opus_review_attempt*.json"
DEFAULT_REVIEW_REQUEST_PACKET_JSON = Path(
    "outputs/summary/paper_critical/claude_opus_review_request_packet_20260613.json"
)


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


def _attempt_sort_key(attempt: dict[str, Any]) -> tuple[str, str]:
    return (
        str(attempt.get("completed_at_utc") or attempt.get("created_at_utc") or ""),
        str(attempt.get("source") or ""),
    )


def _same_error_tail_streak(attempts: list[dict[str, Any]]) -> tuple[str, int]:
    if not attempts:
        return "", 0
    last_error = str(attempts[-1].get("error") or "")
    streak = 0
    for attempt in reversed(attempts):
        if str(attempt.get("error") or "") != last_error:
            break
        streak += 1
    return last_error, streak


def build_claude_review_connector_health_packet(
    *,
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    failed_attempt_jsons: list[str | Path] | None = None,
    failed_attempt_glob: str = DEFAULT_GLOB,
    review_request_packet_json: str | Path = DEFAULT_REVIEW_REQUEST_PACKET_JSON,
    repeat_error_threshold: int = 3,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    out_dir = repo / output_dir
    if failed_attempt_jsons is None:
        candidate_paths = sorted(out_dir.glob(failed_attempt_glob))
    else:
        candidate_paths = [(repo / raw_path).resolve() for raw_path in failed_attempt_jsons]

    attempts: list[dict[str, Any]] = []
    path_states: list[dict[str, Any]] = []
    failures: list[str] = []
    for path in candidate_paths:
        path_states.append(_path_state(path, repo))
        payload, errors = _read_json(path)
        failures.extend(errors)
        if payload:
            payload = dict(payload)
            payload["_path"] = _repo_relative(path, repo)
            attempts.append(payload)
    attempts.sort(key=_attempt_sort_key)

    request_path = (repo / review_request_packet_json).resolve()
    request_packet, request_errors = _read_json(request_path)
    request_warnings = [f"review_request_packet:{error}" for error in request_errors]

    valid_review_count = sum(1 for attempt in attempts if attempt.get("valid_review_evidence") is True)
    failed_attempts = [attempt for attempt in attempts if attempt.get("valid_review_evidence") is not True]
    errors = [str(attempt.get("error") or "") for attempt in failed_attempts if attempt.get("error")]
    error_counts = dict(sorted(Counter(errors).items()))
    last_error, same_error_tail_streak = _same_error_tail_streak(failed_attempts)
    failure_count_unhealthy = bool(last_error) and len(failed_attempts) >= repeat_error_threshold and valid_review_count == 0
    connector_unhealthy = bool(last_error) and (
        same_error_tail_streak >= repeat_error_threshold or failure_count_unhealthy
    )
    same_route_retry_recommended = not connector_unhealthy
    external_json_route_ready = request_packet.get("ok") is True and request_packet.get("claude_review_needed") is True

    warnings: list[str] = []
    warnings.extend(request_warnings)
    if bool(last_error) and same_error_tail_streak >= repeat_error_threshold:
        warnings.append(
            f"same_connector_error_repeated:{same_error_tail_streak}:"
            f"{last_error.replace(' ', '_')}"
        )
    elif failure_count_unhealthy:
        warnings.append(
            f"connector_failed_attempts_without_valid_review:{len(failed_attempts)}:"
            f"threshold={repeat_error_threshold}"
        )

    request_packet_rel = _repo_relative(request_path, repo)

    return {
        "schema_version": "2026-06-13.claude_review_connector_health.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "local_read_only_claude_review_connector_health",
        "project": "uncertainty",
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "ok": not failures,
        "final_submission_ready": False,
        "final_submission_ready_policy": (
            "This health packet cannot close final readiness; it only decides whether "
            "the same Claude connector route is worth retrying."
        ),
        "failures": failures,
        "warnings": warnings,
        "input_paths": {
            "review_request_packet": _path_state(request_path, repo),
            "failed_attempts": path_states,
        },
        "failed_attempt_count": len(failed_attempts),
        "valid_review_evidence_count": valid_review_count,
        "unique_error_count": len(error_counts),
        "error_counts": error_counts,
        "last_error": last_error,
        "same_error_tail_streak": same_error_tail_streak,
        "failure_count_unhealthy": failure_count_unhealthy,
        "repeat_error_threshold": repeat_error_threshold,
        "connector_unhealthy": connector_unhealthy,
        "same_route_retry_recommended": same_route_retry_recommended,
        "external_json_route_ready": external_json_route_ready,
        "review_request_packet_ok": request_packet.get("ok") is True,
        "review_request_packet_claude_needed": request_packet.get("claude_review_needed") is True,
        "recommended_next_route": (
            "external_claude_opus_json_via_request_packet_and_validator"
            if connector_unhealthy and external_json_route_ready
            else "retry_connector_or_refresh_request_packet"
        ),
        "next_actions": [
            "If connector_unhealthy=true, do not keep retrying the same mcp__claude_review route unless the connector/tooling changes.",
            f"Use {request_packet_rel} and its sibling Markdown packet to obtain a substantive external Claude Opus JSON.",
            "Run main_validate_claude_opus_review_json.py before attaching any returned Claude Opus JSON with --additional-review-json.",
            "Keep final_submission_ready=false until the final submission gate reports true.",
        ],
        "attempt_summaries": [
            {
                "path": attempt.get("_path"),
                "created_at_utc": attempt.get("created_at_utc"),
                "completed_at_utc": attempt.get("completed_at_utc"),
                "reviewer": attempt.get("reviewer"),
                "status": attempt.get("status"),
                "valid_review_evidence": attempt.get("valid_review_evidence") is True,
                "error": attempt.get("error"),
                "source": attempt.get("source"),
            }
            for attempt in attempts
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_md(path: Path, packet: dict[str, Any]) -> None:
    lines = [
        "# Claude Review Connector Health",
        "",
        f"- Created UTC: `{packet['created_at_utc']}`",
        f"- OK: `{str(packet['ok']).lower()}`",
        f"- Final submission ready: `{str(packet['final_submission_ready']).lower()}`",
        f"- Failed attempt count: `{packet['failed_attempt_count']}`",
        f"- Valid review evidence count: `{packet['valid_review_evidence_count']}`",
        f"- Last error: `{packet['last_error']}`",
        f"- Same-error tail streak: `{packet['same_error_tail_streak']}`",
        f"- Connector unhealthy: `{str(packet['connector_unhealthy']).lower()}`",
        f"- Same route retry recommended: `{str(packet['same_route_retry_recommended']).lower()}`",
        f"- Recommended next route: `{packet['recommended_next_route']}`",
        "",
        "## Error Counts",
        "",
    ]
    error_counts = packet.get("error_counts") or {}
    if error_counts:
        lines.extend(f"- `{error}`: `{count}`" for error, count in error_counts.items())
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    warnings = packet.get("warnings") or []
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- none")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {action}" for action in packet.get("next_actions") or [])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stamp", default=DEFAULT_STAMP)
    parser.add_argument("--failed-attempt-json", action="append", default=None)
    parser.add_argument("--failed-attempt-glob", default=DEFAULT_GLOB)
    parser.add_argument("--review-request-packet-json", default=str(DEFAULT_REVIEW_REQUEST_PACKET_JSON))
    parser.add_argument("--repeat-error-threshold", type=int, default=3)
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.output_dir
    output_json = (
        Path(args.output_json)
        if args.output_json
        else out_dir / f"claude_review_connector_health_{args.stamp}.json"
    )
    output_md = (
        Path(args.output_md)
        if args.output_md
        else out_dir / f"claude_review_connector_health_{args.stamp}.md"
    )
    packet = build_claude_review_connector_health_packet(
        root=root,
        output_dir=args.output_dir,
        failed_attempt_jsons=args.failed_attempt_json,
        failed_attempt_glob=args.failed_attempt_glob,
        review_request_packet_json=args.review_request_packet_json,
        repeat_error_threshold=args.repeat_error_threshold,
    )
    _write_json(output_json, packet)
    _write_md(output_md, packet)
    print(json.dumps({"ok": packet["ok"], "output_json": str(output_json), "output_md": str(output_md)}, indent=2))
    return 0 if packet["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
