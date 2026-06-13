from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.audit.main_build_claude_review_request_packet import (
    DEFAULT_REVIEW_CONTINUATION_PACKET_JSON,
    EXPECTED_CLAUDE_REVIEW_SCHEMA,
)
from scripts.audit.main_build_review_continuation_packet import (
    CLAUDE_REQUIRED_REMAINING_BLOCKER_ACKS,
    _is_explicit_claude_opus_name,
    _parse_score_10,
    _validate_additional_review,
)


DEFAULT_OUTPUT_DIR = Path("outputs/summary/paper_critical")
DEFAULT_STAMP = "20260613"
DEFAULT_REVIEW_JSON = Path("outputs/summary/paper_critical/claude_opus_review_20260613.json")
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


def _schema_key_state(review: dict[str, Any]) -> dict[str, list[str]]:
    expected = set(EXPECTED_CLAUDE_REVIEW_SCHEMA)
    actual = set(review)
    return {
        "expected_keys": sorted(expected),
        "missing_keys": sorted(expected - actual),
        "extra_keys": sorted(actual - expected),
    }


def build_claude_opus_review_validation_packet(
    *,
    root: str | Path = ".",
    review_json: str | Path = DEFAULT_REVIEW_JSON,
    review_request_packet_json: str | Path = DEFAULT_REVIEW_REQUEST_PACKET_JSON,
    review_continuation_packet_json: str | Path = DEFAULT_REVIEW_CONTINUATION_PACKET_JSON,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    paths = {
        "review_json": (repo / review_json).resolve(),
        "review_request_packet": (repo / review_request_packet_json).resolve(),
        "review_continuation_packet": (repo / review_continuation_packet_json).resolve(),
    }

    review, review_errors = _read_json(paths["review_json"])
    request_packet, request_errors = _read_json(paths["review_request_packet"])
    continuation_packet, continuation_errors = _read_json(paths["review_continuation_packet"])

    validation_ok, validation_failures = (
        _validate_additional_review(
            review,
            required_blocker_ack_groups=CLAUDE_REQUIRED_REMAINING_BLOCKER_ACKS,
        )
        if review
        else (False, [])
    )
    reviewer = str(review.get("reviewer") or "")
    score = _parse_score_10(review.get("score_0_to_10") or review.get("score_10")) if review else None
    explicit_claude_opus = _is_explicit_claude_opus_name(reviewer)
    score_floor_meets_8 = score is not None and score >= 8.0
    schema_state = _schema_key_state(review) if review else _schema_key_state({})

    failures: list[str] = []
    failures.extend([f"review_json:{error}" for error in review_errors])
    failures.extend(validation_failures)
    if review and not explicit_claude_opus:
        failures.append("reviewer_not_explicit_claude_opus")
    if review and not score_floor_meets_8:
        failures.append("score_floor_below_8")
    if request_packet and request_packet.get("claude_review_needed") is not True:
        failures.append("request_packet_does_not_need_claude_review")
    if continuation_packet and continuation_packet.get("final_submission_ready") is not False:
        failures.append("unexpected_review_continuation_final_submission_ready")

    warnings = [f"review_request_packet:{error}" for error in request_errors] + [
        f"review_continuation_packet:{error}" for error in continuation_errors
    ]

    return {
        "schema_version": "2026-06-13.claude_opus_review_validation.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "local_read_only_claude_opus_review_validation",
        "project": "uncertainty",
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "input_paths": {key: _path_state(path, repo) for key, path in paths.items()},
        "reviewer": reviewer,
        "score_0_to_10": score,
        "score_floor_meets_8": score_floor_meets_8,
        "explicit_claude_opus_reviewer": explicit_claude_opus,
        "valid_review_evidence": validation_ok,
        "schema_key_state": schema_state,
        "ready_to_attach_for_review_gate": not failures,
        "review_continuation_command_after_valid_review": request_packet.get(
            "review_continuation_command_after_valid_review"
        ),
        "notes": [
            "This validator does not create reviewer coverage; it only checks a returned Claude Opus JSON before attaching it.",
            "A substantive Claude Opus review below 8/10 should still be preserved as evidence, but it cannot close the at-least-8 review gate.",
            "Final submission readiness remains controlled by the final submission gate.",
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_md(path: Path, packet: dict[str, Any]) -> None:
    lines = [
        "# Claude Opus Review JSON Validation",
        "",
        f"- Created UTC: `{packet['created_at_utc']}`",
        f"- OK: `{str(packet['ok']).lower()}`",
        f"- Reviewer: `{packet.get('reviewer')}`",
        f"- Score: `{packet.get('score_0_to_10')}`",
        f"- Score floor meets 8: `{str(packet.get('score_floor_meets_8')).lower()}`",
        f"- Explicit Claude Opus reviewer: `{str(packet.get('explicit_claude_opus_reviewer')).lower()}`",
        f"- Valid review evidence: `{str(packet.get('valid_review_evidence')).lower()}`",
        f"- Ready to attach for review gate: `{str(packet.get('ready_to_attach_for_review_gate')).lower()}`",
        "",
        "## Schema Keys",
        "",
        f"- Missing keys: `{', '.join(packet['schema_key_state']['missing_keys']) or 'none'}`",
        f"- Extra keys: `{', '.join(packet['schema_key_state']['extra_keys']) or 'none'}`",
        "",
        "## Failures",
        "",
    ]
    failures = packet.get("failures") or []
    lines.extend(f"- {failure}" for failure in failures) if failures else lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    warnings = packet.get("warnings") or []
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- none")
    command = packet.get("review_continuation_command_after_valid_review")
    if command:
        lines.extend(["", "## Follow-Up Command", "", "```bash", str(command), "```"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--stamp", default=DEFAULT_STAMP)
    parser.add_argument("--review-json", default=str(DEFAULT_REVIEW_JSON))
    parser.add_argument("--review-request-packet-json", default=str(DEFAULT_REVIEW_REQUEST_PACKET_JSON))
    parser.add_argument("--review-continuation-packet-json", default=str(DEFAULT_REVIEW_CONTINUATION_PACKET_JSON))
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / DEFAULT_OUTPUT_DIR
    output_json = (
        Path(args.output_json)
        if args.output_json
        else out_dir / f"claude_opus_review_validation_{args.stamp}.json"
    )
    output_md = (
        Path(args.output_md)
        if args.output_md
        else out_dir / f"claude_opus_review_validation_{args.stamp}.md"
    )
    packet = build_claude_opus_review_validation_packet(
        root=root,
        review_json=args.review_json,
        review_request_packet_json=args.review_request_packet_json,
        review_continuation_packet_json=args.review_continuation_packet_json,
    )
    _write_json(output_json, packet)
    _write_md(output_md, packet)
    print(json.dumps({"ok": packet["ok"], "output_json": str(output_json), "output_md": str(output_md)}, indent=2))
    return 0 if packet["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
