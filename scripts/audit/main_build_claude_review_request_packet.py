from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs/summary/paper_critical")
DEFAULT_STAMP = "20260613"
DEFAULT_REVIEW_CONTINUATION_PACKET_JSON = Path(
    "outputs/summary/paper_critical/review_continuation_packet_20260613.json"
)
DEFAULT_PANEL_REVIEW_JSON = Path(
    "outputs/summary/paper_critical/final_full_manuscript_panel_review_20260612.json"
)
DEFAULT_CLAIM_AUDIT_JSON = Path(
    "outputs/summary/paper_critical/final_paper_claim_audit_after_full_panel_review_20260612.json"
)
DEFAULT_GPT55_REVIEW_JSON = Path(
    "outputs/summary/paper_critical/gpt55_section_review_after_handoff_20260613.json"
)


EXPECTED_CLAUDE_REVIEW_SCHEMA: dict[str, str] = {
    "reviewer": "Must include claude and opus, e.g. claude-opus.",
    "created_at_utc": "ISO-8601 UTC timestamp.",
    "source": "Tool, thread, or manual channel that produced the review.",
    "score_0_to_10": "Numeric top-conference review score.",
    "verdict": "ACCEPT, WEAK_ACCEPT, CONDITIONAL_PASS, BORDERLINE, WEAK_REJECT, or REJECT.",
    "claim_boundary_ok": "Boolean; true only if scoped same-candidate claim is respected.",
    "final_submission_ready_claim_allowed": "Boolean; must stay false while final gates are open.",
    "kill_argument": "The strongest remaining rejection argument.",
    "major_concerns": "Array of concrete concerns.",
    "required_changes": "Array of concrete changes before submission/final-ready claim.",
    "remaining_blockers_acknowledged": "Array naming external/manual blockers the review did not waive.",
    "valid_review_evidence": "Boolean; true only for a complete substantive review.",
}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _as_list(values: Any) -> list[Any]:
    return values if isinstance(values, list) else []


def _claim_counts(claim_audit: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in _as_list(claim_audit.get("claims")):
        if isinstance(row, dict):
            status = str(row.get("status") or "UNKNOWN")
            counts[status] = counts.get(status, 0) + 1
    return counts


def _claim_rows_for_prompt(claim_audit: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in _as_list(claim_audit.get("claims")):
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "")
        if status in {"CONTRADICTED", "UNSUPPORTED"}:
            rows.append(
                {
                    "id": str(row.get("id") or ""),
                    "status": status,
                    "allowed_wording": str(row.get("allowed_wording") or ""),
                    "forbidden_wording": str(row.get("forbidden_wording") or ""),
                }
            )
    return rows


def _failed_attempt_summary(review_packet: dict[str, Any]) -> dict[str, Any]:
    attempts = [attempt for attempt in _as_list(review_packet.get("failed_review_attempts")) if isinstance(attempt, dict)]
    errors = sorted({str(attempt.get("error") or "") for attempt in attempts if attempt.get("error")})
    sources = [str(attempt.get("source") or "") for attempt in attempts if attempt.get("source")]
    return {
        "count": len(attempts),
        "unique_errors": errors,
        "sources": sources,
    }


def _failed_attempt_command_args(review_packet: dict[str, Any]) -> str:
    paths: list[str] = []
    for state in _as_list(review_packet.get("failed_review_attempt_paths")):
        if isinstance(state, dict):
            path = str(state.get("path") or "").strip()
            if path:
                paths.append(path.replace("\\", "/"))
    seen: set[str] = set()
    deduped = [path for path in paths if not (path in seen or seen.add(path))]
    return " ".join(f"--failed-review-attempt-json {path}" for path in deduped)


def _build_prompt(
    *,
    review_packet: dict[str, Any],
    panel_review: dict[str, Any],
    claim_audit: dict[str, Any],
    gpt55_review: dict[str, Any],
) -> str:
    coverage = review_packet.get("reviewer_coverage") or {}
    gate_summary = review_packet.get("gate_summary") or {}
    classified = review_packet.get("classified_remaining_blockers") or {}
    failed = _failed_attempt_summary(review_packet)
    panel_consensus = panel_review.get("reviewer_consensus") or {}
    gpt55_blockers = _as_list(gpt55_review.get("critical_blockers"))
    claim_rows = _claim_rows_for_prompt(claim_audit)
    claim_counts = _claim_counts(claim_audit)

    prompt_payload = {
        "review_role": "Claude Opus independent hostile top-conference reviewer",
        "claim_scope": (
            "Task-grounded calibrated uncertainty improves controlled same-candidate "
            "candidate ranking/reranking reliability. Do not expand this into a "
            "full-catalog recommender SOTA claim."
        ),
        "evidence_summary": {
            "official_same_candidate_rows": "32 official baseline rows + 4 C-CRP rows on sports/toys/home/tools",
            "protocol": "10k users per new domain, 101 same candidates, Qwen3-8B backbone, full @5/@10/@20 + MRR metrics",
            "comparison_result": "C-CRP rank 1 on all 7 metrics with 56/56 positive Holm-significant paired tests per domain",
            "phase_2_5_limits": [
                "observation module is motivation-only, not causal or SOTA evidence",
                "component ablation is supplementary diagnostic-only; do not claim every component is necessary",
                "hyperparameter analysis is supplementary stability/sensitivity evidence",
            ],
        },
        "panel_state": {
            "existing_panel_score_floor": coverage.get("score_floor_0_to_10"),
            "existing_reviewers": coverage.get("reviewer_names"),
            "panel_consensus": panel_consensus,
            "explicit_claude_opus_present": coverage.get("explicit_claude_opus_present"),
            "missing_perspectives": coverage.get("missing_perspectives"),
            "failed_claude_attempts": failed,
            "gpt55_verdict": gpt55_review.get("verdict"),
            "gpt55_score_0_to_10": gpt55_review.get("score_0_to_10"),
            "gpt55_critical_blockers": gpt55_blockers,
        },
        "claim_audit_state": {
            "ok": claim_audit.get("ok"),
            "paper_evidence_ready_for_drafting": claim_audit.get("paper_evidence_ready_for_drafting"),
            "final_submission_ready": claim_audit.get("final_submission_ready"),
            "claim_status_counts": claim_counts,
            "non_supported_or_contradicted_claims": claim_rows,
        },
        "current_gates": {
            "review_continuation_ready": review_packet.get("review_continuation_ready"),
            "local_release_candidate_ready": review_packet.get("local_release_candidate_ready"),
            "final_submission_ready": review_packet.get("final_submission_ready"),
            "gate_summary": gate_summary,
            "classified_remaining_blockers": classified,
        },
        "review_instructions": [
            "Return a single JSON object matching the requested schema; do not include markdown.",
            "Be strict and identify the strongest remaining kill argument.",
            "Do not waive ProMax public proceedings metadata blockers.",
            "Do not waive private manual submission-system blockers.",
            "Set final_submission_ready_claim_allowed=false unless all final gates are genuinely closed.",
            "A score below 8.0 is allowed if the evidence does not satisfy top-conference standards.",
        ],
        "required_output_schema": EXPECTED_CLAUDE_REVIEW_SCHEMA,
    }
    return json.dumps(prompt_payload, indent=2, ensure_ascii=False)


def build_claude_review_request_packet(
    *,
    root: str | Path = ".",
    review_continuation_packet_json: str | Path = DEFAULT_REVIEW_CONTINUATION_PACKET_JSON,
    panel_review_json: str | Path = DEFAULT_PANEL_REVIEW_JSON,
    claim_audit_json: str | Path = DEFAULT_CLAIM_AUDIT_JSON,
    gpt55_review_json: str | Path = DEFAULT_GPT55_REVIEW_JSON,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    paths = {
        "review_continuation_packet": (repo / review_continuation_packet_json).resolve(),
        "panel_review": (repo / panel_review_json).resolve(),
        "claim_audit": (repo / claim_audit_json).resolve(),
        "gpt55_review": (repo / gpt55_review_json).resolve(),
    }
    loaded: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for key, path in paths.items():
        payload, errors = _read_json(path)
        loaded[key] = payload
        failures.extend([f"{key}:{error}" for error in errors])

    review_packet = loaded["review_continuation_packet"]
    coverage = review_packet.get("reviewer_coverage") or {}
    missing = _as_list(coverage.get("missing_perspectives"))
    claude_needed = "explicit_claude_opus_review" in [str(item) for item in missing]
    prompt = _build_prompt(
        review_packet=review_packet,
        panel_review=loaded["panel_review"],
        claim_audit=loaded["claim_audit"],
        gpt55_review=loaded["gpt55_review"],
    )

    if review_packet and review_packet.get("review_continuation_ready") is not True:
        failures.append("review_continuation_packet_not_ready")
    if review_packet and review_packet.get("final_submission_ready") is not False:
        failures.append("unexpected_review_packet_final_submission_ready_state")
    if review_packet and not claude_needed and coverage.get("explicit_claude_opus_present") is not True:
        failures.append("claude_gap_state_inconsistent")

    return {
        "schema_version": "2026-06-13.claude_review_request_packet.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "local_read_only_claude_review_request_packet",
        "project": "uncertainty",
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "ok": not failures,
        "failures": failures,
        "input_paths": {key: _path_state(path, repo) for key, path in paths.items()},
        "claude_review_needed": claude_needed,
        "existing_score_floor_0_to_10": coverage.get("score_floor_0_to_10"),
        "missing_perspectives": missing,
        "failed_claude_attempt_summary": _failed_attempt_summary(review_packet),
        "expected_additional_review_json": {
            "recommended_path": "outputs/summary/paper_critical/claude_opus_review_20260613.json",
            "schema": EXPECTED_CLAUDE_REVIEW_SCHEMA,
            "must_count_as_coverage": [
                "reviewer contains both claude and opus",
                "valid_review_evidence is true",
                "score_0_to_10 is present",
                "score_0_to_10 is at least 8.0 for the review gate",
                "claim_boundary_ok is true",
                "final_submission_ready_claim_allowed is false while current blockers remain",
            ],
        },
        "claude_review_prompt": prompt,
        "claude_review_prompt_sha256": _sha256_text(prompt),
        "validation_command_before_attach": (
            "python -m scripts.audit.main_validate_claude_opus_review_json "
            "--review-json outputs/summary/paper_critical/claude_opus_review_20260613.json "
            "--output-json outputs/summary/paper_critical/claude_opus_review_validation_20260613.json "
            "--output-md outputs/summary/paper_critical/claude_opus_review_validation_20260613.md"
        ),
        "review_continuation_command_after_valid_review": " ".join(
            part
            for part in [
                "python -m scripts.audit.main_build_review_continuation_packet",
                "--additional-review-json outputs/summary/paper_critical/claude_opus_review_20260613.json",
                _failed_attempt_command_args(review_packet),
            ]
            if part
        ),
        "notes": [
            "This packet is a request artifact only; it is not valid Claude review coverage.",
            "Do not lower final blockers because this request packet exists.",
            "If the Claude connector fails, store the failure separately with valid_review_evidence=false.",
        ],
    }


def render_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# Claude Opus Review Request Packet",
        "",
        f"- Created UTC: `{packet['created_at_utc']}`",
        f"- OK: `{str(packet['ok']).lower()}`",
        f"- Claude review needed: `{str(packet['claude_review_needed']).lower()}`",
        f"- Existing score floor: `{packet.get('existing_score_floor_0_to_10')}`",
        f"- Failed Claude attempts: `{packet['failed_claude_attempt_summary']['count']}`",
        f"- Prompt sha256: `{packet['claude_review_prompt_sha256']}`",
        "",
        "## Missing Perspectives",
        "",
    ]
    missing = packet.get("missing_perspectives") or []
    if missing:
        lines.extend([f"- `{item}`" for item in missing])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Expected Review JSON",
            "",
            f"- Recommended path: `{packet['expected_additional_review_json']['recommended_path']}`",
            "",
            "```json",
            json.dumps(packet["expected_additional_review_json"]["schema"], indent=2),
            "```",
            "",
            "## Prompt",
            "",
            "```json",
            packet["claude_review_prompt"],
            "```",
            "",
            "## Validation Command Before Attach",
            "",
            "```bash",
            packet["validation_command_before_attach"],
            "```",
            "",
            "## Follow-Up Command",
            "",
            "```bash",
            packet["review_continuation_command_after_valid_review"],
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--stamp", default=DEFAULT_STAMP)
    parser.add_argument("--review-continuation-packet-json", default=str(DEFAULT_REVIEW_CONTINUATION_PACKET_JSON))
    parser.add_argument("--panel-review-json", default=str(DEFAULT_PANEL_REVIEW_JSON))
    parser.add_argument("--claim-audit-json", default=str(DEFAULT_CLAIM_AUDIT_JSON))
    parser.add_argument("--gpt55-review-json", default=str(DEFAULT_GPT55_REVIEW_JSON))
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    args = parser.parse_args()

    out_dir = Path(args.root) / DEFAULT_OUTPUT_DIR
    output_json = Path(args.output_json) if args.output_json else out_dir / f"claude_opus_review_request_packet_{args.stamp}.json"
    output_md = Path(args.output_md) if args.output_md else out_dir / f"claude_opus_review_request_packet_{args.stamp}.md"

    packet = build_claude_review_request_packet(
        root=args.root,
        review_continuation_packet_json=args.review_continuation_packet_json,
        panel_review_json=args.panel_review_json,
        claim_audit_json=args.claim_audit_json,
        gpt55_review_json=args.gpt55_review_json,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(packet), encoding="utf-8")
    print(json.dumps({"ok": packet["ok"], "output_json": str(output_json), "output_md": str(output_md)}, indent=2))
    return 0 if packet["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
