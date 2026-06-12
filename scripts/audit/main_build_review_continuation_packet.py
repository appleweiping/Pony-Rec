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
DEFAULT_PANEL_REVIEW_JSON = Path(
    "outputs/summary/paper_critical/final_full_manuscript_panel_review_20260612.json"
)
DEFAULT_CLAIM_AUDIT_JSON = Path(
    "outputs/summary/paper_critical/final_paper_claim_audit_after_full_panel_review_20260612.json"
)
DEFAULT_SUBMISSION_PACKAGE_AUDIT_JSON = Path(
    "outputs/summary/paper_critical/submission_package_audit_20260613.json"
)
DEFAULT_RELEASE_CANDIDATE_STACK_JSON = Path(
    "outputs/summary/paper_critical/submission_release_candidate_stack_refresh_20260613.json"
)
DEFAULT_CLOSURE_PACKET_JSON = Path(
    "outputs/summary/paper_critical/final_submission_blocker_closure_packet_20260613.json"
)
DEFAULT_PROMAX_PROBE_JSON = Path(
    "outputs/summary/paper_critical/promax_public_metadata_probe_20260613.json"
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


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _parse_score_10(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*10", text)
    if match:
        return float(match.group(1))
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    return float(match.group(1)) if match else None


def _reviewer_names(panel_review: dict[str, Any], additional_reviews: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for review in panel_review.get("panel_reviews") or []:
        if isinstance(review, dict):
            names.append(str(review.get("reviewer") or ""))
    for review in additional_reviews:
        names.append(str(review.get("reviewer") or ""))
    return [name for name in names if name]


def _score_values(panel_review: dict[str, Any], additional_reviews: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    consensus = panel_review.get("reviewer_consensus") or {}
    floor = _parse_score_10(consensus.get("score_floor"))
    if floor is not None:
        values.append(floor)
    for review in panel_review.get("panel_reviews") or []:
        if isinstance(review, dict):
            score = _parse_score_10(review.get("score_10") or review.get("score_0_to_10"))
            if score is not None:
                values.append(score)
    for review in additional_reviews:
        score = _parse_score_10(review.get("score_0_to_10") or review.get("score_10"))
        if score is not None:
            values.append(score)
    return values


def _contains_any(value: str, needles: list[str]) -> bool:
    lower = value.lower()
    return any(needle in lower for needle in needles)


def _reviewer_coverage(
    panel_review: dict[str, Any], additional_reviews: list[dict[str, Any]]
) -> dict[str, Any]:
    names = _reviewer_names(panel_review, additional_reviews)
    score_values = _score_values(panel_review, additional_reviews)
    score_floor = min(score_values) if score_values else None
    explicit_claude = any(_contains_any(name, ["claude", "opus"]) for name in names)
    explicit_gpt55 = any(_contains_any(name, ["gpt-5.5", "gpt5.5", "meitner"]) for name in names)
    panel_count = len(panel_review.get("panel_reviews") or [])
    missing: list[str] = []
    if not explicit_claude:
        missing.append("explicit_claude_opus_review")
    if not explicit_gpt55:
        missing.append("explicit_gpt55_or_recorded_equivalent_review")
    return {
        "reviewer_names": names,
        "panel_review_count": panel_count,
        "additional_review_count": len(additional_reviews),
        "score_values_0_to_10": score_values,
        "score_floor_0_to_10": score_floor,
        "score_floor_meets_8": score_floor is not None and score_floor >= 8.0,
        "explicit_claude_opus_present": explicit_claude,
        "explicit_gpt55_present": explicit_gpt55,
        "final_panel_coverage_complete": not missing,
        "missing_perspectives": missing,
    }


def _load_additional_reviews(
    paths: list[str | Path], *, root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    reviews: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    failures: list[str] = []
    for raw_path in paths:
        path = (root / raw_path).resolve()
        states.append(_path_state(path, root))
        payload, errors = _read_json(path)
        failures.extend(errors)
        if payload:
            reviews.append(payload)
    return reviews, states, failures


def build_review_continuation_packet(
    *,
    root: str | Path = ".",
    panel_review_json: str | Path = DEFAULT_PANEL_REVIEW_JSON,
    claim_audit_json: str | Path = DEFAULT_CLAIM_AUDIT_JSON,
    submission_package_audit_json: str | Path = DEFAULT_SUBMISSION_PACKAGE_AUDIT_JSON,
    release_candidate_stack_json: str | Path = DEFAULT_RELEASE_CANDIDATE_STACK_JSON,
    closure_packet_json: str | Path = DEFAULT_CLOSURE_PACKET_JSON,
    promax_probe_json: str | Path = DEFAULT_PROMAX_PROBE_JSON,
    additional_review_jsons: list[str | Path] | None = None,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    paths = {
        "panel_review": (repo / panel_review_json).resolve(),
        "claim_audit": (repo / claim_audit_json).resolve(),
        "submission_package_audit": (repo / submission_package_audit_json).resolve(),
        "release_candidate_stack": (repo / release_candidate_stack_json).resolve(),
        "closure_packet": (repo / closure_packet_json).resolve(),
        "promax_probe": (repo / promax_probe_json).resolve(),
    }
    loaded: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for key, path in paths.items():
        payload, errors = _read_json(path)
        loaded[key] = payload
        failures.extend([f"{key}:{error}" for error in errors])

    additional_reviews, additional_states, additional_failures = _load_additional_reviews(
        list(additional_review_jsons or []),
        root=repo,
    )
    failures.extend([f"additional_review:{failure}" for failure in additional_failures])

    panel = loaded["panel_review"]
    claim = loaded["claim_audit"]
    package = loaded["submission_package_audit"]
    stack = loaded["release_candidate_stack"]
    closure = loaded["closure_packet"]
    promax = loaded["promax_probe"]

    coverage = _reviewer_coverage(panel, additional_reviews)
    package_gates = package.get("evidence_gates") or {}
    classified = closure.get("classified_remaining_blockers") or {}
    other_blockers = list(classified.get("other") or [])

    panel_ok = coverage["score_floor_meets_8"] and (
        (panel.get("reviewer_consensus") or {}).get("claim_boundary_ok") is True
    )
    claim_ok = (
        claim.get("ok") is True
        and claim.get("paper_evidence_ready_for_drafting") is True
        and claim.get("final_submission_ready") is False
    )
    package_ok = (
        package.get("ok") is True
        and package_gates.get("claim_audit_ok") is True
        and package_gates.get("panel_review_ok") is True
    )
    stack_ok = (
        stack.get("ok") is True
        and stack.get("local_release_candidate_ready") is True
        and stack.get("final_submission_ready") is False
    )
    closure_ok = (
        closure.get("ok") is True
        and closure.get("closure_packet_ready") is True
        and closure.get("ready_for_human_handoff") is True
        and closure.get("final_submission_ready") is False
        and not other_blockers
    )
    promax_ok = (
        promax.get("ok") is True
        and promax.get("promax_public_metadata_ready") is False
        and promax.get("final_submission_ready") is False
    )

    if not panel_ok:
        failures.append("panel_review_score_floor_or_claim_boundary_not_ok")
    if not claim_ok:
        failures.append("claim_audit_not_ready_for_drafting_or_unexpected_final_state")
    if not package_ok:
        failures.append("submission_package_audit_not_ok")
    if not stack_ok:
        failures.append("release_candidate_stack_not_local_ready")
    if not closure_ok:
        failures.append("closure_packet_not_ready_or_has_other_blockers")
    if not promax_ok:
        failures.append("promax_probe_not_in_expected_blocked_state")

    final_submission_ready = any(
        bool(payload.get("final_submission_ready")) for payload in loaded.values()
    )
    if final_submission_ready:
        failures.append("unexpected_input_claims_final_submission_ready")

    blockers = _dedupe(
        list(claim.get("remaining_blockers") or [])
        + list(package.get("remaining_blockers") or [])
        + list(stack.get("remaining_blockers") or [])
        + list(closure.get("remaining_blockers") or [])
        + list(promax.get("remaining_blockers") or [])
        + coverage["missing_perspectives"]
    )
    ok = not failures
    return {
        "schema_version": "2026-06-13.review_continuation_packet.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "local_read_only_review_continuation_packet",
        "project": "uncertainty",
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "aris_skills": ["aris-auto-review-loop", "aris-paper-claim-audit", "aris-citation-audit"],
        "ok": ok,
        "review_continuation_ready": ok,
        "final_panel_coverage_complete": coverage["final_panel_coverage_complete"],
        "local_release_candidate_ready": stack.get("local_release_candidate_ready") is True,
        "ready_for_human_handoff": closure.get("ready_for_human_handoff") is True,
        "final_submission_ready": False,
        "final_submission_ready_policy": (
            "Always false here; this packet is a review-continuation handoff and cannot close "
            "external proceedings metadata or private submission-system confirmations."
        ),
        "verdict": (
            "REVIEW_CONTINUATION_READY_FINAL_BLOCKED"
            if ok
            else "REVIEW_CONTINUATION_NEEDS_ATTENTION"
        ),
        "input_paths": {key: _path_state(path, repo) for key, path in paths.items()},
        "additional_review_paths": additional_states,
        "reviewer_coverage": coverage,
        "gate_summary": {
            "panel_ok": panel_ok,
            "claim_audit_ok": claim_ok,
            "submission_package_audit_ok": package_ok,
            "release_candidate_stack_ok": stack_ok,
            "closure_packet_ok": closure_ok,
            "promax_probe_expected_blocked": promax_ok,
            "closure_other_blockers": other_blockers,
            "external_proceedings_metadata_ready": closure.get("external_proceedings_metadata_ready")
            is True,
            "manual_submission_system_ready": closure.get("manual_submission_system_ready") is True,
            "promax_public_metadata_ready": promax.get("promax_public_metadata_ready") is True,
        },
        "classified_remaining_blockers": classified,
        "remaining_blockers": blockers,
        "failures": _dedupe(failures),
        "warnings": _dedupe(
            list(panel.get("warnings") or [])
            + list(claim.get("warnings") or [])
            + list(package.get("warnings") or [])
            + list(stack.get("warnings") or [])
            + list(closure.get("warnings") or [])
            + list(promax.get("warnings") or [])
        ),
        "next_actions": [
            "Attach explicit Claude Opus and fresh GPT-5.5 reviewer outputs as additional review JSONs if they complete.",
            "Keep final_submission_ready=false until ProMax page-range/Crossref/DOI and private manual checklist close.",
            "After any manuscript, bibliography, package, or metadata change, rerun the release-candidate stack and this packet.",
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_md(path: Path, packet: dict[str, Any]) -> None:
    coverage = packet.get("reviewer_coverage") or {}
    gate = packet.get("gate_summary") or {}
    lines = [
        "# Review Continuation Packet",
        "",
        f"Generated: {packet['created_at_utc']}",
        "",
        f"- OK: `{str(packet['ok']).lower()}`",
        f"- Verdict: `{packet['verdict']}`",
        f"- Review continuation ready: `{str(packet['review_continuation_ready']).lower()}`",
        f"- Final panel coverage complete: `{str(packet['final_panel_coverage_complete']).lower()}`",
        f"- Local release candidate ready: `{str(packet['local_release_candidate_ready']).lower()}`",
        f"- Final submission ready: `{str(packet['final_submission_ready']).lower()}`",
        "",
        "## Reviewer Coverage",
        "",
        f"- Score floor: `{coverage.get('score_floor_0_to_10')}`",
        f"- Score floor meets 8: `{str(coverage.get('score_floor_meets_8')).lower()}`",
        f"- Explicit GPT-5.5 present: `{str(coverage.get('explicit_gpt55_present')).lower()}`",
        f"- Explicit Claude Opus present: `{str(coverage.get('explicit_claude_opus_present')).lower()}`",
        f"- Missing perspectives: `{', '.join(coverage.get('missing_perspectives') or []) or 'none'}`",
        "",
        "## Gate Summary",
        "",
    ]
    for key in [
        "panel_ok",
        "claim_audit_ok",
        "submission_package_audit_ok",
        "release_candidate_stack_ok",
        "closure_packet_ok",
        "promax_probe_expected_blocked",
        "external_proceedings_metadata_ready",
        "manual_submission_system_ready",
        "promax_public_metadata_ready",
    ]:
        lines.append(f"- {key}: `{str(gate.get(key)).lower()}`")
    lines.extend(["", "## Remaining Blockers", ""])
    blockers = packet.get("remaining_blockers") or []
    if blockers:
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("- none")
    lines.extend(["", "## Failures", ""])
    failures = packet.get("failures") or []
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- none")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {action}" for action in packet.get("next_actions") or [])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stamp", default=DEFAULT_STAMP)
    parser.add_argument("--panel-review-json", default=str(DEFAULT_PANEL_REVIEW_JSON))
    parser.add_argument("--claim-audit-json", default=str(DEFAULT_CLAIM_AUDIT_JSON))
    parser.add_argument(
        "--submission-package-audit-json",
        default=str(DEFAULT_SUBMISSION_PACKAGE_AUDIT_JSON),
    )
    parser.add_argument(
        "--release-candidate-stack-json",
        default=str(DEFAULT_RELEASE_CANDIDATE_STACK_JSON),
    )
    parser.add_argument("--closure-packet-json", default=str(DEFAULT_CLOSURE_PACKET_JSON))
    parser.add_argument("--promax-probe-json", default=str(DEFAULT_PROMAX_PROBE_JSON))
    parser.add_argument(
        "--additional-review-json",
        action="append",
        default=[],
        help="Optional structured reviewer JSON to include in coverage.",
    )
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.output_dir
    output_json = Path(args.output_json) if args.output_json else out_dir / f"review_continuation_packet_{args.stamp}.json"
    output_md = Path(args.output_md) if args.output_md else out_dir / f"review_continuation_packet_{args.stamp}.md"

    packet = build_review_continuation_packet(
        root=root,
        panel_review_json=args.panel_review_json,
        claim_audit_json=args.claim_audit_json,
        submission_package_audit_json=args.submission_package_audit_json,
        release_candidate_stack_json=args.release_candidate_stack_json,
        closure_packet_json=args.closure_packet_json,
        promax_probe_json=args.promax_probe_json,
        additional_review_jsons=args.additional_review_json,
    )
    _write_json(output_json, packet)
    _write_md(output_md, packet)
    return 0 if packet["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
