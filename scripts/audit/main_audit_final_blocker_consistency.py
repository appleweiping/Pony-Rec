from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs/summary/paper_critical")
DEFAULT_STAMP = "20260613"
DEFAULT_FINAL_GATE = DEFAULT_OUTPUT_DIR / "final_submission_gate_20260613.json"
DEFAULT_STACK = DEFAULT_OUTPUT_DIR / "submission_release_candidate_stack_refresh_20260613.json"
DEFAULT_CLOSURE = DEFAULT_OUTPUT_DIR / "final_submission_blocker_closure_packet_20260613.json"
DEFAULT_REVIEW = DEFAULT_OUTPUT_DIR / "review_continuation_packet_20260613.json"
DEFAULT_CLAUDE_REQUEST = DEFAULT_OUTPUT_DIR / "claude_opus_review_request_packet_20260613.json"
DEFAULT_PROMAX_PROBE = DEFAULT_OUTPUT_DIR / "promax_public_metadata_probe_20260613.json"
DEFAULT_MANUAL_REQUEST = (
    DEFAULT_OUTPUT_DIR / "manual_submission_private_confirmation_request_packet_20260613.json"
)

RECURSIVE_WARNING_MARKERS = [
    "pre_submission_gate_refresh:final_submission_gate:review_continuation:",
    "review_continuation:pre_submission_gate_refresh:final_submission_gate:",
    "submission_release_candidate:final_submission_gate:review_continuation:",
]


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


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value) if str(item).strip()]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _warning_regressions(**payloads: dict[str, Any]) -> list[dict[str, str]]:
    regressions: list[dict[str, str]] = []
    for name, payload in payloads.items():
        for warning in _string_list(payload.get("warnings")):
            if any(marker in warning for marker in RECURSIVE_WARNING_MARKERS):
                regressions.append({"artifact": name, "warning": warning})
    return regressions


def _closure_group(closure: dict[str, Any], group_id: str) -> dict[str, Any]:
    for group in _as_list(closure.get("closure_groups")):
        if isinstance(group, dict) and group.get("group_id") == group_id:
            return group
    return {}


def audit_final_blocker_consistency(
    *,
    root: str | Path = ".",
    final_gate_json: str | Path = DEFAULT_FINAL_GATE,
    stack_json: str | Path = DEFAULT_STACK,
    closure_json: str | Path = DEFAULT_CLOSURE,
    review_json: str | Path = DEFAULT_REVIEW,
    claude_request_json: str | Path = DEFAULT_CLAUDE_REQUEST,
    promax_probe_json: str | Path = DEFAULT_PROMAX_PROBE,
    manual_request_json: str | Path = DEFAULT_MANUAL_REQUEST,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    paths = {
        "final_gate": (repo / final_gate_json).resolve(),
        "release_stack": (repo / stack_json).resolve(),
        "closure_packet": (repo / closure_json).resolve(),
        "review_continuation": (repo / review_json).resolve(),
        "claude_request": (repo / claude_request_json).resolve(),
        "promax_probe": (repo / promax_probe_json).resolve(),
        "manual_request": (repo / manual_request_json).resolve(),
    }
    payloads: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for key, path in paths.items():
        payload, errors = _read_json(path)
        payloads[key] = payload
        failures.extend(f"{key}:{error}" for error in errors)

    final_gate = payloads["final_gate"]
    stack = payloads["release_stack"]
    closure = payloads["closure_packet"]
    review = payloads["review_continuation"]
    claude_request = payloads["claude_request"]
    promax = payloads["promax_probe"]
    manual = payloads["manual_request"]

    for key, payload in payloads.items():
        if payload and payload.get("ok") is not True:
            failures.append(f"{key}:not_ok")

    final_ready_flags = {
        key: payload.get("final_submission_ready")
        for key, payload in payloads.items()
        if "final_submission_ready" in payload
    }
    true_final_flags = [key for key, value in final_ready_flags.items() if value is True]
    if true_final_flags:
        failures.append("unexpected_final_submission_ready_true:" + ",".join(true_final_flags))

    stack_ready = stack.get("local_release_candidate_ready") is True
    closure_handoff = closure.get("ready_for_human_handoff") is True
    if stack and not stack_ready:
        failures.append("release_stack_not_local_ready")
    if closure and not closure_handoff:
        failures.append("closure_packet_not_handoff_ready")
    if stack.get("blocking_status") != "external_manual_or_review_blocked":
        failures.append(f"release_stack_blocking_status_unexpected:{stack.get('blocking_status')}")
    if final_gate.get("review_panel_coverage_complete") is not False:
        failures.append("final_gate_review_panel_coverage_not_false")

    coverage = review.get("reviewer_coverage") or {}
    failed_attempts = _as_list(review.get("failed_review_attempts"))
    claude_summary = claude_request.get("failed_claude_attempt_summary") or {}
    failed_count = len(failed_attempts)
    if claude_summary.get("count") != failed_count:
        failures.append(f"claude_failed_attempt_count_mismatch:{claude_summary.get('count')} != {failed_count}")
    if coverage.get("explicit_claude_opus_present") is not False:
        failures.append("review_coverage_claude_unexpectedly_present")
    if coverage.get("final_panel_coverage_complete") is not False:
        failures.append("review_coverage_unexpectedly_complete")
    missing = _string_list(coverage.get("missing_perspectives"))
    if "explicit_claude_opus_review" not in missing:
        failures.append("review_coverage_missing_explicit_claude_gap")
    if claude_request.get("claude_review_needed") is not True:
        failures.append("claude_request_not_needed_despite_missing_claude")
    expected_review_ack_groups = ["manual_submission_system", "promax_public_metadata"]
    request_review_spec = claude_request.get("expected_additional_review_json") or {}
    response_template = request_review_spec.get("response_template") or {}
    must_count_rules = _string_list(request_review_spec.get("must_count_as_coverage"))
    required_ack_groups = _string_list(review.get("required_claude_blocker_ack_groups"))
    if not response_template:
        failures.append("claude_request_missing_response_template")
    elif response_template.get("valid_review_evidence") is not False:
        failures.append("claude_request_template_valid_review_evidence_not_false")
    if not request_review_spec.get("response_template_sha256"):
        failures.append("claude_request_missing_response_template_sha256")
    for group in expected_review_ack_groups:
        if group not in required_ack_groups:
            failures.append(f"review_missing_required_claude_ack_group:{group}")
    for expected_rule in [
        "remaining_blockers_acknowledged names the ProMax public metadata blocker",
        "remaining_blockers_acknowledged names the private manual submission-system blocker",
    ]:
        if expected_rule not in must_count_rules:
            failures.append(f"claude_request_missing_must_count_rule:{expected_rule}")
    response_ack_text = " ".join(_string_list(response_template.get("remaining_blockers_acknowledged"))).lower()
    if response_template and "promax" not in response_ack_text:
        failures.append("claude_request_template_missing_promax_ack")
    if response_template and "manual" not in response_ack_text:
        failures.append("claude_request_template_missing_manual_ack")

    promax_blockers = _string_list(promax.get("remaining_blockers"))
    if promax.get("promax_public_metadata_ready") is not False:
        failures.append("promax_probe_not_blocked")
    for blocker in [
        "promax:final_page_range_missing_in_bib",
        "promax:crossref_registry_not_visible",
        "promax:doi_resolver_not_visible",
    ]:
        if blocker not in promax_blockers:
            failures.append(f"promax_probe_missing_blocker:{blocker}")
    direct = promax.get("direct_checks") or {}
    promax_direct_status = {
        "crossref": ((direct.get("crossref") or {}).get("status_code")),
        "doi_resolver": ((direct.get("doi_resolver") or {}).get("status_code")),
        "acm_dl": ((direct.get("acm_dl") or {}).get("status_code")),
    }
    closure_inputs = closure.get("input_paths") or {}
    closure_probe_path = closure_inputs.get("promax_probe_json") or {}
    if closure_probe_path.get("exists") is not True:
        failures.append("closure_missing_promax_probe_input")
    closure_external_group = _closure_group(closure, "external_proceedings_metadata")
    if not closure_external_group:
        failures.append("closure_missing_external_metadata_group")
    latest_probe = closure_external_group.get("latest_public_probe") or {}
    if latest_probe.get("provided") is not True:
        failures.append("closure_missing_latest_public_promax_probe")
    closure_probe_status = {
        "crossref": latest_probe.get("crossref_status_code"),
        "doi_resolver": latest_probe.get("doi_resolver_status_code"),
        "acm_dl": latest_probe.get("acm_dl_status_code"),
    }
    if latest_probe.get("provided") is True and closure_probe_status != promax_direct_status:
        failures.append(
            "closure_promax_probe_status_mismatch:"
            + json.dumps(
                {"closure": closure_probe_status, "probe": promax_direct_status},
                sort_keys=True,
            )
        )
    closure_review_group = _closure_group(closure, "review_panel_coverage")
    closure_review_blockers = _string_list(closure_review_group.get("remaining_blockers"))
    for blocker in ["review_panel_coverage_not_complete", "explicit_claude_opus_review"]:
        if blocker not in closure_review_blockers:
            failures.append(f"closure_review_group_missing_blocker:{blocker}")

    if manual.get("request_packet_ready") is not True:
        failures.append("manual_request_packet_not_ready")
    if manual.get("manual_confirmation_needed") is not True:
        failures.append("manual_request_not_needed_state_unexpected")
    if manual.get("manual_submission_system_ready") is not False:
        failures.append("manual_request_system_ready_unexpected")
    manual_required = manual.get("required_private_confirmation") or {}
    if not manual_required.get("source_manifest_sha256"):
        failures.append("manual_request_missing_source_manifest_sha256")
    if not _string_list(manual_required.get("completed_item_ids_for_full_manual_gate")):
        failures.append("manual_request_missing_completed_item_ids")

    warning_regressions = _warning_regressions(**payloads)
    if warning_regressions:
        failures.append(f"recursive_warning_prefix_regressions:{len(warning_regressions)}")

    final_blockers = _string_list(final_gate.get("remaining_blockers"))
    required_final_blockers = [
        "external_proceedings_metadata_not_ready",
        "manual_submission_system_not_ready",
        "review_panel_coverage_not_complete",
        "explicit_claude_opus_review",
    ]
    for blocker in required_final_blockers:
        if blocker not in final_blockers:
            failures.append(f"final_gate_missing_blocker:{blocker}")

    return {
        "schema_version": "2026-06-13.final_blocker_consistency_audit.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "local_read_only_final_blocker_consistency_audit",
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "ok": not failures,
        "final_blocker_consistency_ok": not failures,
        "final_submission_ready": False,
        "input_paths": {key: _path_state(path, repo) for key, path in paths.items()},
        "summary": {
            "local_release_candidate_ready": stack_ready,
            "closure_ready_for_human_handoff": closure_handoff,
            "blocking_status": stack.get("blocking_status", ""),
            "final_gate_remaining_blocker_count": len(final_blockers),
            "review_failed_claude_attempt_count": failed_count,
            "claude_request_failed_attempt_count": claude_summary.get("count"),
            "claude_request_has_response_template": bool(response_template),
            "claude_request_template_valid_review_evidence": response_template.get("valid_review_evidence"),
            "review_required_claude_ack_groups": required_ack_groups,
            "explicit_claude_opus_present": coverage.get("explicit_claude_opus_present"),
            "final_panel_coverage_complete": coverage.get("final_panel_coverage_complete"),
            "promax_public_metadata_ready": promax.get("promax_public_metadata_ready"),
            "promax_direct_status": promax_direct_status,
            "closure_promax_probe_provided": latest_probe.get("provided") is True,
            "closure_promax_probe_status": closure_probe_status,
            "manual_confirmation_needed": manual.get("manual_confirmation_needed"),
            "manual_submission_system_ready": manual.get("manual_submission_system_ready"),
            "recursive_warning_regression_count": len(warning_regressions),
        },
        "required_open_blockers": {
            "final_gate": required_final_blockers,
            "promax_probe": promax_blockers,
            "review_missing_perspectives": missing,
        },
        "warning_regressions": warning_regressions,
        "failures": _dedupe(failures),
        "next_actions": [
            "Keep final_submission_ready=false until this audit, the final gate, ProMax metadata, manual confirmation, and explicit Claude Opus review coverage all close.",
            "Rerun this audit after any blocker packet, final gate, review-continuation packet, or release-candidate stack refresh.",
            "If this audit fails, repair the inconsistent packet before reporting final readiness.",
        ],
    }


def render_markdown(audit: dict[str, Any]) -> str:
    summary = audit.get("summary") or {}
    lines = [
        "# Final Blocker Consistency Audit",
        "",
        f"- Created UTC: `{audit['created_at_utc']}`",
        f"- OK: `{str(audit['ok']).lower()}`",
        f"- Final blocker consistency OK: `{str(audit['final_blocker_consistency_ok']).lower()}`",
        f"- Final submission ready: `{str(audit['final_submission_ready']).lower()}`",
        f"- Blocking status: `{summary.get('blocking_status', '')}`",
        f"- Local release candidate ready: `{str(summary.get('local_release_candidate_ready')).lower()}`",
        f"- Closure ready for human handoff: `{str(summary.get('closure_ready_for_human_handoff')).lower()}`",
        f"- Failed Claude attempts: `{summary.get('review_failed_claude_attempt_count')}`",
        f"- Claude request has response template: `{str(summary.get('claude_request_has_response_template')).lower()}`",
        f"- Claude template valid_review_evidence: `{summary.get('claude_request_template_valid_review_evidence')}`",
        f"- Claude required ack groups: `{', '.join(summary.get('review_required_claude_ack_groups') or []) or 'none'}`",
        f"- Explicit Claude Opus present: `{str(summary.get('explicit_claude_opus_present')).lower()}`",
        f"- ProMax public metadata ready: `{str(summary.get('promax_public_metadata_ready')).lower()}`",
        f"- Closure carries ProMax probe: `{str(summary.get('closure_promax_probe_provided')).lower()}`",
        f"- Manual confirmation needed: `{str(summary.get('manual_confirmation_needed')).lower()}`",
        f"- Recursive warning regressions: `{summary.get('recursive_warning_regression_count')}`",
        "",
        "## ProMax Direct Status",
        "",
    ]
    for key, value in (summary.get("promax_direct_status") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Required Open Blockers", ""])
    for group, values in (audit.get("required_open_blockers") or {}).items():
        lines.append(f"### {group}")
        if values:
            lines.extend(f"- `{item}`" for item in values)
        else:
            lines.append("- None")
        lines.append("")
    lines.extend(["## Warning Regressions", ""])
    regressions = audit.get("warning_regressions") or []
    if regressions:
        lines.extend(f"- `{item['artifact']}`: `{item['warning']}`" for item in regressions)
    else:
        lines.append("- None")
    lines.extend(["", "## Failures", ""])
    failures = audit.get("failures") or []
    lines.extend(f"- `{item}`" for item in failures) if failures else lines.append("- None")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in audit.get("next_actions", []))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--stamp", default=DEFAULT_STAMP)
    parser.add_argument("--final-gate-json", default=str(DEFAULT_FINAL_GATE))
    parser.add_argument("--stack-json", default=str(DEFAULT_STACK))
    parser.add_argument("--closure-json", default=str(DEFAULT_CLOSURE))
    parser.add_argument("--review-json", default=str(DEFAULT_REVIEW))
    parser.add_argument("--claude-request-json", default=str(DEFAULT_CLAUDE_REQUEST))
    parser.add_argument("--promax-probe-json", default=str(DEFAULT_PROMAX_PROBE))
    parser.add_argument("--manual-request-json", default=str(DEFAULT_MANUAL_REQUEST))
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    args = parser.parse_args()

    out_dir = Path(args.root) / DEFAULT_OUTPUT_DIR
    output_json = (
        Path(args.output_json)
        if args.output_json
        else out_dir / f"final_blocker_consistency_audit_{args.stamp}.json"
    )
    output_md = (
        Path(args.output_md)
        if args.output_md
        else out_dir / f"final_blocker_consistency_audit_{args.stamp}.md"
    )
    audit = audit_final_blocker_consistency(
        root=args.root,
        final_gate_json=args.final_gate_json,
        stack_json=args.stack_json,
        closure_json=args.closure_json,
        review_json=args.review_json,
        claude_request_json=args.claude_request_json,
        promax_probe_json=args.promax_probe_json,
        manual_request_json=args.manual_request_json,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(audit), encoding="utf-8")
    print(json.dumps({"ok": audit["ok"], "output_json": str(output_json), "output_md": str(output_md)}, indent=2))
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
