from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs/summary/paper_critical")
DEFAULT_STAMP = "20260612"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _path_state(path: Path, root: Path) -> dict[str, Any]:
    try:
        display = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        display = str(path)
    return {
        "path": display,
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


def _classify_remaining_blockers(blockers: list[str]) -> dict[str, list[str]]:
    external: list[str] = []
    manual: list[str] = []
    other: list[str] = []
    for blocker in blockers:
        lowered = blocker.lower()
        if "promax" in lowered or "external_proceedings_metadata" in lowered:
            external.append(blocker)
        elif "manual" in lowered or "submission-system" in lowered:
            manual.append(blocker)
        else:
            other.append(blocker)
    return {
        "external_proceedings_metadata": _dedupe(external),
        "manual_submission_system": _dedupe(manual),
        "other": _dedupe(other),
    }


def _promax_evidence(external: dict[str, Any]) -> dict[str, Any]:
    promax = ((external.get("checked_entries") or {}).get("promax") or {})
    bibtex = promax.get("bibtex") or {}
    network = promax.get("network") or {}
    return {
        "external_metadata_ready": promax.get("external_metadata_ready") is True,
        "bibtex": {
            "doi": bibtex.get("doi", ""),
            "pages": bibtex.get("pages", ""),
            "numpages": bibtex.get("numpages", ""),
            "isbn": bibtex.get("isbn", ""),
            "location": bibtex.get("location", ""),
            "eprint": bibtex.get("eprint", ""),
            "booktitle": bibtex.get("booktitle", ""),
        },
        "crossref_status_code": ((network.get("crossref") or {}).get("status_code")),
        "doi_resolver_status_code": ((network.get("doi_resolver") or {}).get("status_code")),
        "source_checks": [
            {
                "name": item.get("name", ""),
                "ok": item.get("ok") is True,
                "status_code": item.get("status_code"),
                "missing_patterns": list(item.get("missing_patterns") or []),
            }
            for item in promax.get("source_checks") or []
        ],
        "blockers": list(promax.get("blockers") or []),
        "warnings": list(promax.get("warnings") or []),
    }


def _promax_probe_evidence(probe: dict[str, Any] | None) -> dict[str, Any]:
    if not probe:
        return {"provided": False}
    direct = probe.get("direct_checks") or {}
    return {
        "provided": True,
        "created_at_utc": probe.get("created_at_utc", ""),
        "promax_public_metadata_ready": probe.get("promax_public_metadata_ready") is True,
        "final_submission_ready": probe.get("final_submission_ready") is True,
        "crossref_status_code": ((direct.get("crossref") or {}).get("status_code")),
        "doi_resolver_status_code": ((direct.get("doi_resolver") or {}).get("status_code")),
        "acm_dl_status_code": ((direct.get("acm_dl") or {}).get("status_code")),
        "remaining_blockers": list(probe.get("remaining_blockers") or []),
        "warnings": list(probe.get("warnings") or []),
        "source_probes": [
            {
                "name": item.get("name", ""),
                "ok": item.get("ok") is True,
                "status_code": item.get("status_code"),
                "missing_patterns": list(item.get("missing_patterns") or []),
            }
            for item in probe.get("source_probes") or []
        ],
    }


def build_final_submission_blocker_closure_packet(
    *,
    root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    stamp: str = DEFAULT_STAMP,
    final_gate_json: str | Path | None = None,
    external_metadata_json: str | Path | None = None,
    manual_checklist_json: str | Path | None = None,
    release_candidate_stack_json: str | Path | None = None,
    promax_probe_json: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    out_dir = repo / output_dir
    final_path = repo / (
        final_gate_json or out_dir / f"final_submission_gate_{stamp}.json"
    )
    external_path = repo / (
        external_metadata_json or out_dir / f"external_proceedings_metadata_recheck_{stamp}.json"
    )
    manual_path = repo / (
        manual_checklist_json or out_dir / f"manual_submission_checklist_{stamp}.json"
    )
    stack_path = repo / (
        release_candidate_stack_json
        or out_dir / f"submission_release_candidate_stack_refresh_{stamp}.json"
    )
    probe_path = repo / promax_probe_json if promax_probe_json else None

    final_gate = _read_json(final_path)
    external = _read_json(external_path)
    manual = _read_json(manual_path)
    stack = _read_json(stack_path)
    probe = _read_json(probe_path) if probe_path else None

    blockers = _dedupe(
        list(final_gate.get("remaining_blockers") or [])
        + list(stack.get("remaining_blockers") or [])
        + list(external.get("remaining_blockers") or [])
        + list(manual.get("remaining_blockers") or [])
    )
    classified = _classify_remaining_blockers(blockers)
    manual_unconfirmed = list(manual.get("unconfirmed_required_item_ids") or [])
    manual_private = list(manual.get("manual_private_item_ids") or [])
    source_manifest_sha256 = ((manual.get("crosscheck") or {}).get("source_manifest_sha256") or "")

    local_ready = stack.get("ok") is True and stack.get("local_release_candidate_ready") is True
    external_ready = external.get("external_proceedings_metadata_ready") is True
    manual_ready = manual.get("manual_submission_system_ready") is True
    final_ready = final_gate.get("final_submission_ready") is True

    closure_groups = [
        {
            "group_id": "local_artifact_handoff",
            "status": "ready" if local_ready else "not_ready",
            "public_safe": True,
            "can_close_without_private_data": True,
            "current_evidence": {
                "local_release_candidate_ready": stack.get("local_release_candidate_ready") is True,
                "readiness_scope": stack.get("readiness_scope", ""),
                "blocking_status": stack.get("blocking_status", ""),
                "failure_count": len(stack.get("failures") or []),
            },
            "remaining_blockers": [],
            "closure_conditions": [
                "Keep the stack artifact fresh after any paper, bibliography, package, or metadata change.",
            ],
            "next_commands": [
                "python -m scripts.audit.main_refresh_submission_release_candidate_stack --stamp YYYYMMDD --output-json outputs/summary/paper_critical/submission_release_candidate_stack_refresh_YYYYMMDD.json --output-md outputs/summary/paper_critical/submission_release_candidate_stack_refresh_YYYYMMDD.md",
            ],
        },
        {
            "group_id": "external_proceedings_metadata",
            "status": "ready" if external_ready else "blocked",
            "public_safe": True,
            "can_close_without_private_data": True,
            "current_evidence": _promax_evidence(external),
            "latest_public_probe": _promax_probe_evidence(probe),
            "remaining_blockers": classified["external_proceedings_metadata"],
            "closure_conditions": [
                "Add the final ProMax ACM page range to Paper/references.bib when it is public.",
                "Direct Crossref /works lookup for DOI 10.1145/3805712.3809600 must return 200 with matching DOI metadata.",
                "The DOI resolver https://doi.org/10.1145/3805712.3809600 must resolve successfully.",
                "Rerun the external proceedings metadata audit and the release-candidate stack after metadata changes.",
            ],
            "next_commands": [
                "python -m scripts.audit.main_probe_promax_public_metadata --network-mode live --timeout-seconds 45 --output-json outputs/summary/paper_critical/promax_public_metadata_probe_YYYYMMDD.json --output-md outputs/summary/paper_critical/promax_public_metadata_probe_YYYYMMDD.md",
                "python -m scripts.audit.main_audit_external_proceedings_metadata --network-mode live --timeout-seconds 45 --output-json outputs/summary/paper_critical/external_proceedings_metadata_recheck_YYYYMMDD.json --output-md outputs/summary/paper_critical/external_proceedings_metadata_recheck_YYYYMMDD.md",
                "python -m scripts.audit.main_refresh_submission_release_candidate_stack --stamp YYYYMMDD --external-timeout-seconds 45 --output-json outputs/summary/paper_critical/submission_release_candidate_stack_refresh_YYYYMMDD.json --output-md outputs/summary/paper_critical/submission_release_candidate_stack_refresh_YYYYMMDD.md",
            ],
        },
        {
            "group_id": "manual_submission_system",
            "status": "ready" if manual_ready else "manual_private_pending",
            "public_safe": False,
            "can_close_without_private_data": False,
            "current_evidence": {
                "manual_submission_checklist_ready": manual.get("manual_submission_checklist_ready") is True,
                "manual_submission_system_ready": manual_ready,
                "private_confirmation_provided": ((manual.get("private_confirmation") or {}).get("provided") is True),
                "source_manifest_sha256": source_manifest_sha256,
                "unconfirmed_required_item_ids": manual_unconfirmed,
                "manual_private_item_ids": manual_private,
            },
            "remaining_blockers": classified["manual_submission_system"],
            "closure_conditions": [
                "A human must complete the submission-system fields outside git.",
                "Create an untracked private confirmation JSON from configs/paper_manual_submission_private_confirmation.template.json.",
                "Set confirmed_in_submission_system=true, private_fields_completed_in_submission_system=true, no_private_fields_stored=true, source_manifest_sha256 to the current value, and completed_item_ids to the required checklist IDs.",
                "Do not store author identities, conflicts, reviewer preferences, declarations, account metadata, or other private payloads in git.",
            ],
            "next_commands": [
                "python -m scripts.audit.main_build_manual_submission_checklist --private-confirmation-json path/to/untracked_private_confirmation.json --output-json outputs/summary/paper_critical/manual_submission_checklist_YYYYMMDD.json --output-md outputs/summary/paper_critical/manual_submission_checklist_YYYYMMDD.md",
                "python -m scripts.audit.main_refresh_submission_release_candidate_stack --stamp YYYYMMDD --manual-private-confirmation-json path/to/untracked_private_confirmation.json --external-timeout-seconds 45 --output-json outputs/summary/paper_critical/submission_release_candidate_stack_refresh_YYYYMMDD.json --output-md outputs/summary/paper_critical/submission_release_candidate_stack_refresh_YYYYMMDD.md",
            ],
        },
    ]

    failures: list[str] = []
    if not local_ready:
        failures.append("local_release_candidate_stack_not_ready")
    if final_ready and (classified["external_proceedings_metadata"] or classified["manual_submission_system"]):
        failures.append("final_submission_ready_true_but_blockers_remain")

    return {
        "schema_version": "2026-06-12.final_submission_blocker_closure_packet.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "local_read_only_final_submission_blocker_closure_packet",
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "stamp": stamp,
        "ok": not failures,
        "closure_packet_ready": not failures,
        "local_release_candidate_ready": local_ready,
        "final_submission_ready": final_ready,
        "external_proceedings_metadata_ready": external_ready,
        "manual_submission_system_ready": manual_ready,
        "ready_for_human_handoff": local_ready and not final_ready,
        "input_paths": {
            "final_gate_json": _path_state(final_path, repo),
            "external_metadata_json": _path_state(external_path, repo),
            "manual_checklist_json": _path_state(manual_path, repo),
            "release_candidate_stack_json": _path_state(stack_path, repo),
            "promax_probe_json": _path_state(probe_path, repo) if probe_path else {"path": "", "exists": False, "size_bytes": 0},
        },
        "remaining_blocker_count": len(blockers),
        "remaining_blockers": blockers,
        "classified_remaining_blockers": classified,
        "closure_groups": closure_groups,
        "failures": failures,
        "warnings": _dedupe(list(final_gate.get("warnings") or []) + list(stack.get("warnings") or [])),
        "next_actions": [
            "Monitor or recheck ProMax public ACM/Crossref/DOI metadata; update BibTeX only after final public page range is available.",
            "Prepare the private manual submission confirmation outside git after the submission-system fields are completed.",
            "Rerun the release-candidate stack and this closure packet after either blocker group changes.",
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_md(path: Path, packet: dict[str, Any]) -> None:
    lines = [
        "# Final Submission Blocker Closure Packet",
        "",
        f"Generated: {packet['created_at_utc']}",
        "",
        f"- OK: `{str(packet['ok']).lower()}`",
        f"- Closure packet ready: `{str(packet['closure_packet_ready']).lower()}`",
        f"- Local release candidate ready: `{str(packet['local_release_candidate_ready']).lower()}`",
        f"- Final submission ready: `{str(packet['final_submission_ready']).lower()}`",
        f"- External proceedings metadata ready: `{str(packet['external_proceedings_metadata_ready']).lower()}`",
        f"- Manual submission system ready: `{str(packet['manual_submission_system_ready']).lower()}`",
        f"- Remaining blocker count: `{packet['remaining_blocker_count']}`",
        "",
        "## Closure Groups",
        "",
    ]
    for group in packet.get("closure_groups", []):
        lines.extend(
            [
                f"### {group['group_id']}",
                "",
                f"- Status: `{group['status']}`",
                f"- Public safe: `{str(group['public_safe']).lower()}`",
                f"- Can close without private data: `{str(group['can_close_without_private_data']).lower()}`",
                "",
                "Remaining blockers:",
            ]
        )
        blockers = group.get("remaining_blockers") or []
        lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- None")
        lines.extend(["", "Closure conditions:"])
        lines.extend(f"- {item}" for item in group.get("closure_conditions", []))
        lines.extend(["", "Next commands:"])
        lines.extend(f"- `{item}`" for item in group.get("next_commands", []))
        evidence = group.get("current_evidence") or {}
        if group["group_id"] == "external_proceedings_metadata":
            bib = (evidence.get("bibtex") or {})
            probe = group.get("latest_public_probe") or {}
            lines.extend(
                [
                    "",
                    "Current ProMax evidence:",
                    f"- DOI: `{bib.get('doi', '')}`",
                    f"- Pages: `{bib.get('pages', '')}`",
                    f"- Num pages: `{bib.get('numpages', '')}`",
                    f"- ISBN: `{bib.get('isbn', '')}`",
                    f"- Location: `{bib.get('location', '')}`",
                    f"- Crossref status: `{evidence.get('crossref_status_code')}`",
                    f"- DOI resolver status: `{evidence.get('doi_resolver_status_code')}`",
                ]
            )
            if probe.get("provided"):
                lines.extend(
                    [
                        "",
                        "Latest public probe:",
                        f"- Created: `{probe.get('created_at_utc', '')}`",
                        f"- ProMax public metadata ready: `{str(probe.get('promax_public_metadata_ready')).lower()}`",
                        f"- Crossref status: `{probe.get('crossref_status_code')}`",
                        f"- DOI resolver status: `{probe.get('doi_resolver_status_code')}`",
                        f"- ACM DL status: `{probe.get('acm_dl_status_code')}`",
                    ]
                )
        if group["group_id"] == "manual_submission_system":
            lines.extend(
                [
                    "",
                    "Manual confirmation safe fields:",
                    f"- Source manifest sha256: `{evidence.get('source_manifest_sha256', '')}`",
                    f"- Unconfirmed item IDs: `{', '.join(evidence.get('unconfirmed_required_item_ids') or [])}`",
                    f"- Private item IDs: `{', '.join(evidence.get('manual_private_item_ids') or [])}`",
                ]
            )
        lines.append("")

    lines.extend(["## Remaining Blockers", ""])
    blockers = packet.get("remaining_blockers") or []
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- None")
    lines.extend(["", "## Failures", ""])
    failures = packet.get("failures") or []
    lines.extend(f"- `{item}`" for item in failures) if failures else lines.append("- None")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in packet.get("next_actions", []))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stamp", default=DEFAULT_STAMP)
    parser.add_argument("--final-gate-json")
    parser.add_argument("--external-metadata-json")
    parser.add_argument("--manual-checklist-json")
    parser.add_argument("--release-candidate-stack-json")
    parser.add_argument("--promax-probe-json")
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packet = build_final_submission_blocker_closure_packet(
        root=args.root,
        output_dir=args.output_dir,
        stamp=args.stamp,
        final_gate_json=args.final_gate_json,
        external_metadata_json=args.external_metadata_json,
        manual_checklist_json=args.manual_checklist_json,
        release_candidate_stack_json=args.release_candidate_stack_json,
        promax_probe_json=args.promax_probe_json,
    )
    if args.output_json:
        _write_json(Path(args.output_json), packet)
    if args.output_md:
        _write_md(Path(args.output_md), packet)
    if not args.output_json:
        print(json.dumps(packet, indent=2, sort_keys=True))
    return 0 if packet["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
