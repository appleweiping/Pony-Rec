from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CLAIM_AUDIT_JSON = Path(
    "outputs/summary/paper_critical/final_paper_claim_audit_after_full_panel_review_20260612.json"
)
DEFAULT_PANEL_REVIEW_JSON = Path(
    "outputs/summary/paper_critical/final_full_manuscript_panel_review_20260612.json"
)
DEFAULT_METADATA_FOLLOWUP_JSON = Path(
    "outputs/summary/paper_critical/final_pdf_polish_metadata_followup_20260612.json"
)
DEFAULT_TARGET_PROFILE_JSON = Path("configs/paper_submission_profiles.json")

INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
OUTPUT_RE = re.compile(r"Output written on .*?\((\d+) pages?, (\d+) bytes\)")
AUTHOR_RE = re.compile(r"\\author\{([^{}]+)\}")
DOCUMENTCLASS_RE = re.compile(r"\\documentclass(?:\[([^\]]*)\])?\{([^}]+)\}")
FORBIDDEN_TOKEN_RE = re.compile(r"\b(TODO|TBD|FIXME|PLACEHOLDER)\b|\?\?\?", re.IGNORECASE)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _file_state(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": _rel(path, root),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_entry(path: Path, *, root: Path, role: str) -> dict[str, Any]:
    state = _file_state(path, root)
    state["role"] = role
    state["sha256"] = _sha256_file(path) if path.exists() and path.is_file() else ""
    return state


def _candidate_tex_paths(raw: str, *, paper_dir: Path, current_dir: Path) -> list[Path]:
    value = raw.strip()
    suffix = "" if Path(value).suffix else ".tex"
    rel = Path(value + suffix)
    return [paper_dir / rel, current_dir / rel]


def _candidate_graphic_paths(raw: str, *, paper_dir: Path, current_dir: Path) -> list[Path]:
    value = raw.strip()
    base = Path(value)
    suffixes = [base.suffix] if base.suffix else [".pdf", ".png", ".jpg", ".jpeg", ".svg"]
    candidates: list[Path] = []
    for suffix in suffixes:
        rel = base if base.suffix else Path(value + suffix)
        candidates.extend([paper_dir / rel, current_dir / rel])
    return candidates


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _discover_source_closure(paper_dir: Path, root: Path) -> dict[str, Any]:
    entry = paper_dir / "main.tex"
    seen: set[Path] = set()
    queue = [entry]
    input_edges: list[dict[str, str]] = []
    missing_inputs: list[str] = []
    graphics: list[dict[str, Any]] = []
    forbidden_hits: list[dict[str, Any]] = []

    while queue:
        path = queue.pop(0).resolve()
        if path in seen:
            continue
        seen.add(path)
        text = _read_text(path)
        if not path.exists():
            continue

        for idx, line in enumerate(text.splitlines(), start=1):
            if FORBIDDEN_TOKEN_RE.search(line):
                forbidden_hits.append(
                    {"file": _rel(path, root), "line": idx, "text": line.strip()[:200]}
                )

        for raw in INPUT_RE.findall(text):
            candidates = _candidate_tex_paths(raw, paper_dir=paper_dir, current_dir=path.parent)
            target = _first_existing(candidates).resolve()
            input_edges.append({"from": _rel(path, root), "input": raw, "resolved": _rel(target, root)})
            if target.exists():
                queue.append(target)
            else:
                missing_inputs.append(raw)

        for raw in GRAPHICS_RE.findall(text):
            candidates = _candidate_graphic_paths(raw, paper_dir=paper_dir, current_dir=path.parent)
            target = _first_existing(candidates)
            graphics.append(
                {
                    "from": _rel(path, root),
                    "graphic": raw,
                    "resolved": _rel(target, root),
                    "exists": target.exists(),
                    "size_bytes": target.stat().st_size if target.exists() else 0,
                }
            )

    return {
        "entry": _file_state(entry, root),
        "tex_files": sorted(_rel(path, root) for path in seen if path.exists()),
        "input_edges": input_edges,
        "missing_inputs": sorted(set(missing_inputs)),
        "graphics": graphics,
        "missing_graphics": [item["graphic"] for item in graphics if not item["exists"]],
        "forbidden_hits": forbidden_hits,
    }


def _parse_build_state(paper_dir: Path, root: Path) -> dict[str, Any]:
    pdf = paper_dir / "main.pdf"
    log = paper_dir / "main.log"
    blg = paper_dir / "main.blg"
    aux = paper_dir / "main.aux"
    log_text = _read_text(log)
    blg_text = _read_text(blg)
    aux_text = _read_text(aux)

    output_matches = OUTPUT_RE.findall(log_text)
    page_count = int(output_matches[-1][0]) if output_matches else None
    logged_pdf_bytes = int(output_matches[-1][1]) if output_matches else None
    warning_match = re.search(r"warning\$\s*--\s*(\d+)", blg_text)
    bibtex_warning_count = int(warning_match.group(1)) if warning_match else None

    cited_keys: set[str] = set()
    for part in aux_text.split("\\citation{")[1:]:
        cited_keys.update(key.strip() for key in part.split("}", 1)[0].split(",") if key.strip())

    return {
        "pdf": _file_state(pdf, root),
        "log": _file_state(log, root),
        "blg": _file_state(blg, root),
        "aux": _file_state(aux, root),
        "page_count": page_count,
        "logged_pdf_bytes": logged_pdf_bytes,
        "bibtex_warning_count": bibtex_warning_count,
        "undefined_citation_count": len(re.findall(r"Citation `[^']+' undefined", log_text)),
        "undefined_reference_count": len(re.findall(r"Reference `[^']+' undefined", log_text)),
        "rerun_warning_count": log_text.count("Rerun to get cross-references right"),
        "overfull_hbox_count": log_text.count("Overfull \\hbox"),
        "underfull_hbox_count": log_text.count("Underfull \\hbox"),
        "underfull_vbox_count": log_text.count("Underfull \\vbox"),
        "cited_key_count": len(cited_keys),
    }


def _build_source_package_manifest(paper_dir: Path, root: Path, source: dict[str, Any]) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}

    def add(path: Path, role: str) -> None:
        if not path.exists() or not path.is_file():
            return
        rel = _rel(path, root)
        existing = files.get(rel)
        if existing is None:
            files[rel] = _manifest_entry(path, root=root, role=role)
        elif role not in existing["role"].split(","):
            existing["role"] = existing["role"] + "," + role

    for rel in source["tex_files"]:
        add(root / rel, "tex_source")
    add(paper_dir / "references.bib", "bibliography_source")
    add(paper_dir / "main.bbl", "compiled_bibliography")
    add(paper_dir / "main.pdf", "compiled_pdf")
    for graphic in source["graphics"]:
        if graphic["exists"]:
            add(root / graphic["resolved"], "figure")

    ordered = [files[key] for key in sorted(files)]
    manifest_lines = [
        f"{entry['sha256']}  {entry['path']}  {entry['role']}  {entry['size_bytes']}"
        for entry in ordered
    ]
    external_refs = [
        entry["path"]
        for entry in ordered
        if not entry["path"].replace("\\", "/").startswith(_rel(paper_dir, root).replace("\\", "/") + "/")
    ]
    return {
        "file_count": len(ordered),
        "total_bytes": sum(int(entry["size_bytes"]) for entry in ordered),
        "manifest_sha256": hashlib.sha256("\n".join(manifest_lines).encode("utf-8")).hexdigest(),
        "files": ordered,
        "external_source_references": external_refs,
    }


def _audit_anonymity(paper_dir: Path) -> dict[str, Any]:
    main_text = _read_text(paper_dir / "main.tex")
    authors = [author.strip() for author in AUTHOR_RE.findall(main_text)]
    return {
        "authors": authors,
        "anonymous_author": bool(authors) and all("anonymous" in author.lower() for author in authors),
        "anonymous_affiliation": "Anonymous Institution" in main_text,
        "email_macro_count": main_text.count("\\email{"),
        "thanks_macro_count": main_text.count("\\thanks{"),
    }


def _documentclass_state(paper_dir: Path) -> dict[str, Any]:
    main_text = _read_text(paper_dir / "main.tex")
    match = DOCUMENTCLASS_RE.search(main_text)
    options = [item.strip() for item in match.group(1).split(",")] if match and match.group(1) else []
    return {
        "documentclass": match.group(2).strip() if match else "",
        "options": sorted(option for option in options if option),
    }


def _load_target_profile(path: Path, profile_id: str | None) -> dict[str, Any]:
    if not path.exists():
        return {
            "configured": False,
            "profile_id": profile_id or "",
            "failures": [f"missing_target_profile_json:{path}"],
        }
    payload = _read_json(path)
    selected_id = profile_id or str(payload.get("default_profile_id") or "")
    profiles = payload.get("profiles") or {}
    profile = profiles.get(selected_id)
    if not isinstance(profile, dict):
        return {
            "configured": False,
            "profile_id": selected_id,
            "failures": [f"missing_target_profile:{selected_id}"],
        }
    return {
        "configured": True,
        "profile_id": selected_id,
        "schema_version": payload.get("schema_version"),
        "profile": profile,
        "failures": [],
    }


def _audit_target_profile(
    *,
    profile_state: dict[str, Any],
    documentclass: dict[str, Any],
    build: dict[str, Any],
    anonymity: dict[str, Any],
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    failures = list(profile_state.get("failures") or [])
    profile = profile_state.get("profile") or {}
    if not profile_state.get("configured"):
        return {
            "configured": False,
            "profile_id": profile_state.get("profile_id") or "",
            "ok": False,
            "profile": profile,
            "documentclass": documentclass,
            "failures": failures,
        }

    required_class = str(profile.get("documentclass") or "")
    if required_class and documentclass["documentclass"] != required_class:
        failures.append(f"documentclass_mismatch:{documentclass['documentclass']} != {required_class}")
    options = set(documentclass["options"])
    for option in profile.get("required_documentclass_options") or []:
        if option not in options:
            failures.append(f"missing_documentclass_option:{option}")

    max_pages = profile.get("max_total_pages")
    if max_pages is not None and build["page_count"] is not None and int(build["page_count"]) > int(max_pages):
        failures.append(f"target_profile_page_count_exceeds_limit:{build['page_count']} > {max_pages}")
    if profile.get("require_anonymous_author_shell") and not anonymity["anonymous_author"]:
        failures.append("target_profile_requires_anonymous_author_shell")
    if profile.get("require_single_pdf") and not build["pdf"]["exists"]:
        failures.append("target_profile_requires_single_pdf")
    if profile.get("require_local_source_manifest") and source_manifest["external_source_references"]:
        failures.append("target_profile_requires_local_source_manifest")
    if profile.get("require_bibtex_clean") and build["bibtex_warning_count"] != 0:
        failures.append("target_profile_requires_clean_bibtex")
    if profile.get("require_no_undefined_references") and (
        build["undefined_citation_count"] or build["undefined_reference_count"]
    ):
        failures.append("target_profile_requires_defined_references")
    if profile.get("require_no_overfull_hbox") and build["overfull_hbox_count"] != 0:
        failures.append("target_profile_requires_no_overfull_hbox")

    return {
        "configured": True,
        "profile_id": profile_state["profile_id"],
        "ok": not failures,
        "profile": profile,
        "documentclass": documentclass,
        "failures": failures,
    }


def _score_floor(panel_review: dict[str, Any]) -> float | None:
    consensus = panel_review.get("reviewer_consensus") or {}
    raw = str(consensus.get("score_floor") or "")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*10", raw)
    return float(match.group(1)) if match else None


def build_submission_package_audit(
    *,
    root: str | Path = ".",
    paper_dir: str | Path = "Paper",
    claim_audit_json: str | Path = DEFAULT_CLAIM_AUDIT_JSON,
    panel_review_json: str | Path = DEFAULT_PANEL_REVIEW_JSON,
    metadata_followup_json: str | Path = DEFAULT_METADATA_FOLLOWUP_JSON,
    target_profile_json: str | Path = DEFAULT_TARGET_PROFILE_JSON,
    target_profile_id: str | None = None,
    max_pages: int = 9,
    max_overfull_hbox: int = 0,
    target_formatting_closed: bool = False,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    paper = (repo / paper_dir).resolve()
    claim_path = repo / claim_audit_json
    panel_path = repo / panel_review_json
    metadata_path = repo / metadata_followup_json
    target_profile_path = repo / target_profile_json

    source = _discover_source_closure(paper, repo)
    build = _parse_build_state(paper, repo)
    source_manifest = _build_source_package_manifest(paper, repo, source)
    anonymity = _audit_anonymity(paper)
    documentclass = _documentclass_state(paper)
    profile_state = _load_target_profile(target_profile_path, target_profile_id)
    target_profile = _audit_target_profile(
        profile_state=profile_state,
        documentclass=documentclass,
        build=build,
        anonymity=anonymity,
        source_manifest=source_manifest,
    )
    claim_audit = _read_json(claim_path)
    panel_review = _read_json(panel_path)
    metadata_followup = _read_json(metadata_path)

    failures: list[str] = []
    warnings: list[str] = []

    required_files = {
        "main_tex": paper / "main.tex",
        "references_bib": paper / "references.bib",
        "main_pdf": paper / "main.pdf",
        "main_log": paper / "main.log",
        "main_blg": paper / "main.blg",
        "main_bbl": paper / "main.bbl",
        "framework_overview_pdf": paper / "figures" / "framework_overview.pdf",
        "framework_overview_svg": paper / "figures" / "framework_overview.svg",
    }
    missing_required = [label for label, path in required_files.items() if not path.exists()]
    failures.extend(f"missing_required_file:{label}:{_rel(required_files[label], repo)}" for label in missing_required)
    failures.extend(f"missing_input:{item}" for item in source["missing_inputs"])
    failures.extend(f"missing_graphic:{item}" for item in source["missing_graphics"])
    failures.extend(
        f"forbidden_token:{hit['file']}:{hit['line']}:{hit['text']}" for hit in source["forbidden_hits"]
    )

    if not build["pdf"]["exists"] or build["pdf"]["size_bytes"] <= 100_000:
        failures.append("main_pdf_missing_or_too_small")
    if build["page_count"] is None:
        failures.append("page_count_not_found_in_log")
    elif build["page_count"] > max_pages:
        failures.append(f"page_count_exceeds_limit:{build['page_count']} > {max_pages}")
    if build["bibtex_warning_count"] is None:
        failures.append("bibtex_warning_count_not_found")
    elif build["bibtex_warning_count"] != 0:
        failures.append(f"bibtex_warnings:{build['bibtex_warning_count']}")
    if build["undefined_citation_count"]:
        failures.append(f"undefined_citations:{build['undefined_citation_count']}")
    if build["undefined_reference_count"]:
        failures.append(f"undefined_references:{build['undefined_reference_count']}")
    if build["rerun_warning_count"]:
        failures.append(f"latex_rerun_warnings:{build['rerun_warning_count']}")
    if build["overfull_hbox_count"] > max_overfull_hbox:
        failures.append(f"overfull_hbox_count:{build['overfull_hbox_count']} > {max_overfull_hbox}")
    if build["underfull_hbox_count"] or build["underfull_vbox_count"]:
        warnings.append(
            "underfull_layout_warnings:"
            f"hbox={build['underfull_hbox_count']},vbox={build['underfull_vbox_count']}"
        )
    for rel in source_manifest["external_source_references"]:
        failures.append(f"external_source_file_reference:{rel}")
    failures.extend(f"target_profile:{failure}" for failure in target_profile["failures"])

    if not anonymity["anonymous_author"] or not anonymity["anonymous_affiliation"]:
        failures.append("anonymous_author_or_affiliation_not_ready")
    if anonymity["email_macro_count"] or anonymity["thanks_macro_count"]:
        failures.append("author_email_or_thanks_macro_present")

    claim_ok = (
        claim_audit.get("ok") is True
        and claim_audit.get("paper_evidence_ready_for_drafting") is True
        and claim_audit.get("final_submission_ready") is False
    )
    if not claim_ok:
        failures.append("claim_audit_not_in_expected_scoped_ready_state")

    floor = _score_floor(panel_review)
    consensus = panel_review.get("reviewer_consensus") or {}
    panel_ok = (
        floor is not None
        and floor >= 8.0
        and consensus.get("new_experiment_required") is False
        and consensus.get("claim_boundary_ok") is True
        and consensus.get("final_submission_ready") is False
    )
    if not panel_ok:
        failures.append("final_panel_review_not_in_expected_conditional_pass_state")

    remaining_blockers = list(metadata_followup.get("remaining_blockers") or [])
    if target_profile["ok"]:
        remaining_blockers = [
            blocker
            for blocker in remaining_blockers
            if "format" not in blocker.lower() and "target" not in blocker.lower()
        ]
        if not target_formatting_closed:
            remaining_blockers.append(
                "Final manual submission-system metadata/format checklist is not closed."
            )
    elif not any("format" in blocker.lower() for blocker in remaining_blockers):
        remaining_blockers.append("External submission-target-specific formatting pass is not closed.")
    final_submission_ready = not failures and target_formatting_closed and not remaining_blockers

    return {
        "schema_version": "2026-06-12.submission_package_audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "local_read_only_submission_package_audit",
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "ok": not failures,
        "submission_package_ready_for_target_formatting": not failures,
        "final_submission_ready": final_submission_ready,
        "verdict": (
            "READY_FOR_TARGET_FORMATTING_NOT_FINAL_SUBMISSION"
            if not failures and not final_submission_ready
            else "FINAL_SUBMISSION_READY"
            if final_submission_ready
            else "NEEDS_SUBMISSION_PACKAGE_REPAIR"
        ),
        "root": str(repo),
        "paper_dir": _rel(paper, repo),
        "limits": {
            "max_pages": max_pages,
            "max_overfull_hbox": max_overfull_hbox,
            "target_formatting_closed": target_formatting_closed,
        },
        "input_paths": {
            "claim_audit": _file_state(claim_path, repo),
            "panel_review": _file_state(panel_path, repo),
            "metadata_followup": _file_state(metadata_path, repo),
            "target_profile": _file_state(target_profile_path, repo),
        },
        "required_files": {label: _file_state(path, repo) for label, path in required_files.items()},
        "source_closure": source,
        "source_package_manifest": source_manifest,
        "build": build,
        "anonymity": anonymity,
        "documentclass": documentclass,
        "target_formatting_profile": target_profile,
        "evidence_gates": {
            "claim_audit_ok": claim_ok,
            "panel_review_ok": panel_ok,
            "panel_score_floor": floor,
            "metadata_followup_verdict": metadata_followup.get("verdict"),
            "target_formatting_profile_ok": target_profile["ok"],
        },
        "warnings": warnings,
        "failures": failures,
        "remaining_blockers": remaining_blockers,
        "next_actions": [
            "Run the final manual submission-system metadata/format checklist on the audited Paper package.",
            "Recheck ProMax final ACM page range and ACM/Crossref visibility immediately before submission.",
            "Keep final_submission_ready=false until external metadata and formatting blockers are closed.",
        ],
    }


def _write_md(path: Path, audit: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Submission Package Audit",
        "",
        f"Generated: {audit['created_at_utc']}",
        "",
        f"- Verdict: `{audit['verdict']}`",
        f"- OK: `{str(audit['ok']).lower()}`",
        "- Submission package ready for target formatting: "
        f"`{str(audit['submission_package_ready_for_target_formatting']).lower()}`",
        f"- Final submission ready: `{str(audit['final_submission_ready']).lower()}`",
        f"- PDF pages: `{audit['build']['page_count']}`",
        f"- PDF bytes: `{audit['build']['pdf']['size_bytes']}`",
        f"- BibTeX warnings: `{audit['build']['bibtex_warning_count']}`",
        f"- Overfull hbox warnings: `{audit['build']['overfull_hbox_count']}`",
        f"- Underfull hbox/vbox warnings: `{audit['build']['underfull_hbox_count']}` / "
        f"`{audit['build']['underfull_vbox_count']}`",
        f"- Cited keys: `{audit['build']['cited_key_count']}`",
        f"- Panel score floor: `{audit['evidence_gates']['panel_score_floor']}`",
        "- Target formatting profile: "
        f"`{audit['target_formatting_profile']['profile_id']}` "
        f"ok=`{str(audit['target_formatting_profile']['ok']).lower()}`",
        f"- Source package manifest files: `{audit['source_package_manifest']['file_count']}`",
        f"- Source package manifest sha256: `{audit['source_package_manifest']['manifest_sha256']}`",
        "",
        "## Remaining Blockers",
        "",
    ]
    blockers = audit.get("remaining_blockers") or []
    lines.extend(f"- {blocker}" for blocker in blockers) if blockers else lines.append("- None")
    lines.extend(["", "## Failures", ""])
    failures = audit.get("failures") or []
    lines.extend(f"- `{failure}`" for failure in failures) if failures else lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    warnings = audit.get("warnings") or []
    lines.extend(f"- `{warning}`" for warning in warnings) if warnings else lines.append("- None")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {action}" for action in audit.get("next_actions", []))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--paper-dir", default="Paper")
    parser.add_argument("--claim-audit-json", default=str(DEFAULT_CLAIM_AUDIT_JSON))
    parser.add_argument("--panel-review-json", default=str(DEFAULT_PANEL_REVIEW_JSON))
    parser.add_argument("--metadata-followup-json", default=str(DEFAULT_METADATA_FOLLOWUP_JSON))
    parser.add_argument("--target-profile-json", default=str(DEFAULT_TARGET_PROFILE_JSON))
    parser.add_argument("--target-profile-id")
    parser.add_argument("--max-pages", type=int, default=9)
    parser.add_argument("--max-overfull-hbox", type=int, default=0)
    parser.add_argument("--target-formatting-closed", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_submission_package_audit(
        root=args.root,
        paper_dir=args.paper_dir,
        claim_audit_json=args.claim_audit_json,
        panel_review_json=args.panel_review_json,
        metadata_followup_json=args.metadata_followup_json,
        target_profile_json=args.target_profile_json,
        target_profile_id=args.target_profile_id,
        max_pages=args.max_pages,
        max_overfull_hbox=args.max_overfull_hbox,
        target_formatting_closed=args.target_formatting_closed,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        _write_md(Path(args.output_md), audit)
    if not args.output_json:
        print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
