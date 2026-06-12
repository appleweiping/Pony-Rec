from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_METADATA_CONFIG = Path("configs/paper_submission_metadata.json")
DEFAULT_SUBMISSION_AUDIT_JSON = Path(
    "outputs/summary/paper_critical/submission_package_audit_20260612.json"
)
TITLE_RE = re.compile(r"\\title\{([^{}]+)\}")
ABSTRACT_RE = re.compile(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.DOTALL)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _strip_latex(text: str) -> str:
    text = re.sub(r"%.*", "", text)
    text = text.replace("~", " ")
    text = re.sub(r"\\cite\{[^{}]*\}", "", text)
    text = re.sub(r"\\(?:methodfull|method)\b", "C-CRP", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\$([^$]*)\$", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_title(main_tex: str) -> str:
    match = TITLE_RE.search(main_tex)
    return _strip_latex(match.group(1)) if match else ""


def _extract_abstract(abstract_tex: str) -> str:
    match = ABSTRACT_RE.search(abstract_tex)
    body = match.group(1) if match else abstract_tex
    return _strip_latex(body)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w@.-]+\b", text))


def build_submission_metadata_packet(
    *,
    root: str | Path = ".",
    paper_dir: str | Path = "Paper",
    metadata_config: str | Path = DEFAULT_METADATA_CONFIG,
    submission_audit_json: str | Path = DEFAULT_SUBMISSION_AUDIT_JSON,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    paper = repo / paper_dir
    config_path = repo / metadata_config
    audit_path = repo / submission_audit_json

    config = _read_json(config_path)
    audit = _read_json(audit_path)
    main_tex = _read_text(paper / "main.tex")
    abstract_tex = _read_text(paper / "sections" / "abstract.tex")
    title = _extract_title(main_tex)
    abstract = _extract_abstract(abstract_tex)
    keywords = [str(item).strip() for item in config.get("suggested_keywords", []) if str(item).strip()]
    topic_areas = [str(item).strip() for item in config.get("suggested_topic_areas", []) if str(item).strip()]

    failures: list[str] = []
    warnings: list[str] = []
    if not title:
        failures.append("missing_title")
    if config.get("submission_title") != title:
        failures.append(f"title_mismatch:{config.get('submission_title')!r} != {title!r}")
    if not abstract:
        failures.append("missing_abstract")
    if not (100 <= _word_count(abstract) <= 250):
        warnings.append(f"abstract_word_count_outside_common_range:{_word_count(abstract)}")
    if len(keywords) < 3:
        failures.append("too_few_keywords")
    if len(topic_areas) < 2:
        warnings.append("few_topic_areas")
    if config.get("anonymous_submission") is not True:
        failures.append("metadata_config_not_anonymous")
    if audit.get("ok") is not True:
        failures.append("submission_package_audit_not_ok")
    if audit.get("submission_package_ready_for_target_formatting") is not True:
        failures.append("submission_package_not_ready_for_target_formatting")
    if audit.get("final_submission_ready") is True:
        failures.append("submission_audit_unexpectedly_final_ready")
    target_profile = audit.get("target_formatting_profile") or {}
    if target_profile.get("profile_id") != config.get("target_profile_id"):
        failures.append("target_profile_mismatch")
    if target_profile.get("ok") is not True:
        failures.append("target_profile_not_ok")

    source_manifest = audit.get("source_package_manifest") or {}
    packet_ready = not failures
    remaining_blockers = list(audit.get("remaining_blockers") or [])

    return {
        "schema_version": "2026-06-12.submission_metadata_packet.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "local_read_only_submission_metadata_packet",
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "ok": packet_ready,
        "submission_metadata_packet_ready": packet_ready,
        "final_submission_ready": False,
        "paper_dir": str(Path(paper_dir)),
        "input_paths": {
            "metadata_config": str(Path(metadata_config)),
            "submission_audit_json": str(Path(submission_audit_json)),
            "main_tex": str(Path(paper_dir) / "main.tex"),
            "abstract_tex": str(Path(paper_dir) / "sections" / "abstract.tex"),
        },
        "submission_fields": {
            "paper_type": config.get("paper_type"),
            "target_profile_id": config.get("target_profile_id"),
            "title": title,
            "abstract": abstract,
            "abstract_word_count": _word_count(abstract),
            "abstract_char_count": len(abstract),
            "keywords": keywords,
            "topic_areas": topic_areas,
            "anonymous_submission": config.get("anonymous_submission"),
        },
        "package_crosscheck": {
            "submission_audit_ok": audit.get("ok"),
            "target_formatting_profile_ok": target_profile.get("ok"),
            "pdf_path": ((audit.get("build") or {}).get("pdf") or {}).get("path"),
            "pdf_size_bytes": ((audit.get("build") or {}).get("pdf") or {}).get("size_bytes"),
            "pdf_pages": (audit.get("build") or {}).get("page_count"),
            "source_manifest_file_count": source_manifest.get("file_count"),
            "source_manifest_sha256": source_manifest.get("manifest_sha256"),
        },
        "manual_fields_not_stored": config.get("manual_fields_not_stored") or [],
        "warnings": warnings,
        "failures": failures,
        "remaining_blockers": remaining_blockers,
        "next_actions": [
            "Use these fields to fill the submission system metadata.",
            "Complete conflict-of-interest, author, and declaration fields inside the submission system.",
            "Recheck ProMax final ACM page range and ACM/Crossref visibility immediately before submission.",
            "Keep final_submission_ready=false until manual submission-system and external metadata blockers are closed.",
        ],
    }


def _write_md(path: Path, packet: dict[str, Any]) -> None:
    fields = packet["submission_fields"]
    cross = packet["package_crosscheck"]
    lines = [
        "# Submission Metadata Packet",
        "",
        f"Generated: {packet['created_at_utc']}",
        "",
        f"- OK: `{str(packet['ok']).lower()}`",
        "- Submission metadata packet ready: "
        f"`{str(packet['submission_metadata_packet_ready']).lower()}`",
        f"- Final submission ready: `{str(packet['final_submission_ready']).lower()}`",
        f"- Target profile: `{fields['target_profile_id']}`",
        f"- Paper type: `{fields['paper_type']}`",
        f"- Title: {fields['title']}",
        f"- Abstract words/chars: `{fields['abstract_word_count']}` / `{fields['abstract_char_count']}`",
        f"- Keywords: {', '.join(fields['keywords'])}",
        f"- Topic areas: {', '.join(fields['topic_areas'])}",
        f"- PDF: `{cross['pdf_path']}`, `{cross['pdf_pages']}` pages, `{cross['pdf_size_bytes']}` bytes",
        f"- Source manifest files: `{cross['source_manifest_file_count']}`",
        f"- Source manifest sha256: `{cross['source_manifest_sha256']}`",
        "",
        "## Abstract",
        "",
        fields["abstract"],
        "",
        "## Remaining Blockers",
        "",
    ]
    blockers = packet.get("remaining_blockers") or []
    lines.extend(f"- {blocker}" for blocker in blockers) if blockers else lines.append("- None")
    lines.extend(["", "## Manual Fields Not Stored", ""])
    lines.extend(f"- {item}" for item in packet.get("manual_fields_not_stored", []))
    lines.extend(["", "## Failures", ""])
    failures = packet.get("failures") or []
    lines.extend(f"- `{failure}`" for failure in failures) if failures else lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    warnings = packet.get("warnings") or []
    lines.extend(f"- `{warning}`" for warning in warnings) if warnings else lines.append("- None")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {action}" for action in packet.get("next_actions", []))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--paper-dir", default="Paper")
    parser.add_argument("--metadata-config", default=str(DEFAULT_METADATA_CONFIG))
    parser.add_argument("--submission-audit-json", default=str(DEFAULT_SUBMISSION_AUDIT_JSON))
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packet = build_submission_metadata_packet(
        root=args.root,
        paper_dir=args.paper_dir,
        metadata_config=args.metadata_config,
        submission_audit_json=args.submission_audit_json,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        _write_md(Path(args.output_md), packet)
    if not args.output_json:
        print(json.dumps(packet, indent=2, sort_keys=True))
    return 0 if packet["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
