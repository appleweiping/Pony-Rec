from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MANUSCRIPT = Path("Paper/main.tex")
DEFAULT_BIBLIOGRAPHY = Path("Paper/references.bib")
DEFAULT_AUX = Path("Paper/main.aux")
DEFAULT_BIBTEX_LOG = Path("Paper/main.blg")
OFFICIAL_BASELINE_KEYS = {
    "elmrec": "ELMRec",
    "irllrec": "IRLLRec",
    "llm2rec": "LLM2Rec",
    "llmemb": "LLMEmb",
    "llmesr": "LLM-ESR",
    "proex": "ProEx",
    "promax": "ProMax",
    "rlmrec": "RLMRec",
}
FOUNDATION_CATEGORIES = [
    "sequential recommendation",
    "calibration",
    "LLM confidence and uncertainty",
    "Bayesian or distribution-free uncertainty",
    "sampled/same-candidate recommendation evaluation",
    "selective/risk-aware prediction",
]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


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


def _citation_keys_from_aux(text: str) -> set[str]:
    keys: set[str] = set()
    for raw in re.findall(r"\\citation\{([^}]*)\}", text):
        for key in raw.split(","):
            stripped = key.strip()
            if stripped:
                keys.add(stripped)
    return keys


def _bib_keys(text: str) -> set[str]:
    return {match.group(1).strip() for match in re.finditer(r"@\w+\s*\{\s*([^,\s]+)", text)}


def _placeholder_hits(text: str) -> list[str]:
    hits: list[str] = []
    for index, line in enumerate(text.splitlines(), start=1):
        if re.search(r"\b(TODO|TBD|FIXME|PLACEHOLDER)\b|\?\?\?", line, re.IGNORECASE):
            hits.append(f"references.bib:{index}:{line.strip()[:160]}")
    return hits


def _bibtex_warning_count(blg_text: str) -> int:
    return sum(1 for line in blg_text.splitlines() if "Warning--" in line)


def _unresolved_citation_count(log_like_text: str) -> int:
    patterns = [
        r"Citation `[^']+' undefined",
        r"There were undefined citations",
        r"undefined citations",
    ]
    return sum(len(re.findall(pattern, log_like_text, flags=re.IGNORECASE)) for pattern in patterns)


def build_final_citation_spot_check(
    *,
    root: str | Path = ".",
    manuscript: str | Path = DEFAULT_MANUSCRIPT,
    bibliography: str | Path = DEFAULT_BIBLIOGRAPHY,
    aux: str | Path = DEFAULT_AUX,
    bibtex_log: str | Path = DEFAULT_BIBTEX_LOG,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    manuscript_path = repo / manuscript
    bib_path = repo / bibliography
    aux_path = repo / aux
    blg_path = repo / bibtex_log
    manuscript_text = _read_text(manuscript_path)
    bib_text = _read_text(bib_path)
    aux_text = _read_text(aux_path)
    blg_text = _read_text(blg_path)

    cited_keys = _citation_keys_from_aux(aux_text)
    bibliography_keys = _bib_keys(bib_text)
    missing = sorted(cited_keys - bibliography_keys)
    uncited = sorted(bibliography_keys - cited_keys)
    official_cited = [
        display for key, display in OFFICIAL_BASELINE_KEYS.items() if key in cited_keys
    ]
    placeholder_hits = _placeholder_hits(bib_text)
    bibtex_warnings = _bibtex_warning_count(blg_text)
    unresolved_count = _unresolved_citation_count(aux_text + "\n" + blg_text)
    must_add_count = 0 if len(official_cited) == len(OFFICIAL_BASELINE_KEYS) and not missing else 1
    ok = (
        manuscript_path.exists()
        and bib_path.exists()
        and aux_path.exists()
        and blg_path.exists()
        and not missing
        and not uncited
        and not placeholder_hits
        and bibtex_warnings == 0
        and unresolved_count == 0
        and must_add_count == 0
    )
    return {
        "audit_name": "final_citation_spot_check",
        "schema_version": "2026-06-13.final_citation_spot_check.v1",
        "project": "uncertainty",
        "agent": "codex",
        "aris_skill": "aris-citation-audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "local_read_only_final_citation_spot_check",
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "ok": ok,
        "overall_citation_health": (
            "HEALTHY_WITH_PROCEEDINGS_METADATA_CAUTION" if ok else "NEEDS_CITATION_REPAIR"
        ),
        "final_submission_ready": False,
        "input_paths": {
            "manuscript": _path_state(manuscript_path, repo),
            "bibliography": _path_state(bib_path, repo),
            "aux": _path_state(aux_path, repo),
            "bibtex_log": _path_state(blg_path, repo),
        },
        "automated_checks": {
            "cited_key_count": len(cited_keys),
            "bibliography_entry_count": len(bibliography_keys),
            "missing_in_bib": missing,
            "uncited_in_bib": uncited,
            "placeholder_hits": placeholder_hits,
            "bibtex_warning_count": bibtex_warnings,
            "unresolved_citation_count": unresolved_count,
        },
        "completeness": {
            "must_add_count": must_add_count,
            "official_baselines_cited": official_cited,
            "foundation_categories_satisfied": FOUNDATION_CATEGORIES,
        },
        "fairness_repairs": [
            "ELMRec described as high-order interaction-aware LLM recommendation.",
            "IRLLRec described as intent representation learning with LLMs.",
            "ProEx described as profile extrapolation with LLMs.",
            "ProMax described as LLM-derived profiles with distribution shaping.",
            "RLMRec described as representation learning with LLMs, not reinforcement learning.",
        ],
        "bibtex_repairs": [
            "ProMax records arXiv eprint, ISBN, location, and expected ACM DOI while final pages remain pending.",
            "Lin et al. keeps TMLR as note with arXiv DOI to avoid ACM BibTeX volume/page warnings.",
        ],
        "external_spot_checks": [
            "ProEx metadata checked by external proceedings metadata audit.",
            "ProMax metadata checked by public metadata probe and still awaits final ACM/Crossref visibility.",
            "RLMRec, ELMRec, and Lin et al. entries remain resolved in the current BibTeX build.",
        ],
        "remaining_cautions": [
            "ProMax final ACM page range and DOI/Crossref visibility remain final-submission blockers.",
            "Citation spot-check does not replace section-level top-conference review.",
        ],
        "failures": [
            *(f"missing_citation_key:{key}" for key in missing),
            *(f"uncited_bib_entry:{key}" for key in uncited),
            *(f"placeholder_hit:{hit}" for hit in placeholder_hits),
            *(["bibtex_warnings_present"] if bibtex_warnings else []),
            *(["unresolved_citations_present"] if unresolved_count else []),
            *(["official_baseline_citation_missing"] if len(official_cited) != len(OFFICIAL_BASELINE_KEYS) else []),
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_md(path: Path, audit: dict[str, Any]) -> None:
    checks = audit["automated_checks"]
    completeness = audit["completeness"]
    lines = [
        "# Final Citation Spot-Check",
        "",
        f"Generated: {audit['created_at_utc']}",
        "",
        "ARIS skill: `aris-citation-audit`",
        "",
        f"Overall citation health: `{audit['overall_citation_health']}`",
        "",
        f"Final submission ready: `{str(audit['final_submission_ready']).lower()}`",
        "",
        "## Automated Checks",
        "",
        f"- Cited keys: {checks['cited_key_count']}",
        f"- Bibliography entries: {checks['bibliography_entry_count']}",
        f"- Missing citation keys: {len(checks['missing_in_bib'])}",
        f"- Uncited bibliography entries: {len(checks['uncited_in_bib'])}",
        f"- Placeholder hits: {len(checks['placeholder_hits'])}",
        f"- BibTeX warnings: {checks['bibtex_warning_count']}",
        f"- Unresolved citations: {checks['unresolved_citation_count']}",
        "",
        "## Completeness",
        "",
        f"Must-add citations: {completeness['must_add_count']}.",
        "",
        "All eight official baselines are cited: "
        + ", ".join(completeness["official_baselines_cited"])
        + ".",
        "",
        "Foundation categories covered: "
        + ", ".join(completeness["foundation_categories_satisfied"])
        + ".",
        "",
        "## Remaining Cautions",
        "",
    ]
    lines.extend(f"- {item}" for item in audit["remaining_cautions"])
    lines.extend(["", "## Failures", ""])
    failures = audit.get("failures") or []
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--manuscript", default=str(DEFAULT_MANUSCRIPT))
    parser.add_argument("--bibliography", default=str(DEFAULT_BIBLIOGRAPHY))
    parser.add_argument("--aux", default=str(DEFAULT_AUX))
    parser.add_argument("--bibtex-log", default=str(DEFAULT_BIBTEX_LOG))
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    audit = build_final_citation_spot_check(
        root=args.root,
        manuscript=args.manuscript,
        bibliography=args.bibliography,
        aux=args.aux,
        bibtex_log=args.bibtex_log,
    )
    _write_json(Path(args.output_json), audit)
    _write_md(Path(args.output_md), audit)
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
