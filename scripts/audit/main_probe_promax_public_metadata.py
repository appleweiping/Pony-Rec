from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from scripts.audit.main_audit_external_proceedings_metadata import (
    DEFAULT_BIB,
    DEFAULT_TIMEOUT_SECONDS,
    _network_check,
    _parse_bib_entries,
    _read_json,
)


DEFAULT_DOI = "10.1145/3805712.3809600"
DEFAULT_ARXIV_HTML_URL = "https://arxiv.org/html/2604.26231v1"
DEFAULT_SIGIR_ACCEPTED_URL = "https://sigir2026.org/en-AU/pages/program/accepted-papers"
DEFAULT_UQ_AUTHOR_PROFILE_URL = "https://eecs.uq.edu.au/profile/2696/hongzhi-yin"
DEFAULT_OUTPUT_DIR = Path("outputs/summary/paper_critical")
DEFAULT_STAMP = "20260612"
EXPECTED_TITLE = (
    "ProMax: Exploring the Potential of LLM-derived Profiles with Distribution "
    "Shaping for Recommender Systems"
)
EXPECTED_AUTHORS = "Yi Zhang, Yiwen Zhang, Kai Zheng, Tong Chen, Hongzhi Yin"
EXPECTED_ISBN_PATTERN = "979-8-4007-2599-9/2026/07"


def _without_text(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "text"}


def _crossref_summary(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"parse_error": "crossref_json_decode_failed"}
    message = payload.get("message") or {}
    return {
        "doi": message.get("DOI") or "",
        "title": (message.get("title") or [""])[0],
        "container_title": (message.get("container-title") or [""])[0],
        "page": message.get("page") or "",
        "published_year": (((message.get("published-print") or {}).get("date-parts") or [[None]])[0][0]),
    }


def _source_probe(
    *,
    name: str,
    url: str,
    required_patterns: list[str],
    network_mode: str,
    timeout_seconds: int,
    fixtures: dict[str, Any],
) -> dict[str, Any]:
    response = _network_check(
        url,
        network_mode=network_mode,
        timeout_seconds=timeout_seconds,
        fixtures=fixtures,
    )
    text = str(response.get("text") or "")
    missing = [pattern for pattern in required_patterns if pattern not in text]
    return {
        "name": name,
        "url": url,
        "ok": response.get("ok") is True and not missing,
        "status_code": response.get("status_code"),
        "final_url": response.get("final_url") or url,
        "missing_patterns": missing,
        "required_patterns": required_patterns,
        "response": _without_text(response),
    }


def build_promax_public_metadata_probe(
    *,
    root: str | Path = ".",
    bib_path: str | Path = DEFAULT_BIB,
    doi: str = DEFAULT_DOI,
    network_mode: str = "live",
    fixture_json: str | Path | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    fixtures = _read_json(repo / fixture_json) if fixture_json else {}
    entries = _parse_bib_entries((repo / bib_path).read_text(encoding="utf-8", errors="replace"))
    entry = entries.get("promax") or {"entry_type": "", "fields": {}}
    fields = entry.get("fields") or {}
    encoded_doi = quote(doi, safe="/.")
    crossref_url = f"https://api.crossref.org/works/{encoded_doi}"
    doi_url = f"https://doi.org/{doi}"
    acm_url = f"https://dl.acm.org/doi/{doi}"

    crossref = _network_check(
        crossref_url,
        network_mode=network_mode,
        timeout_seconds=timeout_seconds,
        fixtures=fixtures,
    )
    doi_resolver = _network_check(
        doi_url,
        network_mode=network_mode,
        timeout_seconds=timeout_seconds,
        fixtures=fixtures,
    )
    acm_dl = _network_check(
        acm_url,
        network_mode=network_mode,
        timeout_seconds=timeout_seconds,
        fixtures=fixtures,
    )
    source_probes = [
        _source_probe(
            name="arxiv_html_promax_acm_metadata",
            url=DEFAULT_ARXIV_HTML_URL,
            required_patterns=[
                doi,
                EXPECTED_ISBN_PATTERN,
                "49th International ACM SIGIR Conference",
                "Melbourne, VIC, Australia",
            ],
            network_mode=network_mode,
            timeout_seconds=timeout_seconds,
            fixtures=fixtures,
        ),
        _source_probe(
            name="sigir2026_accepted_papers_promax",
            url=DEFAULT_SIGIR_ACCEPTED_URL,
            required_patterns=[EXPECTED_TITLE, EXPECTED_AUTHORS],
            network_mode=network_mode,
            timeout_seconds=timeout_seconds,
            fixtures=fixtures,
        ),
        _source_probe(
            name="uq_author_profile_promax_sigir2026",
            url=DEFAULT_UQ_AUTHOR_PROFILE_URL,
            required_patterns=[EXPECTED_TITLE, "SIGIR 2026"],
            network_mode=network_mode,
            timeout_seconds=timeout_seconds,
            fixtures=fixtures,
        ),
    ]
    summary = _crossref_summary(str(crossref.get("text") or "")) if crossref.get("ok") else {}
    bib_pages = str(fields.get("pages") or "").strip()
    crossref_page = str(summary.get("page") or "").strip()
    direct_crossref_ok = (
        crossref.get("ok") is True and str(summary.get("doi") or "").lower() == doi.lower()
    )
    doi_resolver_ok = doi_resolver.get("ok") is True
    bibtex_pages_present = bool(bib_pages)
    crossref_page_present = bool(crossref_page)
    public_metadata_ready = direct_crossref_ok and doi_resolver_ok and bibtex_pages_present

    blocker_checks = [
        {
            "blocker": "promax:final_page_range_missing_in_bib",
            "closed": bibtex_pages_present,
            "current_value": bib_pages,
            "closure_condition": "Add final ACM page range to the ProMax BibTeX entry.",
        },
        {
            "blocker": "promax:crossref_registry_not_visible",
            "closed": direct_crossref_ok,
            "current_value": {
                "status_code": crossref.get("status_code"),
                "summary": summary,
            },
            "closure_condition": "Crossref /works DOI lookup returns 200 with matching DOI metadata.",
        },
        {
            "blocker": "promax:doi_resolver_not_visible",
            "closed": doi_resolver_ok,
            "current_value": {
                "status_code": doi_resolver.get("status_code"),
                "final_url": doi_resolver.get("final_url") or doi_url,
            },
            "closure_condition": "DOI resolver returns a successful response for the expected DOI.",
        },
    ]
    remaining = [item["blocker"] for item in blocker_checks if not item["closed"]]
    warnings: list[str] = []
    if crossref_page_present and not bibtex_pages_present:
        warnings.append("crossref_page_seen_but_bibtex_pages_missing")
    if acm_dl.get("ok") is not True:
        warnings.append(f"acm_dl_not_accessible:status={acm_dl.get('status_code')}")
    for source in source_probes:
        if not source["ok"]:
            warnings.append(f"source_probe_failed:{source['name']}")

    return {
        "schema_version": "2026-06-12.promax_public_metadata_probe.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "local_read_only_promax_public_metadata_probe",
        "aris_skill": "aris-citation-audit",
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "network_mode": network_mode,
        "ok": True,
        "promax_public_metadata_ready": public_metadata_ready,
        "final_submission_ready": False,
        "doi": doi,
        "bibtex": {
            "entry_type": entry.get("entry_type", ""),
            "doi": fields.get("doi", ""),
            "pages": bib_pages,
            "numpages": fields.get("numpages", ""),
            "isbn": fields.get("isbn", ""),
            "location": fields.get("location", ""),
            "eprint": fields.get("eprint", ""),
        },
        "direct_checks": {
            "crossref": _without_text(crossref),
            "crossref_summary": summary,
            "doi_resolver": _without_text(doi_resolver),
            "acm_dl": _without_text(acm_dl),
        },
        "source_probes": source_probes,
        "blocker_checks": blocker_checks,
        "remaining_blockers": remaining,
        "warnings": warnings,
        "next_actions": [
            "If all blocker checks are closed, update the external proceedings metadata audit and release-candidate stack.",
            "Do not set final_submission_ready=true from this probe alone; final readiness still requires the full final submission gate.",
            "If Crossref exposes a page range before BibTeX is updated, copy the final page range into Paper/references.bib and rerun the audits.",
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_md(path: Path, probe: dict[str, Any]) -> None:
    lines = [
        "# ProMax Public Metadata Probe",
        "",
        f"Generated: {probe['created_at_utc']}",
        "",
        f"- OK: `{str(probe['ok']).lower()}`",
        f"- ProMax public metadata ready: `{str(probe['promax_public_metadata_ready']).lower()}`",
        f"- Final submission ready: `{str(probe['final_submission_ready']).lower()}`",
        f"- Network mode: `{probe['network_mode']}`",
        f"- DOI: `{probe['doi']}`",
        "",
        "## Blocker Checks",
        "",
    ]
    for item in probe.get("blocker_checks", []):
        lines.extend(
            [
                f"- `{item['blocker']}`: closed=`{str(item['closed']).lower()}`",
                f"  - closure: {item['closure_condition']}",
            ]
        )
    checks = probe.get("direct_checks") or {}
    lines.extend(
        [
            "",
            "## Direct Checks",
            "",
            f"- Crossref status: `{(checks.get('crossref') or {}).get('status_code')}`",
            f"- DOI resolver status: `{(checks.get('doi_resolver') or {}).get('status_code')}`",
            f"- ACM DL status: `{(checks.get('acm_dl') or {}).get('status_code')}`",
            "",
            "## Source Probes",
            "",
        ]
    )
    for source in probe.get("source_probes", []):
        lines.append(
            f"- `{source['name']}`: ok=`{str(source['ok']).lower()}`, "
            f"status=`{source['status_code']}`, missing_patterns={source['missing_patterns']}"
        )
    lines.extend(["", "## Remaining Blockers", ""])
    blockers = probe.get("remaining_blockers") or []
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    warnings = probe.get("warnings") or []
    lines.extend(f"- `{item}`" for item in warnings) if warnings else lines.append("- None")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in probe.get("next_actions", []))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--bib-path", default=str(DEFAULT_BIB))
    parser.add_argument("--doi", default=DEFAULT_DOI)
    parser.add_argument("--network-mode", choices=["live", "disabled", "fixture"], default="live")
    parser.add_argument("--fixture-json")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    probe = build_promax_public_metadata_probe(
        root=args.root,
        bib_path=args.bib_path,
        doi=args.doi,
        network_mode=args.network_mode,
        fixture_json=args.fixture_json,
        timeout_seconds=args.timeout_seconds,
    )
    if args.output_json:
        _write_json(Path(args.output_json), probe)
    if args.output_md:
        _write_md(Path(args.output_md), probe)
    if not args.output_json:
        print(json.dumps(probe, indent=2, sort_keys=True))
    return 0 if probe["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
