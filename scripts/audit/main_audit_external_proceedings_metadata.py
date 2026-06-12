from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_CONFIG = Path("configs/paper_external_proceedings_metadata_checks.json")
DEFAULT_BIB = Path("Paper/references.bib")
DEFAULT_TIMEOUT_SECONDS = 20
USER_AGENT = "UncertaintyProceedingsAudit/1.0"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _strip_outer(value: str) -> str:
    value = value.strip().rstrip(",").strip()
    changed = True
    while changed and len(value) >= 2:
        changed = False
        if (value[0], value[-1]) in {("{", "}"), ('"', '"')}:
            value = value[1:-1].strip()
            changed = True
    return value


def _split_top_level_fields(body: str) -> list[str]:
    fields: list[str] = []
    start = 0
    depth = 0
    in_quote = False
    escape = False
    for idx, char in enumerate(body):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"' and depth == 0:
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            item = body[start:idx].strip()
            if item:
                fields.append(item)
            start = idx + 1
    tail = body[start:].strip()
    if tail:
        fields.append(tail)
    return fields


def _parse_bib_entries(text: str) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    idx = 0
    while True:
        at = text.find("@", idx)
        if at == -1:
            break
        match = re.match(r"@([A-Za-z]+)\s*\{", text[at:])
        if not match:
            idx = at + 1
            continue
        entry_type = match.group(1).lower()
        open_idx = at + match.end() - 1
        key_end = text.find(",", open_idx + 1)
        if key_end == -1:
            break
        key = text[open_idx + 1 : key_end].strip()
        depth = 1
        pos = key_end + 1
        while pos < len(text) and depth:
            char = text[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            pos += 1
        body = text[key_end + 1 : pos - 1]
        fields: dict[str, str] = {}
        for raw_field in _split_top_level_fields(body):
            if "=" not in raw_field:
                continue
            name, raw_value = raw_field.split("=", 1)
            fields[name.strip().lower()] = _strip_outer(raw_value)
        entries[key] = {"entry_type": entry_type, "fields": fields}
        idx = pos
    return entries


def _fetch_url(url: str, *, timeout_seconds: int) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read(500_000)
            text = raw.decode("utf-8", errors="replace")
            return {
                "url": url,
                "ok": 200 <= int(response.status) < 400,
                "status_code": int(response.status),
                "final_url": response.geturl(),
                "error": "",
                "content_length_sampled": len(text),
                "text": text,
            }
    except HTTPError as exc:
        raw = exc.read(20_000)
        return {
            "url": url,
            "ok": False,
            "status_code": int(exc.code),
            "final_url": url,
            "error": str(exc),
            "content_length_sampled": len(raw),
            "text": raw.decode("utf-8", errors="replace"),
        }
    except URLError as exc:
        return {
            "url": url,
            "ok": False,
            "status_code": None,
            "final_url": url,
            "error": str(exc.reason),
            "content_length_sampled": 0,
            "text": "",
        }


def _network_check(
    url: str,
    *,
    network_mode: str,
    timeout_seconds: int,
    fixtures: dict[str, Any],
) -> dict[str, Any]:
    if network_mode == "disabled":
        return {
            "url": url,
            "ok": False,
            "status_code": None,
            "final_url": url,
            "error": "network_disabled",
            "content_length_sampled": 0,
            "text": "",
        }
    if network_mode == "fixture":
        fixture = dict(fixtures.get(url) or {})
        return {
            "url": url,
            "ok": bool(fixture.get("ok")),
            "status_code": fixture.get("status_code"),
            "final_url": fixture.get("final_url") or url,
            "error": fixture.get("error") or "",
            "content_length_sampled": len(str(fixture.get("text") or "")),
            "text": str(fixture.get("text") or ""),
        }
    return _fetch_url(url, timeout_seconds=timeout_seconds)


def _public_network_state(
    doi: str,
    *,
    network_mode: str,
    timeout_seconds: int,
    fixtures: dict[str, Any],
) -> dict[str, Any]:
    encoded_doi = quote(doi, safe="/.")
    crossref_url = f"https://api.crossref.org/works/{encoded_doi}"
    doi_url = f"https://doi.org/{doi}"
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

    crossref_summary: dict[str, Any] = {}
    if crossref["ok"] and crossref.get("text"):
        try:
            payload = json.loads(crossref["text"])
            message = payload.get("message") or {}
            crossref_summary = {
                "title": (message.get("title") or [""])[0],
                "container_title": (message.get("container-title") or [""])[0],
                "published_year": (((message.get("published-print") or {}).get("date-parts") or [[None]])[0][0]),
                "page": message.get("page") or "",
                "doi": message.get("DOI") or "",
            }
        except json.JSONDecodeError:
            crossref_summary = {"parse_error": "crossref_json_decode_failed"}
    return {
        "crossref": _without_text(crossref),
        "crossref_summary": crossref_summary,
        "doi_resolver": _without_text(doi_resolver),
    }


def _without_text(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "text"}


def _first_text(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0] or "")
    return str(value or "")


def _issued_year(item: dict[str, Any]) -> Any:
    for field in ("published-print", "published-online", "issued"):
        parts = ((item.get(field) or {}).get("date-parts") or [[None]])
        if parts and parts[0]:
            return parts[0][0]
    return None


def _crossref_discovery_url(query: str, rows: int) -> str:
    encoded_query = quote(query, safe="")
    return f"https://api.crossref.org/works?query.bibliographic={encoded_query}&rows={rows}"


def _crossref_title_discovery(
    check: dict[str, Any],
    *,
    network_mode: str,
    timeout_seconds: int,
    fixtures: dict[str, Any],
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    expected_doi = str(check.get("expected_doi") or "").strip().lower()
    expected_year = str(check.get("expected_year") or "").strip()
    for query_def in check.get("crossref_discovery_queries") or []:
        query = str(query_def.get("query") or "").strip()
        if not query:
            reports.append(
                {
                    "name": query_def.get("name") or "unnamed_discovery_query",
                    "query": "",
                    "url": "",
                    "ok": False,
                    "status_code": None,
                    "error": "empty_query",
                    "candidate_count": 0,
                    "candidates": [],
                }
            )
            continue
        rows = int(query_def.get("rows") or 5)
        rows = max(1, min(rows, 20))
        url = _crossref_discovery_url(query, rows)
        fetched = _network_check(
            url,
            network_mode=network_mode,
            timeout_seconds=timeout_seconds,
            fixtures=fixtures,
        )
        candidates: list[dict[str, Any]] = []
        parse_error = ""
        if fetched.get("text"):
            try:
                payload = json.loads(str(fetched.get("text") or ""))
                items = (payload.get("message") or {}).get("items") or []
                for item in items[:rows]:
                    doi = str(item.get("DOI") or "")
                    year = _issued_year(item)
                    page = str(item.get("page") or "")
                    candidates.append(
                        {
                            "doi": doi,
                            "title": _first_text(item.get("title")),
                            "container_title": _first_text(item.get("container-title")),
                            "year": year,
                            "page": page,
                            "url": item.get("URL") or "",
                            "type": item.get("type") or "",
                            "score": item.get("score"),
                            "expected_doi_match": bool(expected_doi)
                            and doi.strip().lower() == expected_doi,
                            "expected_year_match": bool(expected_year) and str(year or "") == expected_year,
                            "has_page_range": bool(page.strip()),
                        }
                    )
            except json.JSONDecodeError:
                parse_error = "crossref_discovery_json_decode_failed"
        reports.append(
            {
                "name": query_def.get("name") or "crossref_discovery_query",
                "query": query,
                "url": url,
                "ok": bool(fetched.get("ok")) and not parse_error,
                "status_code": fetched.get("status_code"),
                "final_url": fetched.get("final_url"),
                "error": parse_error or fetched.get("error") or "",
                "candidate_count": len(candidates),
                "candidates": candidates,
            }
        )

    all_candidates = [candidate for report in reports for candidate in report.get("candidates", [])]
    exact_doi_candidates = [candidate for candidate in all_candidates if candidate.get("expected_doi_match")]
    exact_doi_with_pages = [
        candidate for candidate in exact_doi_candidates if candidate.get("has_page_range")
    ]
    alternate_doi_candidates = [
        candidate
        for candidate in all_candidates
        if expected_doi and candidate.get("doi") and not candidate.get("expected_doi_match")
    ]
    return {
        "enabled": bool(check.get("crossref_discovery_queries")),
        "reports": reports,
        "candidate_count": len(all_candidates),
        "exact_doi_candidate_count": len(exact_doi_candidates),
        "exact_doi_with_pages_count": len(exact_doi_with_pages),
        "alternate_doi_candidate_count": len(alternate_doi_candidates),
        "alternate_doi_candidates": alternate_doi_candidates[:5],
        "policy": (
            "Discovery candidates are advisory only. They help detect newly public "
            "or changed metadata, but they do not by themselves satisfy the exact "
            "BibTeX page-range, DOI resolver, or Crossref final-readiness gates."
        ),
    }


def _has_expected(value: str | None, expected: Any) -> bool:
    if expected is None:
        return True
    return str(value or "").strip().lower() == str(expected).strip().lower()


def _entry_report(
    check: dict[str, Any],
    entry: dict[str, Any] | None,
    *,
    network_mode: str,
    timeout_seconds: int,
    fixtures: dict[str, Any],
) -> dict[str, Any]:
    key = str(check["key"])
    fields = (entry or {}).get("fields") or {}
    failures: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []

    if entry is None:
        failures.append(f"{key}:missing_bib_entry")
        return {
            "key": key,
            "display_name": check.get("display_name") or key,
            "ok": False,
            "external_metadata_ready": False,
            "bibtex": {"exists": False, "fields": {}},
            "network": {},
            "source_checks": [],
            "warnings": warnings,
            "failures": failures,
            "blockers": blockers,
        }

    expected_type = check.get("expected_entry_type")
    if expected_type and entry["entry_type"] != expected_type:
        failures.append(f"{key}:entry_type_mismatch:{entry['entry_type']} != {expected_type}")
    if not _has_expected(fields.get("doi"), check.get("expected_doi")):
        failures.append(f"{key}:doi_mismatch:{fields.get('doi')} != {check.get('expected_doi')}")
    if not _has_expected(fields.get("year"), check.get("expected_year")):
        failures.append(f"{key}:year_mismatch:{fields.get('year')} != {check.get('expected_year')}")
    expected_arxiv = check.get("expected_arxiv")
    if expected_arxiv and not _has_expected(fields.get("eprint"), expected_arxiv):
        failures.append(f"{key}:arxiv_mismatch:{fields.get('eprint')} != {expected_arxiv}")

    has_page_range = bool(str(fields.get("pages") or "").strip())
    if check.get("require_pages_in_bib") and not has_page_range:
        blockers.append(f"{key}:final_page_range_missing_in_bib")

    doi = str(check.get("expected_doi") or fields.get("doi") or "").strip()
    network = _public_network_state(
        doi,
        network_mode=network_mode,
        timeout_seconds=timeout_seconds,
        fixtures=fixtures,
    ) if doi else {}

    crossref_ok = bool((network.get("crossref") or {}).get("ok"))
    if check.get("require_crossref_visible_for_final") and not crossref_ok:
        status = (network.get("crossref") or {}).get("status_code")
        blockers.append(f"{key}:crossref_registry_not_visible:status={status}")
    elif doi and not crossref_ok:
        status = (network.get("crossref") or {}).get("status_code")
        warnings.append(f"{key}:crossref_not_visible:status={status}")

    doi_ok = bool((network.get("doi_resolver") or {}).get("ok"))
    if check.get("require_doi_resolvable_for_final") and not doi_ok:
        status = (network.get("doi_resolver") or {}).get("status_code")
        blockers.append(f"{key}:doi_resolver_not_visible:status={status}")
    elif doi and not doi_ok:
        status = (network.get("doi_resolver") or {}).get("status_code")
        warnings.append(f"{key}:doi_resolver_not_visible:status={status}")

    source_reports = []
    for source in check.get("source_checks") or []:
        url = str(source.get("url") or "")
        fetched = _network_check(
            url,
            network_mode=network_mode,
            timeout_seconds=timeout_seconds,
            fixtures=fixtures,
        )
        text_lower = str(fetched.get("text") or "").lower()
        missing_patterns = [
            pattern
            for pattern in source.get("required_patterns") or []
            if str(pattern).lower() not in text_lower
        ]
        source_ok = bool(fetched.get("ok")) and not missing_patterns
        source_reports.append(
            {
                "name": source.get("name"),
                "url": url,
                "required_for_final": bool(source.get("required_for_final")),
                "ok": source_ok,
                "status_code": fetched.get("status_code"),
                "final_url": fetched.get("final_url"),
                "missing_patterns": missing_patterns,
                "error": fetched.get("error") or "",
            }
        )
        if source.get("required_for_final") and not source_ok:
            blockers.append(f"{key}:required_source_check_failed:{source.get('name')}")
        elif not source_ok:
            warnings.append(f"{key}:source_check_not_visible:{source.get('name')}")

    discovery = _crossref_title_discovery(
        check,
        network_mode=network_mode,
        timeout_seconds=timeout_seconds,
        fixtures=fixtures,
    )
    if discovery["enabled"]:
        if discovery["candidate_count"] == 0:
            warnings.append(f"{key}:crossref_discovery_no_candidates")
        if discovery["alternate_doi_candidate_count"] > 0:
            warnings.append(f"{key}:crossref_discovery_alternate_doi_candidates_present")
        if discovery["exact_doi_candidate_count"] > 0 and not crossref_ok:
            warnings.append(f"{key}:crossref_discovery_expected_doi_seen_but_direct_not_visible")

    ready = not failures and not blockers
    return {
        "key": key,
        "display_name": check.get("display_name") or key,
        "ok": not failures,
        "external_metadata_ready": ready,
        "bibtex": {
            "exists": True,
            "entry_type": entry["entry_type"],
            "title": fields.get("title", ""),
            "booktitle": fields.get("booktitle", ""),
            "year": fields.get("year", ""),
            "pages": fields.get("pages", ""),
            "numpages": fields.get("numpages", ""),
            "doi": fields.get("doi", ""),
            "url": fields.get("url", ""),
            "eprint": fields.get("eprint", ""),
        },
        "network": network,
        "discovery": discovery,
        "source_checks": source_reports,
        "warnings": warnings,
        "failures": failures,
        "blockers": blockers,
        "notes": check.get("notes") or [],
    }


def build_external_proceedings_metadata_audit(
    *,
    root: str | Path = ".",
    bib_path: str | Path = DEFAULT_BIB,
    config_path: str | Path = DEFAULT_CONFIG,
    network_mode: str = "live",
    fixture_json: str | Path | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    bib = repo / bib_path
    config = _read_json(repo / config_path)
    fixtures = _read_json(repo / fixture_json) if fixture_json else {}
    entries = _parse_bib_entries(bib.read_text(encoding="utf-8", errors="replace"))

    reports = [
        _entry_report(
            check,
            entries.get(str(check["key"])),
            network_mode=network_mode,
            timeout_seconds=timeout_seconds,
            fixtures=fixtures,
        )
        for check in config.get("checks") or []
    ]
    failures = [failure for report in reports for failure in report.get("failures", [])]
    warnings = [warning for report in reports for warning in report.get("warnings", [])]
    blockers = [blocker for report in reports for blocker in report.get("blockers", [])]
    ready = not failures and not blockers

    return {
        "schema_version": "2026-06-12.external_proceedings_metadata_audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "local_read_only_external_proceedings_metadata_audit",
        "aris_skill": "aris-citation-audit",
        "read_only": True,
        "will_ssh": False,
        "will_copy": False,
        "will_delete": False,
        "will_start_experiment": False,
        "network_mode": network_mode,
        "ok": not failures,
        "external_proceedings_metadata_ready": ready,
        "final_submission_ready": False,
        "target_profile_id": config.get("target_profile_id"),
        "input_paths": {
            "bib_path": str(Path(bib_path)),
            "config_path": str(Path(config_path)),
            "fixture_json": str(Path(fixture_json)) if fixture_json else "",
        },
        "checked_entry_count": len(reports),
        "checked_entries": {report["key"]: report for report in reports},
        "failures": failures,
        "warnings": warnings,
        "remaining_blockers": blockers
        + ["Final manual submission-system metadata/format checklist is not closed."],
        "next_actions": [
            "If ProMax DOI/Crossref/page-range visibility becomes public, update Paper/references.bib and rerun this audit.",
            "Rerun the submission package and metadata packet audits after this external metadata audit changes.",
            "Keep final_submission_ready=false until this audit, target formatting, and manual submission-system checks are all closed.",
        ],
    }


def _write_md(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# External Proceedings Metadata Audit",
        "",
        f"Generated: {audit['created_at_utc']}",
        "",
        f"- OK: `{str(audit['ok']).lower()}`",
        "- External proceedings metadata ready: "
        f"`{str(audit['external_proceedings_metadata_ready']).lower()}`",
        f"- Final submission ready: `{str(audit['final_submission_ready']).lower()}`",
        f"- Network mode: `{audit['network_mode']}`",
        f"- Checked entries: `{audit['checked_entry_count']}`",
        "",
        "## Entry Checks",
        "",
    ]
    for key, report in audit["checked_entries"].items():
        bib = report["bibtex"]
        network = report.get("network") or {}
        crossref = network.get("crossref") or {}
        doi_resolver = network.get("doi_resolver") or {}
        lines.extend(
            [
                f"### {report['display_name']} (`{key}`)",
                "",
                f"- OK: `{str(report['ok']).lower()}`",
                f"- External metadata ready: `{str(report['external_metadata_ready']).lower()}`",
                f"- DOI: `{bib.get('doi', '')}`",
                f"- Pages: `{bib.get('pages', '')}`",
                f"- Num pages: `{bib.get('numpages', '')}`",
                f"- arXiv: `{bib.get('eprint', '')}`",
                f"- Crossref: ok=`{str(crossref.get('ok')).lower()}`, status=`{crossref.get('status_code')}`",
                f"- DOI resolver: ok=`{str(doi_resolver.get('ok')).lower()}`, status=`{doi_resolver.get('status_code')}`",
            ]
        )
        for source in report.get("source_checks", []):
            lines.append(
                f"- Source `{source['name']}`: ok=`{str(source['ok']).lower()}`, "
                f"status=`{source['status_code']}`, missing_patterns={source['missing_patterns']}"
            )
        discovery = report.get("discovery") or {}
        if discovery.get("enabled"):
            lines.extend(
                [
                    f"- Crossref discovery candidates: `{discovery.get('candidate_count', 0)}`",
                    f"- Discovery exact-DOI candidates: `{discovery.get('exact_doi_candidate_count', 0)}`",
                    f"- Discovery exact-DOI candidates with pages: `{discovery.get('exact_doi_with_pages_count', 0)}`",
                    f"- Discovery alternate-DOI candidates: `{discovery.get('alternate_doi_candidate_count', 0)}`",
                    f"- Discovery policy: {discovery.get('policy', '')}",
                ]
            )
            for query in discovery.get("reports", []):
                lines.append(
                    f"- Discovery query `{query.get('name')}`: ok=`{str(query.get('ok')).lower()}`, "
                    f"status=`{query.get('status_code')}`, candidates=`{query.get('candidate_count')}`, "
                    f"error=`{query.get('error') or ''}`"
                )
                for candidate in (query.get("candidates") or [])[:3]:
                    lines.append(
                        "  - candidate "
                        f"doi=`{candidate.get('doi', '')}`, "
                        f"year=`{candidate.get('year', '')}`, "
                        f"pages=`{candidate.get('page', '')}`, "
                        f"expected_doi_match=`{str(candidate.get('expected_doi_match')).lower()}`, "
                        f"title=`{candidate.get('title', '')}`"
                    )
        lines.extend(["", "Blockers:"])
        blockers = report.get("blockers") or []
        lines.extend(f"- `{item}`" for item in blockers) if blockers else lines.append("- None")
        lines.extend(["", "Warnings:"])
        warnings = report.get("warnings") or []
        lines.extend(f"- `{item}`" for item in warnings) if warnings else lines.append("- None")
        lines.append("")

    lines.extend(["## Remaining Blockers", ""])
    blockers = audit.get("remaining_blockers") or []
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- None")
    lines.extend(["", "## Failures", ""])
    failures = audit.get("failures") or []
    lines.extend(f"- `{item}`" for item in failures) if failures else lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    warnings = audit.get("warnings") or []
    lines.extend(f"- `{item}`" for item in warnings) if warnings else lines.append("- None")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in audit.get("next_actions", []))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--bib-path", default=str(DEFAULT_BIB))
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--network-mode",
        choices=["live", "disabled", "fixture"],
        default="live",
        help="Use live HTTP checks, disable network, or read URL responses from --fixture-json.",
    )
    parser.add_argument("--fixture-json")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_external_proceedings_metadata_audit(
        root=args.root,
        bib_path=args.bib_path,
        config_path=args.config_path,
        network_mode=args.network_mode,
        fixture_json=args.fixture_json,
        timeout_seconds=args.timeout_seconds,
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
