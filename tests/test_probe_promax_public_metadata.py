import json
from pathlib import Path

from scripts.audit.main_probe_promax_public_metadata import (
    _write_md,
    build_promax_public_metadata_probe,
)


DOI = "10.1145/3805712.3809600"
QUERY_CROSSREF = f"https://api.crossref.org/works/{DOI}"
QUERY_DOI = f"https://doi.org/{DOI}"
QUERY_ACM = f"https://dl.acm.org/doi/{DOI}"
ARXIV_HTML = "https://arxiv.org/html/2604.26231v1"
SIGIR_ACCEPTED = "https://sigir2026.org/en-AU/pages/program/accepted-papers"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _bib(tmp_path: Path, *, pages: str = "") -> Path:
    pages_line = f"  pages = {{{pages}}},\n" if pages else ""
    return _write(
        tmp_path / "references.bib",
        "@inproceedings{promax,\n"
        "  title = {{ProMax}: Exploring the Potential of {LLM}-derived Profiles},\n"
        "  author = {Zhang, Yi},\n"
        "  booktitle = {Proceedings of SIGIR},\n"
        "  year = {2026},\n"
        f"{pages_line}"
        "  numpages = {11},\n"
        "  isbn = {979-8-4007-2599-9},\n"
        "  location = {Melbourne, VIC, Australia},\n"
        "  eprint = {2604.26231},\n"
        f"  doi = {{{DOI}}},\n"
        f"  url = {{https://doi.org/{DOI}}}\n"
        "}\n",
    )


def _source_fixtures(*, crossref_ok: bool, doi_ok: bool, page: str = "") -> dict:
    return {
        QUERY_CROSSREF: {
            "ok": crossref_ok,
            "status_code": 200 if crossref_ok else 404,
            "text": json.dumps(
                {
                    "message": {
                        "DOI": DOI,
                        "title": ["ProMax: Exploring the Potential of LLM-derived Profiles"],
                        "container-title": ["Proceedings of SIGIR"],
                        "published-print": {"date-parts": [[2026]]},
                        "page": page,
                    }
                }
            )
            if crossref_ok
            else "",
        },
        QUERY_DOI: {
            "ok": doi_ok,
            "status_code": 200 if doi_ok else 404,
            "final_url": "https://dl.acm.org/doi/10.1145/3805712.3809600" if doi_ok else QUERY_DOI,
            "text": "<html>ProMax</html>" if doi_ok else "",
        },
        QUERY_ACM: {
            "ok": doi_ok,
            "status_code": 200 if doi_ok else 403,
            "text": "<html>ProMax</html>" if doi_ok else "",
        },
        ARXIV_HTML: {
            "ok": True,
            "status_code": 200,
            "text": (
                "10.1145/3805712.3809600 979-8-4007-2599-9/2026/07 "
                "49th International ACM SIGIR Conference Melbourne, VIC, Australia"
            ),
        },
        SIGIR_ACCEPTED: {
            "ok": True,
            "status_code": 200,
            "text": (
                "ProMax: Exploring the Potential of LLM-derived Profiles with "
                "Distribution Shaping for Recommender Systems\n"
                "Yi Zhang, Yiwen Zhang, Kai Zheng, Tong Chen, Hongzhi Yin"
            ),
        },
    }


def test_promax_public_metadata_probe_keeps_current_blockers(tmp_path: Path) -> None:
    bib = _bib(tmp_path)
    fixture = _write_json(
        tmp_path / "fixture.json",
        _source_fixtures(crossref_ok=False, doi_ok=False),
    )

    probe = build_promax_public_metadata_probe(
        root=tmp_path,
        bib_path=bib.relative_to(tmp_path),
        network_mode="fixture",
        fixture_json=fixture.relative_to(tmp_path),
    )

    assert probe["ok"] is True
    assert probe["promax_public_metadata_ready"] is False
    assert probe["final_submission_ready"] is False
    assert "promax:final_page_range_missing_in_bib" in probe["remaining_blockers"]
    assert "promax:crossref_registry_not_visible" in probe["remaining_blockers"]
    assert "promax:doi_resolver_not_visible" in probe["remaining_blockers"]
    assert probe["bibtex"]["isbn"] == "979-8-4007-2599-9"
    assert probe["source_probes"][0]["ok"] is True


def test_promax_public_metadata_probe_ready_requires_direct_gates(tmp_path: Path) -> None:
    bib = _bib(tmp_path, pages="1--11")
    fixture = _write_json(
        tmp_path / "fixture.json",
        _source_fixtures(crossref_ok=True, doi_ok=True, page="1-11"),
    )

    probe = build_promax_public_metadata_probe(
        root=tmp_path,
        bib_path=bib.relative_to(tmp_path),
        network_mode="fixture",
        fixture_json=fixture.relative_to(tmp_path),
    )

    assert probe["promax_public_metadata_ready"] is True
    assert probe["final_submission_ready"] is False
    assert probe["remaining_blockers"] == []
    assert probe["direct_checks"]["crossref_summary"]["page"] == "1-11"


def test_promax_public_metadata_probe_markdown_lists_statuses(tmp_path: Path) -> None:
    bib = _bib(tmp_path)
    fixture = _write_json(
        tmp_path / "fixture.json",
        _source_fixtures(crossref_ok=False, doi_ok=False),
    )
    probe = build_promax_public_metadata_probe(
        root=tmp_path,
        bib_path=bib.relative_to(tmp_path),
        network_mode="fixture",
        fixture_json=fixture.relative_to(tmp_path),
    )
    output = tmp_path / "probe.md"

    _write_md(output, probe)

    text = output.read_text(encoding="utf-8")
    assert "ProMax Public Metadata Probe" in text
    assert "Crossref status: `404`" in text
    assert "DOI resolver status: `404`" in text
    assert "arxiv_html_promax_acm_metadata" in text
