import json
from pathlib import Path

from scripts.audit.main_audit_external_proceedings_metadata import (
    build_external_proceedings_metadata_audit,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _seed_config(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "checks.json",
        {
            "target_profile_id": "test_profile",
            "checks": [
                {
                    "key": "promax",
                    "display_name": "ProMax",
                    "expected_entry_type": "inproceedings",
                    "expected_doi": "10.1145/3805712.3809600",
                    "expected_year": 2026,
                    "expected_arxiv": "2604.26231",
                    "require_pages_in_bib": True,
                    "require_crossref_visible_for_final": True,
                    "require_doi_resolvable_for_final": True,
                    "source_checks": [
                        {
                            "name": "arxiv",
                            "url": "https://arxiv.org/abs/2604.26231",
                            "required_patterns": ["accepted by SIGIR 2026"],
                            "required_for_final": True,
                        }
                    ],
                }
            ],
        },
    )


def _seed_bib(tmp_path: Path, *, doi: str = "10.1145/3805712.3809600", pages: str = "") -> Path:
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
        "  eprint = {2604.26231},\n"
        "  archivePrefix = {arXiv},\n"
        f"  doi = {{{doi}}},\n"
        f"  url = {{https://doi.org/{doi}}}\n"
        "}\n",
    )


def test_external_proceedings_metadata_flags_promax_final_blockers(tmp_path: Path) -> None:
    bib = _seed_bib(tmp_path)
    config = _seed_config(tmp_path)
    fixture = _write_json(
        tmp_path / "fixture.json",
        {
            "https://api.crossref.org/works/10.1145/3805712.3809600": {
                "ok": False,
                "status_code": 404,
                "text": "",
            },
            "https://doi.org/10.1145/3805712.3809600": {
                "ok": False,
                "status_code": 404,
                "text": "",
            },
            "https://arxiv.org/abs/2604.26231": {
                "ok": True,
                "status_code": 200,
                "text": "accepted by SIGIR 2026",
            },
        },
    )

    audit = build_external_proceedings_metadata_audit(
        root=tmp_path,
        bib_path=bib.relative_to(tmp_path),
        config_path=config.relative_to(tmp_path),
        network_mode="fixture",
        fixture_json=fixture.relative_to(tmp_path),
    )

    assert audit["ok"] is True
    assert audit["external_proceedings_metadata_ready"] is False
    assert audit["checked_entries"]["promax"]["external_metadata_ready"] is False
    assert "promax:final_page_range_missing_in_bib" in audit["remaining_blockers"]
    assert "promax:crossref_registry_not_visible:status=404" in audit["remaining_blockers"]
    assert "promax:doi_resolver_not_visible:status=404" in audit["remaining_blockers"]


def test_external_proceedings_metadata_fails_on_doi_mismatch(tmp_path: Path) -> None:
    bib = _seed_bib(tmp_path, doi="10.1145/wrong")
    config = _seed_config(tmp_path)

    audit = build_external_proceedings_metadata_audit(
        root=tmp_path,
        bib_path=bib.relative_to(tmp_path),
        config_path=config.relative_to(tmp_path),
        network_mode="disabled",
    )

    assert audit["ok"] is False
    assert audit["external_proceedings_metadata_ready"] is False
    assert any(failure.startswith("promax:doi_mismatch") for failure in audit["failures"])
