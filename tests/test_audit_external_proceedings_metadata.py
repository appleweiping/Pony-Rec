import json
from pathlib import Path

from scripts.audit.main_audit_external_proceedings_metadata import (
    _write_md,
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
                    "crossref_discovery_queries": [
                        {
                            "name": "crossref_title_search_promax",
                            "query": (
                                "ProMax Exploring the Potential of LLM-derived Profiles "
                                "with Distribution Shaping for Recommender Systems"
                            ),
                            "rows": 5,
                        }
                    ],
                    "source_checks": [
                        {
                            "name": "arxiv",
                            "url": "https://arxiv.org/abs/2604.26231",
                            "required_patterns": ["accepted by SIGIR 2026"],
                            "required_for_final": True,
                        },
                        {
                            "name": "sigir_accepted",
                            "url": "https://sigir2026.org/en-AU/pages/program/accepted-papers",
                            "required_patterns": [
                                "ProMax: Exploring the Potential of LLM-derived Profiles with Distribution Shaping for Recommender Systems",
                                "Yi Zhang, Yiwen Zhang, Kai Zheng, Tong Chen, Hongzhi Yin",
                            ],
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
        "  isbn = {979-8-4007-2599-9},\n"
        "  location = {Melbourne, VIC, Australia},\n"
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
            "https://sigir2026.org/en-AU/pages/program/accepted-papers": {
                "ok": True,
                "status_code": 200,
                "text": (
                    "ProMax: Exploring the Potential of LLM-derived Profiles with "
                    "Distribution Shaping for Recommender Systems\n"
                    "Yi Zhang, Yiwen Zhang, Kai Zheng, Tong Chen, Hongzhi Yin"
                ),
            },
            (
                "https://api.crossref.org/works?query.bibliographic=ProMax%20"
                "Exploring%20the%20Potential%20of%20LLM-derived%20Profiles%20"
                "with%20Distribution%20Shaping%20for%20Recommender%20Systems&rows=5"
            ): {
                "ok": True,
                "status_code": 200,
                "text": json.dumps(
                    {
                        "message": {
                            "items": [
                                {
                                    "DOI": "10.1145/3805712.3809600",
                                    "title": [
                                        "ProMax: Exploring the Potential of LLM-derived Profiles"
                                    ],
                                    "container-title": ["SIGIR"],
                                    "issued": {"date-parts": [[2026]]},
                                    "page": "1-11",
                                    "URL": "https://doi.org/10.1145/3805712.3809600",
                                    "type": "proceedings-article",
                                    "score": 80.0,
                                }
                            ]
                        }
                    }
                ),
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
    discovery = audit["checked_entries"]["promax"]["discovery"]
    assert discovery["enabled"] is True
    assert discovery["exact_doi_candidate_count"] == 1
    assert discovery["exact_doi_with_pages_count"] == 1
    assert "promax:crossref_discovery_expected_doi_seen_but_direct_not_visible" in audit["warnings"]
    sources = {item["name"]: item for item in audit["checked_entries"]["promax"]["source_checks"]}
    assert sources["sigir_accepted"]["ok"] is True
    assert audit["checked_entries"]["promax"]["bibtex"]["isbn"] == "979-8-4007-2599-9"
    assert (
        audit["checked_entries"]["promax"]["bibtex"]["location"]
        == "Melbourne, VIC, Australia"
    )


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


def test_external_proceedings_metadata_can_close_when_direct_final_gates_pass(tmp_path: Path) -> None:
    bib = _seed_bib(tmp_path, pages="1-11")
    config = _seed_config(tmp_path)
    fixture = _write_json(
        tmp_path / "fixture.json",
        {
            "https://api.crossref.org/works/10.1145/3805712.3809600": {
                "ok": True,
                "status_code": 200,
                "text": json.dumps(
                    {
                        "message": {
                            "DOI": "10.1145/3805712.3809600",
                            "title": ["ProMax: Exploring the Potential of LLM-derived Profiles"],
                            "container-title": ["Proceedings of SIGIR"],
                            "published-print": {"date-parts": [[2026]]},
                            "page": "1-11",
                        }
                    }
                ),
            },
            "https://doi.org/10.1145/3805712.3809600": {
                "ok": True,
                "status_code": 200,
                "text": "<html>ProMax</html>",
            },
            "https://arxiv.org/abs/2604.26231": {
                "ok": True,
                "status_code": 200,
                "text": "accepted by SIGIR 2026",
            },
            "https://sigir2026.org/en-AU/pages/program/accepted-papers": {
                "ok": True,
                "status_code": 200,
                "text": (
                    "ProMax: Exploring the Potential of LLM-derived Profiles with "
                    "Distribution Shaping for Recommender Systems\n"
                    "Yi Zhang, Yiwen Zhang, Kai Zheng, Tong Chen, Hongzhi Yin"
                ),
            },
            (
                "https://api.crossref.org/works?query.bibliographic=ProMax%20"
                "Exploring%20the%20Potential%20of%20LLM-derived%20Profiles%20"
                "with%20Distribution%20Shaping%20for%20Recommender%20Systems&rows=5"
            ): {
                "ok": True,
                "status_code": 200,
                "text": json.dumps(
                    {
                        "message": {
                            "items": [
                                {
                                    "DOI": "10.1145/3805712.3809600",
                                    "title": [
                                        "ProMax: Exploring the Potential of LLM-derived Profiles"
                                    ],
                                    "container-title": ["SIGIR"],
                                    "issued": {"date-parts": [[2026]]},
                                    "page": "1-11",
                                    "URL": "https://doi.org/10.1145/3805712.3809600",
                                    "type": "proceedings-article",
                                    "score": 80.0,
                                }
                            ]
                        }
                    }
                ),
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
    assert audit["external_proceedings_metadata_ready"] is True
    assert audit["final_submission_ready"] is False
    assert audit["checked_entries"]["promax"]["external_metadata_ready"] is True
    assert audit["checked_entries"]["promax"]["discovery"]["exact_doi_with_pages_count"] == 1
    assert audit["remaining_blockers"] == [
        "Final manual submission-system metadata/format checklist is not closed."
    ]


def test_external_proceedings_metadata_reports_alternate_discovery_doi(tmp_path: Path) -> None:
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
            "https://sigir2026.org/en-AU/pages/program/accepted-papers": {
                "ok": True,
                "status_code": 200,
                "text": (
                    "ProMax: Exploring the Potential of LLM-derived Profiles with "
                    "Distribution Shaping for Recommender Systems\n"
                    "Yi Zhang, Yiwen Zhang, Kai Zheng, Tong Chen, Hongzhi Yin"
                ),
            },
            (
                "https://api.crossref.org/works?query.bibliographic=ProMax%20"
                "Exploring%20the%20Potential%20of%20LLM-derived%20Profiles%20"
                "with%20Distribution%20Shaping%20for%20Recommender%20Systems&rows=5"
            ): {
                "ok": True,
                "status_code": 200,
                "text": json.dumps(
                    {
                        "message": {
                            "items": [
                                {
                                    "DOI": "10.1145/alternate",
                                    "title": [
                                        "ProMax: Exploring the Potential of LLM-derived Profiles"
                                    ],
                                    "container-title": ["SIGIR"],
                                    "issued": {"date-parts": [[2026]]},
                                    "page": "1-11",
                                    "URL": "https://doi.org/10.1145/alternate",
                                    "type": "proceedings-article",
                                    "score": 70.0,
                                }
                            ]
                        }
                    }
                ),
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

    discovery = audit["checked_entries"]["promax"]["discovery"]
    assert audit["external_proceedings_metadata_ready"] is False
    assert discovery["alternate_doi_candidate_count"] == 1
    assert "promax:crossref_discovery_alternate_doi_candidates_present" in audit["warnings"]
    assert audit["checked_entries"]["promax"]["bibtex"]["doi"] == "10.1145/3805712.3809600"
    assert "promax:crossref_registry_not_visible:status=404" in audit["remaining_blockers"]


def test_external_proceedings_metadata_handles_malformed_discovery_json(tmp_path: Path) -> None:
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
            "https://sigir2026.org/en-AU/pages/program/accepted-papers": {
                "ok": True,
                "status_code": 200,
                "text": (
                    "ProMax: Exploring the Potential of LLM-derived Profiles with "
                    "Distribution Shaping for Recommender Systems\n"
                    "Yi Zhang, Yiwen Zhang, Kai Zheng, Tong Chen, Hongzhi Yin"
                ),
            },
            (
                "https://api.crossref.org/works?query.bibliographic=ProMax%20"
                "Exploring%20the%20Potential%20of%20LLM-derived%20Profiles%20"
                "with%20Distribution%20Shaping%20for%20Recommender%20Systems&rows=5"
            ): {
                "ok": True,
                "status_code": 200,
                "text": "{not json",
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

    discovery_report = audit["checked_entries"]["promax"]["discovery"]["reports"][0]
    assert audit["external_proceedings_metadata_ready"] is False
    assert discovery_report["ok"] is False
    assert discovery_report["error"] == "crossref_discovery_json_decode_failed"
    assert "promax:crossref_registry_not_visible:status=404" in audit["remaining_blockers"]


def test_external_proceedings_metadata_handles_empty_discovery_query(tmp_path: Path) -> None:
    bib = _seed_bib(tmp_path)
    config = _seed_config(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["checks"][0]["crossref_discovery_queries"].append(
        {"name": "empty_query", "query": "", "rows": 5}
    )
    config.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    audit = build_external_proceedings_metadata_audit(
        root=tmp_path,
        bib_path=bib.relative_to(tmp_path),
        config_path=config.relative_to(tmp_path),
        network_mode="disabled",
    )

    reports = audit["checked_entries"]["promax"]["discovery"]["reports"]
    empty = next(item for item in reports if item["name"] == "empty_query")
    assert empty["ok"] is False
    assert empty["error"] == "empty_query"
    assert audit["external_proceedings_metadata_ready"] is False


def test_external_proceedings_metadata_markdown_includes_discovery_summary(tmp_path: Path) -> None:
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
            "https://sigir2026.org/en-AU/pages/program/accepted-papers": {
                "ok": True,
                "status_code": 200,
                "text": (
                    "ProMax: Exploring the Potential of LLM-derived Profiles with "
                    "Distribution Shaping for Recommender Systems\n"
                    "Yi Zhang, Yiwen Zhang, Kai Zheng, Tong Chen, Hongzhi Yin"
                ),
            },
            (
                "https://api.crossref.org/works?query.bibliographic=ProMax%20"
                "Exploring%20the%20Potential%20of%20LLM-derived%20Profiles%20"
                "with%20Distribution%20Shaping%20for%20Recommender%20Systems&rows=5"
            ): {
                "ok": True,
                "status_code": 200,
                "text": json.dumps({"message": {"items": []}}),
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
    output_md = tmp_path / "audit.md"

    _write_md(output_md, audit)

    text = output_md.read_text(encoding="utf-8")
    assert "Crossref discovery candidates" in text
    assert "Discovery policy" in text
    assert "ISBN: `979-8-4007-2599-9`" in text
    assert "Location: `Melbourne, VIC, Australia`" in text
