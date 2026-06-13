import json
import subprocess
from pathlib import Path

from scripts.audit.main_refresh_pre_submission_gates import refresh_pre_submission_gates


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _fake_latex_runner(command, cwd, **kwargs):
    cwd = Path(cwd)
    if command[0] == "pdflatex":
        (cwd / "main.log").write_text(
            "Output written on main.pdf (9 pages, 120000 bytes).\n",
            encoding="utf-8",
        )
        (cwd / "main.pdf").write_bytes((b"/Type /Page\n" * 9) + (b"p" * 119_892))
    if command[0] == "bibtex":
        (cwd / "main.blg").write_text("warning$ -- 0\n", encoding="utf-8")
    return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")


def _seed_package_inputs(tmp_path: Path) -> dict[str, Path]:
    paper = tmp_path / "Paper"
    _write(
        paper / "main.tex",
        r"""
\documentclass[sigconf,nonacm]{acmart}
\begin{document}
\title{Actionable Uncertainty for LLM-Based Recommendation}
\author{Anonymous}
\affiliation{\institution{Anonymous Institution}\city{Anonymous}\country{Anonymous}}
\begin{abstract}
""" + " ".join(f"word{i}" for i in range(130)) + r"""
\end{abstract}
\maketitle
\input{sections/intro}
\includegraphics{figures/framework_overview.pdf}
\bibliography{references}
\end{document}
""".strip()
        + "\n",
    )
    _write(paper / "sections" / "intro.tex", "\\section{Intro}\nScoped claim.\n")
    _write(paper / "sections" / "abstract.tex", "\\begin{abstract}\n" + " ".join(f"word{i}" for i in range(130)) + "\n\\end{abstract}\n")
    _write(
        paper / "references.bib",
        "@inproceedings{promax,\n"
        "  title = {{ProMax}},\n"
        "  year = {2026},\n"
        "  eprint = {2604.26231},\n"
        "  doi = {10.1145/3805712.3809600}\n"
        "}\n",
    )
    _write(paper / "main.pdf", "x" * 120_000)
    _write(paper / "main.log", "Output written on main.pdf (9 pages, 120000 bytes).\n")
    _write(paper / "main.blg", "warning$ -- 0\n")
    _write(paper / "main.aux", "\\citation{promax}\n")
    _write(paper / "main.bbl", "\\begin{thebibliography}{1}\\bibitem{promax} X.\\end{thebibliography}\n")
    _write(paper / "figures" / "framework_overview.pdf", "pdf\n")
    _write(paper / "figures" / "framework_overview.svg", "<svg />\n")
    claim = _write_json(
        tmp_path / "claim.json",
        {"ok": True, "paper_evidence_ready_for_drafting": True, "final_submission_ready": False},
    )
    panel = _write_json(
        tmp_path / "panel.json",
        {
            "reviewer_consensus": {
                "score_floor": "8.0/10",
                "new_experiment_required": False,
                "claim_boundary_ok": True,
                "final_submission_ready": False,
            }
        },
    )
    followup = _write_json(
        tmp_path / "metadata_followup.json",
        {
            "verdict": "PDF_PACKAGE_POLISH_PASS_METADATA_CAUTION",
            "final_submission_ready": False,
            "remaining_blockers": ["external metadata"],
        },
    )
    profile = _write_json(
        tmp_path / "profiles.json",
        {
            "default_profile_id": "test_profile",
            "profiles": {
                "test_profile": {
                    "documentclass": "acmart",
                    "required_documentclass_options": ["sigconf"],
                    "max_total_pages": 9,
                    "require_anonymous_author_shell": True,
                    "require_single_pdf": True,
                    "require_local_source_manifest": True,
                    "require_bibtex_clean": True,
                    "require_no_undefined_references": True,
                    "require_no_overfull_hbox": True,
                }
            },
        },
    )
    metadata_config = _write_json(
        tmp_path / "metadata_config.json",
        {
            "target_profile_id": "test_profile",
            "paper_type": "full research paper",
            "submission_title": "Actionable Uncertainty for LLM-Based Recommendation",
            "anonymous_submission": True,
            "suggested_keywords": ["recommendation", "uncertainty", "calibration"],
            "suggested_topic_areas": ["Recommender systems", "Evaluation"],
            "manual_fields_not_stored": ["authors"],
        },
    )
    external_config = _write_json(
        tmp_path / "external_config.json",
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
                    "source_checks": [],
                }
            ],
        },
    )
    manual_config = _write_json(
        tmp_path / "manual_config.json",
        {
            "target_profile_id": "test_profile",
            "checklist_id": "test_manual_checklist",
            "manual_private_fields_not_stored": ["authors"],
            "items": [
                {
                    "id": "paste_title",
                    "category": "metadata",
                    "label": "Paste title",
                    "storage_policy": "repo_prefill_public_value",
                    "private": False,
                    "requires_submission_system": True,
                    "prefill_source": "submission_fields.title",
                },
                {
                    "id": "enter_authors",
                    "category": "private_metadata",
                    "label": "Enter authors",
                    "storage_policy": "not_stored_in_repo",
                    "private": True,
                    "requires_submission_system": True,
                },
                {
                    "id": "confirm_external_proceedings_metadata",
                    "category": "external_metadata",
                    "label": "Confirm external",
                    "storage_policy": "repo_external_audit_plus_manual_confirmation",
                    "private": False,
                    "requires_submission_system": False,
                    "external_gate": "external_proceedings_metadata_ready",
                },
            ],
        },
    )
    review_packet = _write_json(
        tmp_path / "review_continuation.json",
        {
            "schema_version": "review.v1",
            "created_at_utc": "2026-06-13T00:00:00Z",
            "ok": True,
            "review_continuation_ready": True,
            "final_panel_coverage_complete": False,
            "final_submission_ready": False,
            "warnings": [],
            "failures": [],
            "remaining_blockers": ["explicit_claude_opus_review"],
        },
    )
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
        },
    )
    return {
        "paper": paper,
        "claim": claim,
        "panel": panel,
        "followup": followup,
        "profile": profile,
        "metadata_config": metadata_config,
        "external_config": external_config,
        "manual_config": manual_config,
        "review_packet": review_packet,
        "fixture": fixture,
    }


def test_refresh_pre_submission_gates_runs_in_dependency_order(tmp_path: Path) -> None:
    paths = _seed_package_inputs(tmp_path)

    refresh = refresh_pre_submission_gates(
        root=tmp_path,
        output_dir="out",
        stamp="test",
        paper_dir=paths["paper"].relative_to(tmp_path),
        external_config_path=paths["external_config"].relative_to(tmp_path),
        external_network_mode="fixture",
        external_fixture_json=paths["fixture"].relative_to(tmp_path),
        claim_audit_json=paths["claim"].relative_to(tmp_path),
        panel_review_json=paths["panel"].relative_to(tmp_path),
        metadata_followup_json=paths["followup"].relative_to(tmp_path),
        target_profile_json=paths["profile"].relative_to(tmp_path),
        target_profile_id="test_profile",
        metadata_config=paths["metadata_config"].relative_to(tmp_path),
        manual_config=paths["manual_config"].relative_to(tmp_path),
        review_continuation_packet_json=paths["review_packet"].relative_to(tmp_path),
        source_package_output_dir="artifacts/source_package",
        source_rebuild_build_dir="artifacts/source_rebuild",
        source_rebuild_runner=_fake_latex_runner,
    )

    assert refresh["ok"] is True
    assert refresh["final_submission_ready"] is False
    assert refresh["final_verdict"] == "LOCAL_PACKAGE_READY_BUT_EXTERNAL_MANUAL_OR_REVIEW_BLOCKED"
    assert "git_state_before_refresh" in refresh
    assert refresh["input_fingerprints"]
    assert [step["step_id"] for step in refresh["steps"]] == [
        "external_proceedings_metadata",
        "submission_package",
        "submission_source_package",
        "submission_source_package_rebuild",
        "submission_metadata_packet",
        "manual_submission_checklist",
        "final_submission_gate",
    ]
    for name in [
        "external_proceedings_metadata_recheck_test.json",
        "submission_package_audit_test.json",
        "submission_source_package_test.json",
        "submission_source_package_rebuild_test.json",
        "submission_metadata_packet_test.json",
        "manual_submission_checklist_test.json",
        "final_submission_gate_test.json",
    ]:
        assert (tmp_path / "out" / name).exists()
    assert all(step["json"]["sha256"] for step in refresh["steps"])
    input_paths = {item["path"].replace("\\", "/") for item in refresh["input_fingerprints"]}
    assert "Paper/main.tex" in input_paths
    assert "Paper/references.bib" in input_paths
    assert "scripts/audit/main_audit_pre_submission_refresh_freshness.py" in input_paths
    assert "scripts/audit/main_build_submission_source_package.py" in input_paths
    assert "scripts/audit/main_audit_submission_source_package_rebuild.py" in input_paths
    assert "promax:final_page_range_missing_in_bib" in refresh["remaining_blockers"]
    assert "manual_submission_system_items_not_confirmed" in refresh["remaining_blockers"]
    assert "explicit_claude_opus_review" in refresh["remaining_blockers"]
