import json
from pathlib import Path

from scripts.audit.main_audit_submission_package import build_submission_package_audit


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _seed_package(tmp_path: Path) -> dict[str, Path]:
    paper = tmp_path / "Paper"
    _write(
        paper / "main.tex",
        r"""
\documentclass[sigconf,nonacm]{acmart}
\begin{document}
\title{Actionable Uncertainty for LLM-Based Recommendation}
\author{Anonymous}
\affiliation{\institution{Anonymous Institution}\city{Anonymous}\country{Anonymous}}
\maketitle
\input{sections/intro}
\includegraphics{figures/framework_overview.pdf}
\bibliography{references}
\end{document}
""".strip()
        + "\n",
    )
    _write(paper / "sections" / "intro.tex", "\\section{Intro}\nScoped claim.\n")
    _write(paper / "references.bib", "@inproceedings{x, title={X}, year={2026}}\n")
    _write(paper / "main.pdf", "x" * 120_000)
    _write(
        paper / "main.log",
        "Output written on main.pdf (9 pages, 120000 bytes).\n",
    )
    _write(paper / "main.blg", "warning$ -- 0\n")
    _write(paper / "main.aux", "\\citation{x}\n")
    _write(paper / "figures" / "framework_overview.pdf", "pdf\n")
    _write(paper / "figures" / "framework_overview.svg", "<svg />\n")
    claim = _write_json(
        tmp_path / "claim.json",
        {
            "ok": True,
            "paper_evidence_ready_for_drafting": True,
            "final_submission_ready": False,
        },
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
    metadata = _write_json(
        tmp_path / "metadata.json",
        {
            "verdict": "PDF_PACKAGE_POLISH_PASS_METADATA_CAUTION",
            "final_submission_ready": False,
            "remaining_blockers": [
                "ProMax final ACM page range and ACM/Crossref registry visibility must be rechecked."
            ],
        },
    )
    return {"paper": paper, "claim": claim, "panel": panel, "metadata": metadata}


def test_submission_package_audit_passes_package_but_keeps_final_ready_false(tmp_path: Path) -> None:
    paths = _seed_package(tmp_path)

    audit = build_submission_package_audit(
        root=tmp_path,
        paper_dir=paths["paper"].relative_to(tmp_path),
        claim_audit_json=paths["claim"].relative_to(tmp_path),
        panel_review_json=paths["panel"].relative_to(tmp_path),
        metadata_followup_json=paths["metadata"].relative_to(tmp_path),
    )

    assert audit["ok"] is True
    assert audit["submission_package_ready_for_target_formatting"] is True
    assert audit["final_submission_ready"] is False
    assert audit["verdict"] == "READY_FOR_TARGET_FORMATTING_NOT_FINAL_SUBMISSION"
    assert audit["build"]["page_count"] == 9
    assert audit["build"]["bibtex_warning_count"] == 0
    assert audit["build"]["overfull_hbox_count"] == 0
    assert audit["evidence_gates"]["panel_score_floor"] == 8.0
    assert audit["remaining_blockers"]


def test_submission_package_audit_fails_on_overfull_or_missing_graphic(tmp_path: Path) -> None:
    paths = _seed_package(tmp_path)
    (paths["paper"] / "figures" / "framework_overview.pdf").unlink()
    _write(
        paths["paper"] / "main.log",
        "Overfull \\hbox (5.0pt too wide)\nOutput written on main.pdf (10 pages, 120000 bytes).\n",
    )

    audit = build_submission_package_audit(
        root=tmp_path,
        paper_dir=paths["paper"].relative_to(tmp_path),
        claim_audit_json=paths["claim"].relative_to(tmp_path),
        panel_review_json=paths["panel"].relative_to(tmp_path),
        metadata_followup_json=paths["metadata"].relative_to(tmp_path),
        max_pages=9,
    )

    assert audit["ok"] is False
    assert "missing_graphic:figures/framework_overview.pdf" in audit["failures"]
    assert "page_count_exceeds_limit:10 > 9" in audit["failures"]
    assert "overfull_hbox_count:1 > 0" in audit["failures"]


def test_submission_package_audit_rejects_nonanonymous_author(tmp_path: Path) -> None:
    paths = _seed_package(tmp_path)
    text = (paths["paper"] / "main.tex").read_text(encoding="utf-8")
    (paths["paper"] / "main.tex").write_text(
        text.replace("\\author{Anonymous}", "\\author{Named Author}"),
        encoding="utf-8",
    )

    audit = build_submission_package_audit(
        root=tmp_path,
        paper_dir=paths["paper"].relative_to(tmp_path),
        claim_audit_json=paths["claim"].relative_to(tmp_path),
        panel_review_json=paths["panel"].relative_to(tmp_path),
        metadata_followup_json=paths["metadata"].relative_to(tmp_path),
    )

    assert audit["ok"] is False
    assert "anonymous_author_or_affiliation_not_ready" in audit["failures"]
