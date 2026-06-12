from pathlib import Path

from scripts.audit.main_build_final_citation_spot_check import build_final_citation_spot_check


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_final_citation_spot_check_accepts_complete_current_bib(tmp_path: Path) -> None:
    aux = _write(
        tmp_path / "Paper" / "main.aux",
        r"""
\citation{llm2rec,llmemb,llmesr,irllrec,proex,promax,elmrec,rlmrec}
\citation{sasrec,guo2017calibration,niculescu2005predicting}
\citation{platt1999probabilistic,zadrozny2002transforming}
\citation{kadavath2022language,tian2023just,lin2022teaching}
\citation{elyaniv2010foundations,geifman2017selective}
\citation{wang2016bayesian,angelopoulos2021gentle,krichene2020sampled}
""",
    )
    keys = [
        "llm2rec",
        "llmemb",
        "llmesr",
        "irllrec",
        "proex",
        "promax",
        "elmrec",
        "rlmrec",
        "sasrec",
        "guo2017calibration",
        "niculescu2005predicting",
        "platt1999probabilistic",
        "zadrozny2002transforming",
        "kadavath2022language",
        "tian2023just",
        "lin2022teaching",
        "elyaniv2010foundations",
        "geifman2017selective",
        "wang2016bayesian",
        "angelopoulos2021gentle",
        "krichene2020sampled",
    ]
    bib = _write(
        tmp_path / "Paper" / "references.bib",
        "\n".join(f"@article{{{key}, title={{T}}, author={{A}}, year={{2026}}}}" for key in keys),
    )
    manuscript = _write(tmp_path / "Paper" / "main.tex", r"\input{sections/abstract}")
    blg = _write(tmp_path / "Paper" / "main.blg", "You've used 21 entries,\nwarning$ -- 0\n")

    audit = build_final_citation_spot_check(
        root=tmp_path,
        manuscript=manuscript.relative_to(tmp_path),
        bibliography=bib.relative_to(tmp_path),
        aux=aux.relative_to(tmp_path),
        bibtex_log=blg.relative_to(tmp_path),
    )

    assert audit["ok"] is True
    assert audit["final_submission_ready"] is False
    assert audit["automated_checks"]["cited_key_count"] == 21
    assert audit["automated_checks"]["bibliography_entry_count"] == 21
    assert audit["automated_checks"]["missing_in_bib"] == []
    assert audit["automated_checks"]["uncited_in_bib"] == []
    assert audit["completeness"]["must_add_count"] == 0
    assert len(audit["completeness"]["official_baselines_cited"]) == 8


def test_final_citation_spot_check_flags_missing_key(tmp_path: Path) -> None:
    aux = _write(tmp_path / "Paper" / "main.aux", r"\citation{elmrec,missing_key}")
    bib = _write(tmp_path / "Paper" / "references.bib", "@article{elmrec, title={T}, year={2024}}\n")
    manuscript = _write(tmp_path / "Paper" / "main.tex", "body")
    blg = _write(tmp_path / "Paper" / "main.blg", "warning$ -- 0\n")

    audit = build_final_citation_spot_check(
        root=tmp_path,
        manuscript=manuscript.relative_to(tmp_path),
        bibliography=bib.relative_to(tmp_path),
        aux=aux.relative_to(tmp_path),
        bibtex_log=blg.relative_to(tmp_path),
    )

    assert audit["ok"] is False
    assert "missing_key" in audit["automated_checks"]["missing_in_bib"]
    assert "official_baseline_citation_missing" in audit["failures"]
