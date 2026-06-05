import json

from scripts.analysis.main_build_framework_overview_figure import build_framework_figure


def test_build_framework_overview_figure_outputs_editable_and_export_files(tmp_path):
    provenance = build_framework_figure(
        tmp_path,
        title="Actionable Uncertainty for LLM-based Recommendation",
        subtitle="Controlled same-candidate ranking and evidence gates",
    )

    outputs = provenance["outputs"]
    assert set(outputs) >= {"svg", "pdf", "png", "caption", "provenance", "manifest"}
    assert (tmp_path / "framework_overview.svg").exists()
    assert (tmp_path / "framework_overview.pdf").exists()
    assert (tmp_path / "framework_overview.png").exists()
    assert (tmp_path / "framework_overview_caption.md").exists()
    assert (tmp_path / "framework_overview_provenance.json").exists()
    assert (tmp_path / "framework_overview_manifest.sha256").exists()

    caption = (tmp_path / "framework_overview_caption.md").read_text(encoding="utf-8")
    assert "controlled same-candidate" in caption
    assert "required gates before a paper-ready claim" in caption
    assert "full-catalog" not in caption.lower()
    svg = (tmp_path / "framework_overview.svg").read_text(encoding="utf-8")
    assert "risk_score = base_score" in svg
    assert "* (1 - uncertainty)^eta" in svg
    assert "Counterevidence" in svg
    assert "Required method-evidence gates" in svg
    assert "Paper-critical method evidence" not in svg
    assert all(line == line.rstrip() for line in svg.splitlines())
    manifest = (tmp_path / "framework_overview_manifest.sha256").read_text(encoding="utf-8")
    assert "framework_overview.svg" in manifest
    assert "framework_overview_provenance.json" in manifest
    saved = json.loads((tmp_path / "framework_overview_provenance.json").read_text(encoding="utf-8"))
    assert saved["status_label"] == "paper_critical_framework_overview_review_ready"
    assert saved["paper_claim_ready"] is True
    assert saved["claim_boundary"] == "controlled_same_candidate_ranking_not_full_catalog"
    assert saved["caption"] == caption.strip()
    assert "not_substitute_for_observation_ablation_or_hyperparameter" in saved["module_scope"]
    assert saved["evidence_gate_status"]["observation_motivation"] == "required_not_claimed_by_figure"
    assert saved["claim_limits"][-1] == "Does not make observation, ablation, or hyperparameter evidence complete."
    assert saved["formula_alignment"]["matches_src_shadow_ccrp_multiplicative_form"] is True
    assert saved["command"].startswith("python scripts/analysis/main_build_framework_overview_figure.py")
