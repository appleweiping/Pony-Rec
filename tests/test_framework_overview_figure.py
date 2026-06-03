import json

from scripts.analysis.main_build_framework_overview_figure import build_framework_figure


def test_build_framework_overview_figure_outputs_editable_and_export_files(tmp_path):
    provenance = build_framework_figure(
        tmp_path,
        title="Actionable Uncertainty for LLM-based Recommendation",
        subtitle="Controlled same-candidate ranking and evidence gates",
    )

    outputs = provenance["outputs"]
    assert set(outputs) >= {"svg", "pdf", "png", "caption", "provenance"}
    assert (tmp_path / "framework_overview.svg").exists()
    assert (tmp_path / "framework_overview.pdf").exists()
    assert (tmp_path / "framework_overview.png").exists()
    assert (tmp_path / "framework_overview_caption.md").exists()
    assert (tmp_path / "framework_overview_provenance.json").exists()

    caption = (tmp_path / "framework_overview_caption.md").read_text(encoding="utf-8")
    assert "controlled same-candidate" in caption
    assert "full-catalog" not in caption.lower()
    saved = json.loads((tmp_path / "framework_overview_provenance.json").read_text(encoding="utf-8"))
    assert saved["claim_boundary"] == "controlled_same_candidate_ranking_not_full_catalog"
