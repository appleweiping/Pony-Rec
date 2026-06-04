import hashlib
import json
from pathlib import Path

from scripts.audit.main_audit_paper_critical_modules import build_module_audit, write_markdown


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_framework_package(root: Path) -> None:
    base = root / "outputs/summary/paper_critical/framework_overview"
    files = {
        "framework_overview.svg": "<svg><text>C-CRP uncertainty</text></svg>\n",
        "framework_overview.pdf": "%PDF-1.4\n",
        "framework_overview.png": "PNG bytes\n",
        "framework_overview_caption.md": "caption\n",
    }
    for name, text in files.items():
        _write(base / name, text)
    _write(
        base / "framework_overview_provenance.json",
        json.dumps(
            {
                "status_label": "paper_critical_framework_overview_draft",
                "claim_boundary": "controlled_same_candidate_ranking_not_full_catalog",
                "git_commit": "abc123",
                "generated_at_utc": "2026-06-04T00:00:00+00:00",
            }
        )
        + "\n",
    )
    manifest_names = [
        "framework_overview.svg",
        "framework_overview.pdf",
        "framework_overview.png",
        "framework_overview_caption.md",
        "framework_overview_provenance.json",
    ]
    _write(base / "framework_overview_manifest.sha256", "".join(f"{_sha(base / name)}  {name}\n" for name in manifest_names))


def _seed_signal_audits(root: Path) -> None:
    base = root / "outputs/summary/paper_critical"
    for domain in ("sports", "toys", "home", "tools"):
        _write(
            base / f"ccrp_uncertainty_source_audit_{domain}_fixed_filter_20260604_0502.json",
            json.dumps(
                {
                    "sources": [
                        {
                            "audit_status": "score_only_not_uncertainty",
                            "audit_paper_ready_uncertainty_rows": False,
                            "audit_recomputable_signal_rows": False,
                            "audit_candidate_key_coverage_rate": 1.0,
                        }
                    ]
                }
            )
            + "\n",
        )
    _write(
        base / "ccrp_formal_signal_path_trace_20260604_0535.json",
        json.dumps(
            {
                "can_rebuild_paper_ready_uncertainty_rows_from_formal_scores_only": False,
                "blockers": ["formal_scores_schema_has_no_uncertainty_column"],
            }
        )
        + "\n",
    )


def _seed_guarded_plan(root: Path) -> None:
    base = root / "outputs/summary/paper_critical/ccrp_signal_generation_plan"
    _write(
        base / "ccrp_signal_generation_plan_20260604.json",
        json.dumps(
            {
                "will_start_experiment": False,
                "status_label": "planning_only_not_executed",
                "domains": ["sports", "toys"],
                "current_blocker": "Full-scale artifacts are score-only.",
            }
        )
        + "\n",
    )
    _write(
        base / "ccrp_signal_generation_plan_20260604.sh",
        "#!/usr/bin/env bash\nexit 2\ncd /repo\nTODO_VALID_SPORTS_CCRP_SIGNAL_JSONL_OR_CSV\nTODO_TEST_SPORTS_CCRP_SIGNAL_JSONL_OR_CSV\n",
    )


def _seed_component_inventory(root: Path) -> None:
    base = root / "outputs/summary/paper_critical/ccrp_component_inventory"
    components = [
        {"id": "boundary_uncertainty"},
        {"id": "calibration_gap"},
        {"id": "evidence_support_insufficiency"},
        {"id": "counterevidence"},
        {"id": "risk_penalty"},
        {"id": "eta_risk_exponent"},
        {"id": "confidence_weight"},
        {"id": "uncertainty_weight_triple"},
        {"id": "raw_vs_calibrated_posterior"},
        {"id": "temperature_prompt_variants"},
    ]
    _write(
        base / "ccrp_component_inventory_20260604.json",
        json.dumps(
            {
                "status_label": "paper_critical_ccrp_component_inventory",
                "paper_claim_ready": False,
                "component_count": len(components),
                "blocked_by": ["missing_full_scale_uncertainty_or_recomputable_signal_rows"],
                "formula_alignment": {"figure_formula_contains_multiplicative_form": True},
                "components": components,
            }
        )
        + "\n",
    )
    _write(base / "ccrp_component_inventory_20260604.md", "# C-CRP Component Inventory\n")


def test_audit_marks_framework_scaffold_ready_but_signal_modules_blocked(tmp_path):
    _seed_framework_package(tmp_path)
    _seed_signal_audits(tmp_path)
    _seed_guarded_plan(tmp_path)
    _seed_component_inventory(tmp_path)

    audit = build_module_audit(tmp_path)

    assert audit["ok"] is True
    assert audit["paper_ready"] is False
    assert audit["summary"]["component_inventory_ready"] is True
    assert audit["modules"]["framework_overview"]["artifact_scaffold_ready"] is True
    assert audit["modules"]["observation_motivation"]["status"] == "blocked_missing_signal_rows"
    assert audit["modules"]["component_ablation"]["status"] == "blocked_missing_signal_rows"
    assert audit["modules"]["hyperparameter_analysis"]["status"] == "blocked_missing_signal_rows"
    assert audit["signal_source_state"]["score_only_source_count"] == 4
    assert audit["guarded_signal_plan"]["status"] == "guarded_plan_ready_not_executable"


def test_audit_detects_framework_manifest_mismatch(tmp_path):
    _seed_framework_package(tmp_path)
    _seed_signal_audits(tmp_path)
    _seed_guarded_plan(tmp_path)
    _seed_component_inventory(tmp_path)
    _write(tmp_path / "outputs/summary/paper_critical/framework_overview/framework_overview.svg", "<svg>changed</svg>\n")

    audit = build_module_audit(tmp_path)

    assert audit["ok"] is False
    assert audit["modules"]["framework_overview"]["artifact_scaffold_ready"] is False
    assert "manifest_mismatch:framework_overview.svg" in audit["modules"]["framework_overview"]["remaining_blockers"]


def test_write_markdown_summary(tmp_path):
    _seed_framework_package(tmp_path)
    _seed_signal_audits(tmp_path)
    _seed_guarded_plan(tmp_path)
    _seed_component_inventory(tmp_path)
    audit = build_module_audit(tmp_path)
    output = tmp_path / "audit.md"

    write_markdown(output, audit)

    text = output.read_text(encoding="utf-8")
    assert "Paper-Critical Module Audit" in text
    assert "`observation_motivation`" in text
    assert "blocked_missing_signal_rows" in text
    assert "Component inventory ready" in text
