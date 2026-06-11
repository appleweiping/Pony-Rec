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


def _valid_png_header(width: int = 2000, height: int = 1000) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + width.to_bytes(4, "big") + height.to_bytes(4, "big")


def _framework_svg() -> str:
    return (
        "<svg><text>Same-candidate task</text><text>LLM signal extraction</text>"
        "<text>Calibration layer</text><text>C-CRP uncertainty</text>"
        "<text>Risk-adjusted ranking</text><text>Official baseline block</text>"
        "<text>Required method-evidence gates</text><text>Shared evidence gates</text>"
        "<text>before paper-ready claim</text><text>risk_score = base_score</text>"
        "<text>* (1 - uncertainty)^eta</text></svg>\n"
    )


def _seed_framework_package(root: Path) -> None:
    base = root / "outputs/summary/paper_critical/framework_overview"
    caption = (
        "Figure: framework overview. The motivation, ablation, and hyperparameter modules are required gates "
        "before a paper-ready claim.\n"
    )
    files = {
        "framework_overview.svg": _framework_svg(),
        "framework_overview.pdf": "%PDF-1.4\n",
        "framework_overview_caption.md": caption,
    }
    for name, text in files.items():
        _write(base / name, text)
    (base / "framework_overview.png").write_bytes(_valid_png_header())
    _write(
        base / "framework_overview_provenance.json",
        json.dumps(
            {
                "status_label": "paper_critical_framework_overview_review_ready",
                "paper_claim_ready": True,
                "review_status": "review_ready_not_camera_final_template",
                "module_scope": "framework_figure_only_not_substitute_for_observation_ablation_or_hyperparameter_evidence",
                "claim_boundary": "controlled_same_candidate_ranking_not_full_catalog",
                "formula_alignment": {"matches_src_shadow_ccrp_multiplicative_form": True},
                "caption": caption.strip(),
                "evidence_gate_status": {
                    "observation_motivation": "required_not_claimed_by_figure",
                    "component_ablation": "required_not_claimed_by_figure",
                    "hyperparameter_analysis": "required_not_claimed_by_figure",
                },
                "claim_limits": [
                    "Does not claim full-catalog recommendation.",
                    "Does not make observation, ablation, or hyperparameter evidence complete.",
                ],
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
    commands = {
        "generate_valid_signal_rows_template": "python experiments/rsc/run_ccrp_v3_signal_rows.py --split valid",
        "generate_test_signal_rows_template": "python experiments/rsc/run_ccrp_v3_signal_rows.py --split test",
        "select_ccrp_ablation_and_scores_template": "python scripts/misc/main_select_ccrp_variant_on_valid.py",
        "build_component_ablation_summary_template": "python scripts/analysis/main_build_ccrp_component_ablation_summary.py",
        "audit_component_ablation_package_template": "python scripts/audit/main_audit_phase2_5_module_package.py --module component_ablation",
        "build_observation_study_template": "python scripts/analysis/main_build_uncertainty_observation_study.py",
        "audit_observation_package_template": "python scripts/audit/main_audit_phase2_5_module_package.py --module observation_motivation",
        "build_hyperparameter_sweep_template": "python scripts/analysis/main_build_ccrp_hyperparameter_sweep.py",
        "plot_hyperparameter_curves_template": "python scripts/analysis/main_plot_ccrp_hyperparameter_sweep.py",
        "audit_hyperparameter_package_template": "python scripts/audit/main_audit_phase2_5_module_package.py --module hyperparameter_analysis",
    }
    _write(
        base / "ccrp_signal_generation_plan_20260604.json",
        json.dumps(
            {
                "will_start_experiment": False,
                "status_label": "planning_only_not_executed",
                "domains": ["sports", "toys"],
                "current_blocker": "Full-scale artifacts are score-only.",
                "domain_plans": [
                    {"domain": "sports", "commands": commands},
                    {"domain": "toys", "commands": commands},
                ],
            }
        )
        + "\n",
    )
    _write(
        base / "ccrp_signal_generation_plan_20260604.sh",
        "#!/usr/bin/env bash\n"
        "exit 2\n"
        "cd /repo\n"
        "TODO_VALID_SPORTS_CCRP_SIGNAL_JSONL_OR_CSV\n"
        "TODO_TEST_SPORTS_CCRP_SIGNAL_JSONL_OR_CSV\n"
        "python experiments/rsc/run_ccrp_v3_signal_rows.py\n"
        "python scripts/analysis/main_build_uncertainty_observation_study.py\n"
        "python scripts/analysis/main_build_ccrp_component_ablation_summary.py\n"
        "python scripts/analysis/main_plot_ccrp_hyperparameter_sweep.py\n"
        "python scripts/audit/main_audit_phase2_5_module_package.py --module component_ablation\n",
    )


def _seed_paper_critical_support_scripts(root: Path) -> None:
    _write(
        root / "scripts/analysis/main_build_uncertainty_observation_study.py",
        "DEFAULT_KS = (5, 10, 20)\n"
        "paper_critical_observation_motivation\n"
        "paper_critical_observation_ready\n"
        "motivation_only_not_main_table_sota\n"
        "join_report\n"
        "expected_candidates_per_event\n"
        "min_join_rate\n"
        "No uncertainty column found\n"
        "observation_summary.csv\n"
        "observation_event_bins.csv\n"
        "observation_provenance.json\n"
        "fig_uncertainty_motivation\n",
    )
    _write(
        root / "scripts/misc/main_select_ccrp_variant_on_valid.py",
        "FULL_REPORTING_KS = (5, 10, 20)\n"
        "metrics = compute(..., ks=FULL_REPORTING_KS)\n"
        "scripts/misc/main_import_same_candidate_baseline_scores.py\n"
        "\"--tie_break_seed\"\n",
    )
    _write(
        root / "scripts/analysis/main_build_ccrp_component_ablation_summary.py",
        "FULL_KS = (5, 10, 20)\n"
        "component_ablation_summary.csv\n"
        "component_ablation_provenance.json\n"
        "main_config_mode\n"
        "preregistered\n"
        "selected_score_mode_not_full_for_component_ablation\n"
        "valid_sweep_missing_main_ablation\n"
        "_evaluate_candidate_scores(\n",
    )
    _write(
        root / "scripts/audit/main_audit_phase2_5_module_package.py",
        "observation_summary.csv\n"
        "observation_event_bins.csv\n"
        "observation_provenance.json\n"
        "component_ablation_summary.csv\n"
        "valid_ccrp_sweep.csv\n"
        "selected_test_metrics.csv\n"
        "ccrp_hyperparameter_curve_summary.csv\n"
        "ccrp_hyperparameter_curve_provenance.json\n"
        "test_sweep_sha256\n"
        "sweep_source_provenance\n"
        "test_not_used_for_selection\n"
        "retained_scored_temp_rows\n"
        "DEFAULT_CONTROLS = (\"eta\", \"weight_grid_label\")\n",
    )
    _write(
        root / "scripts/analysis/main_build_ccrp_hyperparameter_sweep.py",
        "valid_ccrp_hyperparameter_sweep.csv\n"
        "test_ccrp_hyperparameter_sweep.csv\n"
        "ccrp_hyperparameter_sweep_provenance.json\n"
        "test_not_used_for_selection\n"
        "MAIN_CONTROLS = (\"eta\", \"weight_grid_label\")\n"
        "DIAGNOSTIC_CONTROLS = (\"confidence_weight\",)\n"
        "retained_scored_temp_rows\n"
        "degeneracy_audit_ok\n"
        "score_coverage_rate\n",
    )
    _write(
        root / "scripts/analysis/main_plot_ccrp_hyperparameter_sweep.py",
        "--test_sweep_csv\n"
        "--sweep_provenance_json\n"
        "--require_audit_ok\n"
        "paper_critical_hyperparameter_curve_ready\n"
        "valid_and_test_stability_curve_candidate\n"
        "test_sweep_sha256\n"
        "sweep_source_provenance\n"
        "audit_summary\n"
        "ccrp_hyperparameter_curve_summary.csv\n"
        "ccrp_hyperparameter_curve_provenance.json\n"
        "fig_hyper_eta_curve\n"
        "eta,weight_grid_label\n",
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


def _seed_evidence_consistency(root: Path, *, ok: bool = True) -> Path:
    path = root / "outputs/summary/paper_critical/local_server_evidence_consistency_new_domains_post_backfill_20260606.json"
    _write(
        path,
        json.dumps(
            {
                "ok": ok,
                "row_count": 32,
                "ok_count": 32 if ok else 31,
                "failure_count": 0 if ok else 1,
                "failures": [] if ok else ["example_failure"],
            }
        )
        + "\n",
    )
    return path


def _seed_storage_audit(root: Path, *, launch_allowed: bool = False) -> Path:
    path = root / "outputs/summary/paper_critical/server_storage_phase2_5_retention_audit_current_20260606.json"
    return _seed_storage_audit_at(root, path.name, launch_allowed=launch_allowed)


def _seed_storage_audit_at(root: Path, filename: str, *, launch_allowed: bool = False) -> Path:
    path = root / "outputs/summary/paper_critical" / filename
    current_free = 18_000_000_000 if launch_allowed else 12_000_000_000
    required_free = 16_106_127_360
    recommended = None
    if not launch_allowed:
        recommended = {
            "path": "/srv/project/cache/pony_qwen3_8b_title_item_embs.npy",
            "size_bytes": 5_662_687_360,
            "would_clear_min_free_gate": True,
            "approval_decision_required": True,
        }
    _write(
        path,
        json.dumps(
            {
                "server": {
                    "relevant_python_processes": [],
                    "disk": {"free_bytes": current_free, "used_pct": 93 if launch_allowed else 94},
                },
                "phase2_5_disk_gate": {
                    "current_free_bytes": current_free,
                    "required_free_bytes_min": required_free,
                    "deficit_to_min_free_bytes": max(required_free - current_free, 0),
                    "experiment_launch_allowed": launch_allowed,
                },
                "safe_now_total_recoverable_bytes": 0,
                "recommended_approval_candidate": recommended,
            }
        )
        + "\n",
    )
    return path


def test_audit_marks_framework_scaffold_ready_but_signal_modules_blocked(tmp_path):
    _seed_framework_package(tmp_path)
    _seed_signal_audits(tmp_path)
    _seed_guarded_plan(tmp_path)
    _seed_paper_critical_support_scripts(tmp_path)
    _seed_component_inventory(tmp_path)

    audit = build_module_audit(tmp_path)

    assert audit["ok"] is True
    assert audit["paper_ready"] is False
    assert audit["summary"]["component_inventory_ready"] is True
    assert audit["summary"]["observation_execution_support_ready"] is True
    assert audit["summary"]["component_ablation_execution_support_ready"] is True
    assert audit["summary"]["hyperparameter_execution_support_ready"] is True
    assert audit["summary"]["four_domain_evidence_consistent"] is False
    assert audit["summary"]["phase2_5_storage_launch_allowed"] is False
    assert audit["modules"]["framework_overview"]["artifact_scaffold_ready"] is True
    assert audit["modules"]["framework_overview"]["paper_claim_ready"] is True
    assert audit["modules"]["framework_overview"]["status"] == "review_ready"
    assert audit["modules"]["observation_motivation"]["status"] == "blocked_missing_signal_rows"
    assert audit["modules"]["component_ablation"]["status"] == "blocked_missing_signal_rows"
    assert audit["modules"]["hyperparameter_analysis"]["status"] == "blocked_missing_signal_rows"
    assert audit["signal_source_state"]["score_only_source_count"] == 4
    assert audit["guarded_signal_plan"]["status"] == "guarded_plan_ready_not_executable"
    assert "missing_phase2_5_storage_audit" in audit["modules"]["observation_motivation"]["blockers"]


def test_audit_integrates_evidence_consistency_and_storage_gate(tmp_path):
    _seed_framework_package(tmp_path)
    _seed_signal_audits(tmp_path)
    _seed_guarded_plan(tmp_path)
    _seed_paper_critical_support_scripts(tmp_path)
    _seed_component_inventory(tmp_path)
    evidence_path = _seed_evidence_consistency(tmp_path, ok=True)
    storage_path = _seed_storage_audit(tmp_path, launch_allowed=False)

    audit = build_module_audit(
        tmp_path,
        evidence_consistency_json=evidence_path,
        storage_audit_json=storage_path,
    )

    assert audit["summary"]["four_domain_evidence_consistent"] is True
    assert audit["evidence_consistency"]["row_count"] == 32
    assert audit["summary"]["phase2_5_storage_launch_allowed"] is False
    assert audit["summary"]["storage_cleanup_decision_required"] is True
    assert audit["summary"]["storage_approval_decision_required"] is True
    assert audit["summary"]["storage_recommended_candidate_would_clear_min_free_gate"] is True
    assert audit["summary"]["storage_recommended_candidate_size_bytes"] == 5_662_687_360
    assert audit["storage_gate"]["deficit_to_min_free_bytes"] > 0
    assert "pony_qwen3_8b_title_item_embs.npy" in audit["next_action"]
    assert "server_disk_below_phase2_5_floor" in audit["modules"]["hyperparameter_analysis"]["blockers"]


def test_audit_prefers_after_cleanup_storage_gate_by_default(tmp_path):
    _seed_framework_package(tmp_path)
    _seed_signal_audits(tmp_path)
    _seed_guarded_plan(tmp_path)
    _seed_paper_critical_support_scripts(tmp_path)
    _seed_component_inventory(tmp_path)
    _seed_storage_audit_at(
        tmp_path,
        "server_storage_phase2_5_retention_audit_current_20260606_0432.json",
        launch_allowed=False,
    )
    after_cleanup_path = _seed_storage_audit_at(
        tmp_path,
        "server_storage_phase2_5_retention_audit_after_cleanup_final_20260606_0650.json",
        launch_allowed=True,
    )

    audit = build_module_audit(tmp_path)

    assert audit["summary"]["phase2_5_storage_launch_allowed"] is True
    assert audit["storage_gate"]["path"].endswith(after_cleanup_path.name)
    assert "phase2_5_experiment_launch_not_allowed" not in audit["modules"]["observation_motivation"]["blockers"]


def test_audit_detects_framework_manifest_mismatch(tmp_path):
    _seed_framework_package(tmp_path)
    _seed_signal_audits(tmp_path)
    _seed_guarded_plan(tmp_path)
    _seed_paper_critical_support_scripts(tmp_path)
    _seed_component_inventory(tmp_path)
    _write(tmp_path / "outputs/summary/paper_critical/framework_overview/framework_overview.svg", "<svg>changed</svg>\n")

    audit = build_module_audit(tmp_path)

    assert audit["ok"] is False
    assert audit["modules"]["framework_overview"]["artifact_scaffold_ready"] is False
    assert "manifest_mismatch:framework_overview.svg" in audit["modules"]["framework_overview"]["remaining_blockers"]


def _refresh_framework_manifest(root: Path) -> None:
    base = root / "outputs/summary/paper_critical/framework_overview"
    manifest_names = [
        "framework_overview.svg",
        "framework_overview.pdf",
        "framework_overview.png",
        "framework_overview_caption.md",
        "framework_overview_provenance.json",
    ]
    _write(base / "framework_overview_manifest.sha256", "".join(f"{_sha(base / name)}  {name}\n" for name in manifest_names))


def test_audit_detects_framework_overclaiming_gate_status(tmp_path):
    _seed_framework_package(tmp_path)
    _seed_signal_audits(tmp_path)
    _seed_guarded_plan(tmp_path)
    _seed_paper_critical_support_scripts(tmp_path)
    _seed_component_inventory(tmp_path)
    provenance_path = tmp_path / "outputs/summary/paper_critical/framework_overview/framework_overview_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["evidence_gate_status"]["observation_motivation"] = "completed_by_figure"
    provenance_path.write_text(json.dumps(provenance, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_framework_manifest(tmp_path)

    audit = build_module_audit(tmp_path)

    assert audit["ok"] is False
    assert audit["modules"]["framework_overview"]["artifact_scaffold_ready"] is False
    assert (
        "framework_evidence_gate_status_mismatch:observation_motivation"
        in audit["modules"]["framework_overview"]["remaining_blockers"]
    )


def test_audit_detects_framework_caption_overclaim(tmp_path):
    _seed_framework_package(tmp_path)
    _seed_signal_audits(tmp_path)
    _seed_guarded_plan(tmp_path)
    _seed_paper_critical_support_scripts(tmp_path)
    _seed_component_inventory(tmp_path)
    caption_path = tmp_path / "outputs/summary/paper_critical/framework_overview/framework_overview_caption.md"
    caption_path.write_text("Figure: observation complete and ablations complete.\n", encoding="utf-8")
    provenance_path = tmp_path / "outputs/summary/paper_critical/framework_overview/framework_overview_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["caption"] = caption_path.read_text(encoding="utf-8").strip()
    provenance_path.write_text(json.dumps(provenance, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_framework_manifest(tmp_path)

    audit = build_module_audit(tmp_path)

    blockers = audit["modules"]["framework_overview"]["remaining_blockers"]
    assert audit["ok"] is False
    assert "framework_caption_overclaim_phrase:observation complete" in blockers
    assert "framework_caption_missing_required_phrase:required gates before a paper-ready claim" in blockers


def test_audit_detects_framework_svg_overclaim(tmp_path):
    _seed_framework_package(tmp_path)
    _seed_signal_audits(tmp_path)
    _seed_guarded_plan(tmp_path)
    _seed_paper_critical_support_scripts(tmp_path)
    _seed_component_inventory(tmp_path)
    svg_path = tmp_path / "outputs/summary/paper_critical/framework_overview/framework_overview.svg"
    svg_path.write_text(_framework_svg().replace("before paper-ready claim", "hyperparameter analysis complete"), encoding="utf-8")
    _refresh_framework_manifest(tmp_path)

    audit = build_module_audit(tmp_path)

    blockers = audit["modules"]["framework_overview"]["remaining_blockers"]
    assert audit["ok"] is False
    assert "framework_svg_overclaim_phrase:hyperparameter analysis complete" in blockers
    assert "framework_svg_missing_required_phrase:before paper-ready claim" in blockers


def test_audit_detects_framework_claim_limit_removal(tmp_path):
    _seed_framework_package(tmp_path)
    _seed_signal_audits(tmp_path)
    _seed_guarded_plan(tmp_path)
    _seed_paper_critical_support_scripts(tmp_path)
    _seed_component_inventory(tmp_path)
    provenance_path = tmp_path / "outputs/summary/paper_critical/framework_overview/framework_overview_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["claim_limits"] = ["Does not claim full-catalog recommendation."]
    provenance_path.write_text(json.dumps(provenance, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_framework_manifest(tmp_path)

    audit = build_module_audit(tmp_path)

    assert audit["ok"] is False
    assert (
        "framework_claim_limit_missing:no_module_completion_claim"
        in audit["modules"]["framework_overview"]["remaining_blockers"]
    )


def test_write_markdown_summary(tmp_path):
    _seed_framework_package(tmp_path)
    _seed_signal_audits(tmp_path)
    _seed_guarded_plan(tmp_path)
    _seed_paper_critical_support_scripts(tmp_path)
    _seed_component_inventory(tmp_path)
    audit = build_module_audit(tmp_path)
    output = tmp_path / "audit.md"

    write_markdown(output, audit)

    text = output.read_text(encoding="utf-8")
    assert "Paper-Critical Module Audit" in text
    assert "`observation_motivation`" in text
    assert "blocked_missing_signal_rows" in text
    assert "Component inventory ready" in text
    assert "Observation execution support ready" in text
    assert "Component-ablation execution support ready" in text
    assert "Hyperparameter execution support ready" in text
    assert "Storage cleanup decision required" in text
