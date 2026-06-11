import pytest

from scripts.analysis.main_plot_ccrp_hyperparameter_sweep import build_hyperparameter_summary


def _write_sweep(path):
    path.write_text(
        "domain,score_mode,ablation,eta,confidence_weight,weight_grid_label,audit_ok,degeneracy_audit_ok,score_coverage_rate,candidate_key_count,NDCG@10,HR@10,MRR\n"
        "sports,full,full,0.5,0.5,\"0.5,0.3,0.2\",true,true,1.0,1010000,0.20,0.30,0.18\n"
        "sports,full,full,1.0,0.5,\"0.5,0.3,0.2\",true,true,1.0,1010000,0.24,0.34,0.20\n"
        "sports,full,full,2.0,0.5,\"0.5,0.3,0.2\",true,true,1.0,1010000,0.22,0.32,0.19\n"
        "sports,confidence_plus_evidence,full,1.0,0.1,\"0.5,0.3,0.2\",true,true,1.0,1010000,0.21,0.31,0.18\n"
        "sports,confidence_plus_evidence,full,1.0,0.5,\"0.5,0.3,0.2\",true,true,1.0,1010000,0.25,0.35,0.21\n"
        "sports,confidence_plus_evidence,full,1.0,0.9,\"0.5,0.3,0.2\",true,true,1.0,1010000,0.23,0.33,0.20\n"
        "sports,full,full,1.0,0.5,\"0.7,0.2,0.1\",true,true,1.0,1010000,0.23,0.33,0.19\n"
        "sports,full,full,1.0,0.5,\"0.4,0.4,0.2\",true,true,1.0,1010000,0.26,0.36,0.22\n"
        "sports,full,full,1.0,0.5,\"0.4,0.2,0.4\",true,true,1.0,1010000,0.21,0.31,0.18\n",
        encoding="utf-8",
    )


def _write_unstable_test_sweep(path):
    path.write_text(
        "domain,score_mode,ablation,eta,confidence_weight,weight_grid_label,audit_ok,degeneracy_audit_ok,score_coverage_rate,candidate_key_count,NDCG@10,HR@10,MRR\n"
        "sports,full,full,0.5,0.5,\"0.5,0.3,0.2\",true,true,1.0,1010000,0.10,0.20,0.08\n"
        "sports,full,full,1.0,0.5,\"0.5,0.3,0.2\",true,true,1.0,1010000,0.12,0.22,0.10\n"
        "sports,full,full,2.0,0.5,\"0.5,0.3,0.2\",true,true,1.0,1010000,0.30,0.40,0.28\n",
        encoding="utf-8",
    )


def test_build_hyperparameter_summary_requires_three_values(tmp_path):
    sweep = tmp_path / "valid_ccrp_sweep.csv"
    _write_sweep(sweep)

    summary, provenance = build_hyperparameter_summary(
        sweep,
        domain="sports",
        metric="NDCG@10",
        controls=["eta", "confidence_weight", "weight_grid_label"],
        min_values=3,
    )

    assert set(summary["control"]) == {"eta", "confidence_weight", "weight_grid_label"}
    assert set(summary["domain"]) == {"sports"}
    assert set(summary["split"]) == {"valid"}
    assert len(summary) == 10
    assert all(row["meets_min_values"] for row in provenance["control_reports"])
    assert provenance["reporting_mode"] == "valid_only"
    assert provenance["artifact_class"] == "paper_critical_hyperparameter_analysis"
    assert provenance["status_label"] == "validation_only_hyperparameter_selection_curve"
    assert provenance["paper_claim_scope"] == "validation_only_not_stability_claim"
    assert provenance["stability_report"][0]["has_valid_and_test"] is False
    assert summary.loc[
        (summary["control"] == "eta") & (summary["control_value"] == "1"),
        "metric_value",
    ].item() == pytest.approx(0.24)


def test_build_hyperparameter_summary_uses_control_column_and_embeds_source_provenance(tmp_path):
    sweep = tmp_path / "valid_ccrp_hyperparameter_sweep.csv"
    sweep.write_text(
        "domain,split,row_kind,control,control_value,score_mode,ablation,eta,confidence_weight,weight_grid_label,audit_ok,degeneracy_audit_ok,score_coverage_rate,candidate_key_count,NDCG@10,HR@10,MRR\n"
        "sports,valid,main_control,eta,0,full,full,0,0.7,\"0.5,0.3,0.2\",true,true,1.0,1010000,0.20,0.30,0.18\n"
        "sports,valid,main_control,eta,1,full,full,1,0.7,\"0.5,0.3,0.2\",true,true,1.0,1010000,0.24,0.34,0.20\n"
        "sports,valid,main_control,eta,2,full,full,2,0.7,\"0.5,0.3,0.2\",true,true,1.0,1010000,0.22,0.32,0.19\n"
        "sports,valid,main_control,weight_grid_label,\"0.5,0.3,0.2\",full,full,1,0.7,\"0.5,0.3,0.2\",true,true,1.0,1010000,0.24,0.34,0.20\n"
        "sports,valid,main_control,weight_grid_label,\"0.7,0.2,0.1\",full,full,1,0.7,\"0.7,0.2,0.1\",true,true,1.0,1010000,0.23,0.33,0.19\n"
        "sports,valid,main_control,weight_grid_label,\"0.4,0.4,0.2\",full,full,1,0.7,\"0.4,0.4,0.2\",true,true,1.0,1010000,0.26,0.36,0.22\n",
        encoding="utf-8",
    )
    source_provenance = tmp_path / "ccrp_hyperparameter_sweep_provenance.json"
    source_provenance.write_text(
        '{"status_label":"valid_test_saved_signal_hyperparameter_sweep_ready","test_not_used_for_selection":true}\n',
        encoding="utf-8",
    )

    summary, provenance = build_hyperparameter_summary(
        sweep,
        sweep_provenance_json=source_provenance,
        domain="sports",
        controls=["eta", "weight_grid_label"],
        min_values=3,
    )

    assert len(summary[summary["control"] == "eta"]) == 3
    assert len(summary[summary["control"] == "weight_grid_label"]) == 3
    assert provenance["test_not_used_for_selection"] is True
    assert provenance["sweep_source_provenance"]["status_label"] == "valid_test_saved_signal_hyperparameter_sweep_ready"


def test_build_hyperparameter_summary_reports_valid_and_test_separately(tmp_path):
    valid = tmp_path / "valid_ccrp_sweep.csv"
    test = tmp_path / "test_ccrp_sweep.csv"
    _write_sweep(valid)
    _write_sweep(test)

    summary, provenance = build_hyperparameter_summary(
        valid,
        test_sweep_csv=test,
        domain="sports",
        controls=["eta"],
        min_values=3,
    )

    assert set(summary["split"]) == {"valid", "test"}
    assert len(summary) == 6
    assert provenance["reporting_mode"] == "valid_and_test"
    assert provenance["status_label"] == "paper_critical_hyperparameter_curve_ready"
    assert provenance["paper_claim_scope"] == "valid_and_test_stability_curve_candidate"
    assert all(row["curve_values"] == 3 for row in provenance["control_reports"])
    assert provenance["stability_report"][0]["control"] == "eta"
    assert provenance["stability_report"][0]["valid_best_value"] == "1"
    assert provenance["stability_report"][0]["test_best_value"] == "1"
    assert provenance["stability_report"][0]["relative_drop_from_test_best"] == pytest.approx(0.0)
    assert provenance["stability_report"][0]["stable_within_tolerance"] is True


def test_build_hyperparameter_summary_downgrades_unstable_valid_test_curves(tmp_path):
    valid = tmp_path / "valid_ccrp_sweep.csv"
    test = tmp_path / "test_ccrp_sweep.csv"
    _write_sweep(valid)
    _write_unstable_test_sweep(test)

    _, provenance = build_hyperparameter_summary(
        valid,
        test_sweep_csv=test,
        domain="sports",
        controls=["eta"],
        min_values=3,
    )

    assert provenance["status_label"] == "valid_and_test_instability_report_not_stability_claim"
    assert provenance["paper_claim_scope"] == "diagnostic_only_not_paper_stability"
    assert provenance["stability_report"][0]["valid_best_value"] == "1"
    assert provenance["stability_report"][0]["test_best_value"] == "2"
    assert provenance["stability_report"][0]["stable_within_tolerance"] is False
    assert provenance["stability_report"][0]["relative_drop_from_test_best"] > 0.05


def test_build_hyperparameter_summary_fails_when_curve_too_short(tmp_path):
    sweep = tmp_path / "valid_ccrp_sweep.csv"
    sweep.write_text(
        "score_mode,ablation,eta,confidence_weight,weight_grid_label,audit_ok,degeneracy_audit_ok,score_coverage_rate,candidate_key_count,NDCG@10\n"
        "full,full,0.5,0.5,\"0.5,0.3,0.2\",true,true,1.0,1010000,0.20\n"
        "full,full,1.0,0.5,\"0.5,0.3,0.2\",true,true,1.0,1010000,0.24\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="below min_values"):
        build_hyperparameter_summary(sweep, controls=["eta"], min_values=3)


def test_build_hyperparameter_summary_requires_audit_columns_by_default(tmp_path):
    sweep = tmp_path / "valid_ccrp_sweep.csv"
    sweep.write_text(
        "score_mode,ablation,eta,confidence_weight,weight_grid_label,NDCG@10\n"
        "full,full,0.5,0.5,\"0.5,0.3,0.2\",0.20\n"
        "full,full,1.0,0.5,\"0.5,0.3,0.2\",0.24\n"
        "full,full,2.0,0.5,\"0.5,0.3,0.2\",0.22\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required audit columns"):
        build_hyperparameter_summary(sweep, controls=["eta"], min_values=3)


def test_build_hyperparameter_summary_without_audit_is_diagnostic(tmp_path):
    sweep = tmp_path / "valid_ccrp_sweep.csv"
    sweep.write_text(
        "score_mode,ablation,eta,confidence_weight,weight_grid_label,NDCG@10\n"
        "full,full,0.5,0.5,\"0.5,0.3,0.2\",0.20\n"
        "full,full,1.0,0.5,\"0.5,0.3,0.2\",0.24\n"
        "full,full,2.0,0.5,\"0.5,0.3,0.2\",0.22\n",
        encoding="utf-8",
    )

    _, provenance = build_hyperparameter_summary(
        sweep,
        controls=["eta"],
        min_values=3,
        require_audit_ok=False,
    )

    assert provenance["status_label"] == "diagnostic_hyperparameter_curve_audit_not_enforced"
    assert provenance["paper_claim_scope"] == "diagnostic_only_not_paper_stability"
    assert provenance["audit_summary"]["require_audit_ok"] is False
