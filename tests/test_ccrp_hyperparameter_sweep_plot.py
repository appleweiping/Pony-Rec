import pytest

from scripts.analysis.main_plot_ccrp_hyperparameter_sweep import build_hyperparameter_summary


def _write_sweep(path):
    path.write_text(
        "domain,score_mode,ablation,eta,confidence_weight,weight_grid_label,audit_ok,degeneracy_audit_ok,NDCG@10,HR@10,MRR\n"
        "sports,full,full,0.5,0.5,\"0.5,0.3,0.2\",true,true,0.20,0.30,0.18\n"
        "sports,full,full,1.0,0.5,\"0.5,0.3,0.2\",true,true,0.24,0.34,0.20\n"
        "sports,full,full,2.0,0.5,\"0.5,0.3,0.2\",true,true,0.22,0.32,0.19\n"
        "sports,confidence_plus_evidence,full,1.0,0.1,\"0.5,0.3,0.2\",true,true,0.21,0.31,0.18\n"
        "sports,confidence_plus_evidence,full,1.0,0.5,\"0.5,0.3,0.2\",true,true,0.25,0.35,0.21\n"
        "sports,confidence_plus_evidence,full,1.0,0.9,\"0.5,0.3,0.2\",true,true,0.23,0.33,0.20\n"
        "sports,full,full,1.0,0.5,\"0.7,0.2,0.1\",true,true,0.23,0.33,0.19\n"
        "sports,full,full,1.0,0.5,\"0.4,0.4,0.2\",true,true,0.26,0.36,0.22\n"
        "sports,full,full,1.0,0.5,\"0.4,0.2,0.4\",true,true,0.21,0.31,0.18\n",
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
    assert summary.loc[
        (summary["control"] == "eta") & (summary["control_value"] == "1"),
        "metric_value",
    ].item() == pytest.approx(0.24)


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
    assert all(row["curve_values"] == 3 for row in provenance["control_reports"])


def test_build_hyperparameter_summary_fails_when_curve_too_short(tmp_path):
    sweep = tmp_path / "valid_ccrp_sweep.csv"
    sweep.write_text(
        "score_mode,ablation,eta,confidence_weight,weight_grid_label,audit_ok,degeneracy_audit_ok,NDCG@10\n"
        "full,full,0.5,0.5,\"0.5,0.3,0.2\",true,true,0.20\n"
        "full,full,1.0,0.5,\"0.5,0.3,0.2\",true,true,0.24\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="below min_values"):
        build_hyperparameter_summary(sweep, controls=["eta"], min_values=3)
