import csv
import json
from pathlib import Path

from scripts.analysis.main_aggregate_ccrp_hyperparameter_analysis import aggregate_hyperparameter_analysis


METRICS = ("MRR", "HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20")


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _json(path, payload):
    return _write(path, json.dumps(payload, indent=2) + "\n")


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _seed_package(root, domain, *, audit_ok=True, stable=True):
    package = root / f"ccrp_hyperparameter_{domain}"
    _json(
        package / "phase2_5_hyperparameter_package_audit.json",
        {
            "ok": audit_ok,
            "paper_claim_ready": audit_ok,
            "module_audit": {
                "status_label": "paper_critical_hyperparameter_curve_ready" if audit_ok else "blocked",
            },
        },
    )
    stability = [
        {
            "control": "eta",
            "metric": "NDCG@10",
            "relative_drop_tolerance": 0.05,
            "has_valid_and_test": True,
            "valid_best_value": "0",
            "test_best_value": "0",
            "valid_metric_at_best": 0.2,
            "test_best_metric": 0.21,
            "test_metric_at_valid_best": 0.21 if stable else 0.15,
            "relative_drop_from_test_best": 0.0 if stable else 0.285,
            "test_rank_of_valid_best": 1 if stable else 6,
            "best_value_match": True,
            "stable_within_tolerance": stable,
            "reason": "within_tolerance" if stable else "test_drop_exceeds_tolerance",
        },
        {
            "control": "weight_grid_label",
            "metric": "NDCG@10",
            "relative_drop_tolerance": 0.05,
            "has_valid_and_test": True,
            "valid_best_value": "0.5,0.3,0.2",
            "test_best_value": "0.4,0.4,0.2",
            "valid_metric_at_best": 0.2,
            "test_best_metric": 0.21,
            "test_metric_at_valid_best": 0.209,
            "relative_drop_from_test_best": 0.0047619,
            "test_rank_of_valid_best": 2,
            "best_value_match": False,
            "stable_within_tolerance": True,
            "reason": "within_tolerance",
        },
    ]
    _json(
        package / "ccrp_hyperparameter_curve_provenance.json",
        {
            "artifact_class": "paper_critical_hyperparameter_analysis",
            "status_label": "paper_critical_hyperparameter_curve_ready",
            "paper_claim_scope": "valid_and_test_stability_curve_candidate",
            "domain": domain,
            "reporting_mode": "valid_and_test",
            "metric": "NDCG@10",
            "controls": ["eta", "weight_grid_label"],
            "test_not_used_for_selection": True,
            "stability_report": stability,
            "sweep_source_provenance": {
                "test_not_used_for_selection": True,
                "row_counts": {
                    "valid_signal_rows": 1010000,
                    "test_signal_rows": 1010000,
                    "valid_candidate_rows": 1010000,
                    "test_candidate_rows": 1010000,
                    "valid_ranking_events": 10000,
                    "test_ranking_events": 10000,
                },
            },
        },
    )
    rows = []
    for split in ("valid", "test"):
        for control, values in {
            "eta": ("0", "0.25", "1"),
            "weight_grid_label": ("0.5,0.3,0.2", "0.4,0.4,0.2", "0.7,0.2,0.1"),
        }.items():
            for idx, value in enumerate(values):
                row = {
                    "domain": domain,
                    "split": split,
                    "row_kind": "main_control",
                    "control": control,
                    "control_value": value,
                    "score_mode": "full",
                    "ablation": "full",
                    "candidate_key_count": "1010000",
                    "score_coverage_rate": "1.0",
                    "audit_ok": "True",
                    "degeneracy_audit_ok": "True",
                    "metric_name": "NDCG@10",
                    "metric_value": str(0.2 + idx / 1000.0),
                }
                for metric in METRICS:
                    row[metric] = row["metric_value"]
                rows.append(row)
    _csv(package / "ccrp_hyperparameter_curve_summary.csv", rows)
    return package


def test_aggregate_hyperparameter_analysis_passes_four_domain_stability(tmp_path):
    root = tmp_path / "packages"
    for domain in ("sports", "toys", "home", "tools"):
        _seed_package(root, domain)

    provenance = aggregate_hyperparameter_analysis(package_root=root, output_dir=tmp_path / "out", skip_plot=True)

    assert provenance["ok"] is True
    assert provenance["paper_claim_ready"] is True
    assert provenance["all_controls_stable"] is True
    assert provenance["table_eligibility"] == "supplementary_hyperparameter_stability_only"
    summary = (tmp_path / "out" / "ccrp_hyperparameter_four_domain_control_summary.csv").read_text(encoding="utf-8")
    assert "stable_with_domain_specific_best_values" in summary


def test_aggregate_hyperparameter_analysis_fails_closed_on_unready_package(tmp_path):
    root = tmp_path / "packages"
    for domain in ("sports", "toys", "home", "tools"):
        _seed_package(root, domain, audit_ok=(domain != "home"))

    provenance = aggregate_hyperparameter_analysis(package_root=root, output_dir=tmp_path / "out", skip_plot=True)

    assert provenance["ok"] is False
    assert "package_audit_not_ready:home" in provenance["failures"]


def test_aggregate_hyperparameter_analysis_fails_on_unstable_control(tmp_path):
    root = tmp_path / "packages"
    for domain in ("sports", "toys", "home", "tools"):
        _seed_package(root, domain, stable=(domain != "toys"))

    provenance = aggregate_hyperparameter_analysis(package_root=root, output_dir=tmp_path / "out", skip_plot=True)

    assert provenance["ok"] is False
    assert "stability_not_ready:toys:eta:test_drop_exceeds_tolerance" in provenance["failures"]


def test_aggregate_hyperparameter_analysis_writes_figures(tmp_path):
    root = tmp_path / "packages"
    for domain in ("sports", "toys", "home", "tools"):
        _seed_package(root, domain)

    provenance = aggregate_hyperparameter_analysis(package_root=root, output_dir=tmp_path / "out")

    figure_paths = provenance["outputs"]["figure_paths"]
    assert len(figure_paths) == 6
    assert any(path.endswith("fig_hyperparameter_four_domain_stability.png") for path in figure_paths)
    assert any(path.endswith("fig_hyperparameter_four_domain_eta.png") for path in figure_paths)
    assert any(path.endswith("fig_hyperparameter_four_domain_weight.png") for path in figure_paths)
    for path in figure_paths:
        assert (tmp_path / "out" / Path(path).name).exists()
