import json
from pathlib import Path

from scripts.audit.main_audit_phase2_5_module_package import build_audit


METRICS = {
    "MRR": 0.2,
    "HR@5": 0.3,
    "HR@10": 0.4,
    "HR@20": 0.5,
    "NDCG@5": 0.21,
    "NDCG@10": 0.31,
    "NDCG@20": 0.41,
}
ABLATIONS = (
    "full",
    "without_boundary_uncertainty",
    "without_calibration_gap",
    "without_evidence_support",
    "without_counterevidence",
    "without_risk_penalty",
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _json(path: Path, payload: dict) -> Path:
    return _write(path, json.dumps(payload, indent=2) + "\n")


def _general_files(root: Path) -> None:
    _write(root / "log_snippets.md", "completed cleanly; no fatal markers observed\n")
    _json(root / "run_config.json", {"seed": 13, "expected_users": 2})
    _json(
        root / "local_server_manifest_comparison.json",
        {
            "ok": True,
            "row_count": 1,
            "ok_count": 1,
            "rows": [
                {
                    "path": "tables/ranking_metrics.csv",
                    "ok": True,
                    "files": {"tables/ranking_metrics.csv": {"present": True, "size": 10}},
                }
            ],
        },
    )


def _complete_observation_package(root: Path) -> None:
    _general_files(root)
    header = _metric_header(["domain", "method", "uncertainty_bin_index", "uncertainty_bin", "n_events"])
    rows = [_metric_values(["sports", method, -1, "ALL", 2]) for method in ("ccrp", "proex")]
    rows += [_metric_values(["sports", method, 0, "low", 1]) for method in ("ccrp", "proex")]
    _write(root / "observation_summary.csv", _csv_line(header) + "".join(_csv_line(row) for row in rows))
    _json(root / "observation_summary.json", {"rows": [dict(zip(header, row)) for row in rows]})
    event_header = _metric_header(
        ["event_id", "method", "uncertainty_bin_index", "uncertainty_bin", "positive_rank", "num_candidates"]
    )
    event_rows = [
        _metric_values([f"e{idx}", method, idx % 2, "low" if idx % 2 == 0 else "high", idx + 1, 101])
        for method in ("ccrp", "proex")
        for idx in range(2)
    ]
    _write(root / "observation_event_bins.csv", _csv_line(event_header) + "".join(_csv_line(row) for row in event_rows))
    _write(root / "fig_uncertainty_motivation.png", "png")
    _write(root / "fig_uncertainty_motivation.pdf", "%PDF")
    _json(
        root / "observation_provenance.json",
        {
            "artifact_class": "paper_critical_observation_motivation",
            "status_label": "paper_critical_observation_ready",
            "paper_claim_scope": "motivation_only_not_main_table_sota",
            "git_commit": "abc123",
            "command": "python observation.py",
            "input_sha256": {"signals": "a"},
            "expected_candidates_per_event": 101,
            "join_report": [
                {"method": "ccrp", "join_rate": 1.0, "exact_event_match": True},
                {"method": "proex", "join_rate": 1.0, "exact_event_match": True},
            ],
            "figure_paths": ["fig_uncertainty_motivation.png", "fig_uncertainty_motivation.pdf"],
        },
    )


def _csv_line(values: list[object]) -> str:
    return ",".join(str(value) for value in values) + "\n"


def _metric_header(prefix: list[str]) -> list[str]:
    return prefix + list(METRICS)


def _metric_values(prefix: list[object]) -> list[object]:
    return prefix + [METRICS[key] for key in METRICS]


def test_observation_package_audit_accepts_complete_package(tmp_path):
    _complete_observation_package(tmp_path)

    audit = build_audit(module="observation_motivation", package_dir=tmp_path, expected_events=2)

    assert audit["ok"] is True
    assert audit["paper_claim_ready"] is True


def test_observation_package_audit_requires_seed_record(tmp_path):
    _complete_observation_package(tmp_path)
    _json(tmp_path / "run_config.json", {"expected_users": 2})

    audit = build_audit(module="observation_motivation", package_dir=tmp_path, expected_events=2)

    assert audit["ok"] is False
    assert "missing_seed_record" in audit["failures"]


def test_observation_package_audit_rejects_failure_log_markers(tmp_path):
    _complete_observation_package(tmp_path)
    _write(tmp_path / "log_snippets.md", "Traceback (most recent call last):\n")

    audit = build_audit(module="observation_motivation", package_dir=tmp_path, expected_events=2)

    assert audit["ok"] is False
    assert "log_snippet_contains_failure_marker:log_snippets.md:Traceback" in audit["failures"]


def test_observation_package_audit_rejects_nested_bulk_predictions(tmp_path):
    _complete_observation_package(tmp_path)
    _write(tmp_path / "predictions" / "rank_predictions.jsonl", "{}\n")

    audit = build_audit(module="observation_motivation", package_dir=tmp_path, expected_events=2)

    assert audit["ok"] is False
    assert "disallowed_bulk_prediction_jsonl:predictions/rank_predictions.jsonl" in audit["failures"]


def test_observation_package_audit_rejects_nonfinite_metrics(tmp_path):
    _complete_observation_package(tmp_path)
    text = (tmp_path / "observation_summary.csv").read_text(encoding="utf-8")
    (tmp_path / "observation_summary.csv").write_text(text.replace(",0.2,", ",nan,", 1), encoding="utf-8")

    audit = build_audit(module="observation_motivation", package_dir=tmp_path, expected_events=2)

    assert audit["ok"] is False
    assert any(failure.startswith("observation_summary:nonfinite_metric:") for failure in audit["failures"])


def _complete_component_ablation_package(root: Path, *, ablations: tuple[str, ...] = ABLATIONS) -> None:
    _general_files(root)
    header = _metric_header(
        ["domain", "ablation", "n_events", "status_label", "selected_on_test", "audit_ok", "degeneracy_audit_ok", "score_coverage_rate"]
    )
    rows = [
        _metric_values(["sports", ablation, 2, "same_schema_internal_ablation", "false", "true", "true", 1.0])
        for ablation in ablations
    ]
    _write(root / "component_ablation_summary.csv", _csv_line(header) + "".join(_csv_line(row) for row in rows))
    _write(
        root / "selected_test_metrics.csv",
        _csv_line(_metric_header(["domain", "split", "audit_ok", "degeneracy_audit_ok", "score_coverage_rate", "candidate_key_count"]))
        + _csv_line(_metric_values(["sports", "test", "true", "true", 1.0, 202])),
    )
    _write(
        root / "valid_ccrp_sweep.csv",
        "ablation,audit_ok,degeneracy_audit_ok,NDCG@10\n"
        + "".join(f"{ablation},true,true,0.1\n" for ablation in ABLATIONS),
    )
    _json(root / "selected_valid_config.json", {"score_mode": "full", "eta": 1.0})
    _json(
        root / "ccrp_internal_provenance.json",
        {"status_label": "same_schema_internal_ablation", "audit_ok": True, "score_coverage_rate": 1.0},
    )
    _write(root / "tables" / "ranking_metrics.csv", "baseline_name,MRR,HR@5,HR@10,HR@20,NDCG@5,NDCG@10,NDCG@20\nccrp,0.2,0.3,0.4,0.5,0.21,0.31,0.41\n")
    _write(root / "tables" / "external_score_coverage.csv", "baseline_name,ranking_events,total_candidates,matched_candidates,score_coverage_rate\nccrp,2,202,202,1.0\n")
    _write(root / "tables" / "same_candidate_external_baseline_summary.csv", "baseline_name,status_label\nccrp,same_schema_internal_ablation\n")
    _write(root / "tables" / "ranking_eval_records.csv", "source_event_id,positive_rank\n1,1\n2,2\n")
    _write(root / "fig_component_ablation.png", "png")
    _write(root / "fig_component_ablation.pdf", "%PDF")
    _json(
        root / "component_ablation_provenance.json",
        {
            "artifact_class": "paper_critical_component_ablation",
            "status_label": "paper_critical_component_ablation_ready",
            "git_commit": "abc123",
            "command": "python ablation.py",
            "input_sha256": {"valid_signal": "a", "test_signal": "b"},
            "figure_paths": ["fig_component_ablation.png", "fig_component_ablation.pdf"],
        },
    )


def test_component_ablation_package_audit_requires_every_ablation(tmp_path):
    _complete_component_ablation_package(tmp_path, ablations=ABLATIONS[:-1])

    audit = build_audit(module="component_ablation", package_dir=tmp_path, expected_events=2)

    assert audit["ok"] is False
    assert "component_ablation_summary_missing:without_risk_penalty" in audit["failures"]


def test_component_ablation_package_audit_accepts_complete_package(tmp_path):
    _complete_component_ablation_package(tmp_path)

    audit = build_audit(module="component_ablation", package_dir=tmp_path, expected_events=2)

    assert audit["ok"] is True
    assert audit["paper_claim_ready"] is True


def test_component_ablation_package_audit_rejects_bad_ranking_eval_count(tmp_path):
    _complete_component_ablation_package(tmp_path)
    _write(tmp_path / "tables" / "ranking_eval_records.csv", "source_event_id,positive_rank\n1,1\n")

    audit = build_audit(module="component_ablation", package_dir=tmp_path, expected_events=2)

    assert audit["ok"] is False
    assert "ranking_eval_records_row_count:1!=2" in audit["failures"]


def test_component_ablation_package_audit_rejects_bad_coverage_totals(tmp_path):
    _complete_component_ablation_package(tmp_path)
    _write(
        tmp_path / "tables" / "external_score_coverage.csv",
        "baseline_name,ranking_events,total_candidates,matched_candidates,score_coverage_rate\nccrp,1,101,101,1.0\n",
    )

    audit = build_audit(module="component_ablation", package_dir=tmp_path, expected_events=2)

    assert audit["ok"] is False
    assert "external_score_coverage_ranking_events:ccrp:1!=2" in audit["failures"]
    assert "external_score_coverage_total_candidates:ccrp:101!=202" in audit["failures"]
    assert "external_score_coverage_matched_candidates:ccrp:101!=202" in audit["failures"]


def test_hyperparameter_package_audit_accepts_valid_and_test_package(tmp_path):
    _general_files(tmp_path)
    rows = []
    for split in ("valid", "test"):
        for control in ("eta", "confidence_weight", "weight_grid_label"):
            for value in ("0.5", "1.0", "2.0"):
                rows.append(f"{split},{control},{value},NDCG@10,0.2\n")
    _write(
        tmp_path / "ccrp_hyperparameter_curve_summary.csv",
        "split,control,control_value,metric_name,metric_value\n"
        + "".join(rows),
    )
    figures = []
    for stem in ("fig_hyper_eta_curve", "fig_hyper_confidence_weight_curve", "fig_hyper_weight_simplex_or_lines"):
        for suffix, body in (("png", "png"), ("pdf", "%PDF")):
            name = f"{stem}.{suffix}"
            figures.append(name)
            _write(tmp_path / name, body)
    _json(
        tmp_path / "ccrp_hyperparameter_curve_provenance.json",
        {
            "artifact_class": "paper_critical_hyperparameter_analysis",
            "status_label": "paper_critical_hyperparameter_curve_ready",
            "paper_claim_scope": "valid_and_test_stability_curve_candidate",
            "reporting_mode": "valid_and_test",
            "git_commit": "abc123",
            "command": "python hyper.py",
            "sweep_sha256": "a",
            "test_sweep_sha256": "b",
            "controls": ["eta", "confidence_weight", "weight_grid_label"],
            "filters": {"score_mode": "full", "ablation": "full"},
            "audit_summary": {
                "require_audit_ok": True,
                "missing_audit_columns": [],
                "audited_rows": 18,
                "dropped_audit_rows": 0,
            },
            "control_reports": [
                {"split": split, "control": control, "curve_values": 3, "meets_min_values": True}
                for split in ("valid", "test")
                for control in ("eta", "confidence_weight", "weight_grid_label")
            ],
            "figure_paths": figures,
        },
    )

    audit = build_audit(module="hyperparameter_analysis", package_dir=tmp_path)

    assert audit["ok"] is True
    assert audit["paper_claim_ready"] is True


def test_hyperparameter_package_audit_requires_expected_controls(tmp_path):
    _general_files(tmp_path)
    _write(
        tmp_path / "ccrp_hyperparameter_curve_summary.csv",
        "split,control,control_value,metric_name,metric_value\n"
        "valid,eta,0.5,NDCG@10,0.2\n"
        "valid,eta,1.0,NDCG@10,0.3\n"
        "valid,eta,2.0,NDCG@10,0.25\n"
        "test,eta,0.5,NDCG@10,0.19\n"
        "test,eta,1.0,NDCG@10,0.29\n"
        "test,eta,2.0,NDCG@10,0.24\n",
    )
    _write(tmp_path / "fig_hyper_eta_curve.png", "png")
    _write(tmp_path / "fig_hyper_eta_curve.pdf", "%PDF")
    _json(
        tmp_path / "ccrp_hyperparameter_curve_provenance.json",
        {
            "artifact_class": "paper_critical_hyperparameter_analysis",
            "status_label": "paper_critical_hyperparameter_curve_ready",
            "paper_claim_scope": "valid_and_test_stability_curve_candidate",
            "reporting_mode": "valid_and_test",
            "git_commit": "abc123",
            "command": "python hyper.py",
            "sweep_sha256": "a",
            "test_sweep_sha256": "b",
            "controls": ["eta"],
            "filters": {"score_mode": "full", "ablation": "full"},
            "audit_summary": {
                "require_audit_ok": True,
                "missing_audit_columns": [],
                "audited_rows": 6,
                "dropped_audit_rows": 0,
            },
            "control_reports": [
                {"split": "valid", "control": "eta", "curve_values": 3, "meets_min_values": True},
                {"split": "test", "control": "eta", "curve_values": 3, "meets_min_values": True},
            ],
            "figure_paths": ["fig_hyper_eta_curve.png", "fig_hyper_eta_curve.pdf"],
        },
    )

    audit = build_audit(module="hyperparameter_analysis", package_dir=tmp_path)

    assert audit["ok"] is False
    assert "hyperparameter_missing_expected_control:confidence_weight" in audit["failures"]
    assert "hyperparameter_missing_expected_control:weight_grid_label" in audit["failures"]


def test_hyperparameter_package_audit_rejects_out_of_range_metric_values(tmp_path):
    _general_files(tmp_path)
    _write(
        tmp_path / "ccrp_hyperparameter_curve_summary.csv",
        "split,control,control_value,metric_name,metric_value\n"
        "valid,eta,0.5,NDCG@10,1.5\n"
        "valid,eta,1.0,NDCG@10,0.3\n"
        "valid,eta,2.0,NDCG@10,0.25\n"
        "test,eta,0.5,NDCG@10,0.19\n"
        "test,eta,1.0,NDCG@10,0.29\n"
        "test,eta,2.0,NDCG@10,0.24\n",
    )
    _write(tmp_path / "fig_hyper_eta_curve.png", "png")
    _write(tmp_path / "fig_hyper_eta_curve.pdf", "%PDF")
    _json(
        tmp_path / "ccrp_hyperparameter_curve_provenance.json",
        {
            "artifact_class": "paper_critical_hyperparameter_analysis",
            "status_label": "paper_critical_hyperparameter_curve_ready",
            "paper_claim_scope": "valid_and_test_stability_curve_candidate",
            "reporting_mode": "valid_and_test",
            "git_commit": "abc123",
            "command": "python hyper.py",
            "sweep_sha256": "a",
            "test_sweep_sha256": "b",
            "controls": ["eta"],
            "filters": {"score_mode": "full", "ablation": "full"},
            "audit_summary": {
                "require_audit_ok": True,
                "missing_audit_columns": [],
                "audited_rows": 6,
                "dropped_audit_rows": 0,
            },
            "control_reports": [
                {"split": "valid", "control": "eta", "curve_values": 3, "meets_min_values": True},
                {"split": "test", "control": "eta", "curve_values": 3, "meets_min_values": True},
            ],
            "figure_paths": ["fig_hyper_eta_curve.png", "fig_hyper_eta_curve.pdf"],
        },
    )

    audit = build_audit(
        module="hyperparameter_analysis",
        package_dir=tmp_path,
        expected_controls=("eta",),
    )

    assert audit["ok"] is False
    assert "hyperparameter_summary:metric_out_of_range:0:metric_value:1.5" in audit["failures"]
