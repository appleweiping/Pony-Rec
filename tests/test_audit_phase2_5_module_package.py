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
TEST_SHA256 = "a" * 64
TEST_SHA256_B = "b" * 64
DEFAULT_KEY_COUNT = 10000 * 101


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _json(path: Path, payload: dict) -> Path:
    return _write(path, json.dumps(payload, indent=2) + "\n")


def _stability_report(controls: tuple[str, ...] = ("eta", "weight_grid_label")) -> list[dict[str, object]]:
    return [
        {
            "control": control,
            "metric": "NDCG@10",
            "relative_drop_tolerance": 0.05,
            "has_valid_and_test": True,
            "valid_best_value": "0.5",
            "test_best_value": "0.5",
            "valid_metric_at_best": 0.2,
            "test_best_metric": 0.2,
            "test_metric_at_valid_best": 0.2,
            "relative_drop_from_test_best": 0.0,
            "test_rank_of_valid_best": 1,
            "best_value_match": True,
            "stable_within_tolerance": True,
            "reason": "within_tolerance",
        }
        for control in controls
    ]


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
                    "local_sha256": TEST_SHA256,
                    "server_sha256": TEST_SHA256,
                    "files": {
                        "tables/ranking_metrics.csv": {
                            "ok": True,
                            "local_sha256": TEST_SHA256,
                            "server_sha256": TEST_SHA256,
                        }
                    },
                }
            ],
        },
    )


def _write_manifest_checks(root: Path, files: tuple[str, ...]) -> None:
    manifest_checks = {
        name: {
            "ok": True,
            "local_sha256": TEST_SHA256,
            "server_sha256": TEST_SHA256,
            "expected": TEST_SHA256,
            "actual": TEST_SHA256,
            "local_size": 1,
            "server_size": 1,
        }
        for name in files
    }
    _json(
        root / "local_server_manifest_comparison.json",
        {
            "ok": True,
            "comparison_scope": "server_to_local_lightweight_sync",
            "manifest_checks": manifest_checks,
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
        ["event_id", "method", "uncertainty_bin_index", "uncertainty_bin", "candidate_rows", "positive_rank", "num_candidates"]
    )
    event_rows = [
        _metric_values([f"e{idx}", method, idx % 2, "low" if idx % 2 == 0 else "high", 101, idx + 1, 101])
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
            "expected_uncertainty_rows_per_event": 101,
            "uncertainty_summary": {
                "uncertainty_input_rows": 202,
                "finite_uncertainty_rows": 202,
                "invalid_uncertainty_rows": 0,
                "event_count": 2,
                "expected_events": 2,
                "expected_uncertainty_rows_per_event": 101,
                "expected_finite_uncertainty_rows": 202,
                "candidate_rows_min": 101,
                "candidate_rows_max": 101,
                "candidate_rows_bad_event_count": 0,
            },
            "join_report": [
                {"method": "ccrp", "join_rate": 1.0, "exact_event_match": True},
                {"method": "proex", "join_rate": 1.0, "exact_event_match": True},
            ],
            "figure_paths": ["fig_uncertainty_motivation.png", "fig_uncertainty_motivation.pdf"],
        },
    )
    _json(
        root / "same_candidate_alignment.json",
        {
            "ok": True,
            "status_label": "same_candidate_alignment_evidence",
            "expected_events": 2,
            "expected_candidates_per_event": 101,
            "expected_candidate_key_count": 202,
            "candidate_key_count": 202,
            "score_key_count": 202,
            "score_coverage_rate": 1.0,
            "missing_score_keys": 0,
            "extra_score_keys": 0,
            "duplicate_score_keys": 0,
            "invalid_scores": 0,
            "blank_score_keys": 0,
            "test_candidate_items_sha256": TEST_SHA256,
            "source_provenance_sha256": TEST_SHA256,
            "score_sha256": TEST_SHA256,
            "method_eval_rows": {"ccrp": 2, "proex": 2},
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


def test_observation_package_audit_requires_uncertainty_row_coverage(tmp_path):
    _complete_observation_package(tmp_path)
    provenance = json.loads((tmp_path / "observation_provenance.json").read_text(encoding="utf-8"))
    provenance["uncertainty_summary"]["candidate_rows_max"] = 100
    provenance["uncertainty_summary"]["finite_uncertainty_rows"] = 201
    _json(tmp_path / "observation_provenance.json", provenance)

    audit = build_audit(module="observation_motivation", package_dir=tmp_path, expected_events=2)

    assert audit["ok"] is False
    assert "observation_uncertainty_summary_mismatch:candidate_rows_max:100!=101" in audit["failures"]
    assert "observation_uncertainty_summary_mismatch:finite_uncertainty_rows:201!=202" in audit["failures"]


def test_observation_package_audit_requires_event_bin_candidate_rows(tmp_path):
    _complete_observation_package(tmp_path)
    text = (tmp_path / "observation_event_bins.csv").read_text(encoding="utf-8")
    (tmp_path / "observation_event_bins.csv").write_text(text.replace(",101,1,101,", ",100,1,101,", 1), encoding="utf-8")

    audit = build_audit(module="observation_motivation", package_dir=tmp_path, expected_events=2)

    assert audit["ok"] is False
    assert any(failure.startswith("observation_candidate_rows_mismatch:") for failure in audit["failures"])


def test_observation_package_audit_requires_same_candidate_alignment(tmp_path):
    _complete_observation_package(tmp_path)
    (tmp_path / "same_candidate_alignment.json").unlink()

    audit = build_audit(module="observation_motivation", package_dir=tmp_path, expected_events=2)

    assert audit["ok"] is False
    assert "missing_required_file:same_candidate_alignment.json" in audit["failures"]


def test_observation_package_audit_rejects_bad_same_candidate_alignment(tmp_path):
    _complete_observation_package(tmp_path)
    alignment = json.loads((tmp_path / "same_candidate_alignment.json").read_text(encoding="utf-8"))
    alignment["score_coverage_rate"] = 0.99
    alignment["candidate_key_count"] = 201
    _json(tmp_path / "same_candidate_alignment.json", alignment)

    audit = build_audit(module="observation_motivation", package_dir=tmp_path, expected_events=2)

    assert audit["ok"] is False
    assert "observation_same_candidate_alignment_mismatch:candidate_key_count:201!=202" in audit["failures"]
    assert "observation_same_candidate_alignment_score_coverage:0.99!=1.0" in audit["failures"]


def test_observation_package_audit_rejects_vague_manifest_comparison(tmp_path):
    _complete_observation_package(tmp_path)
    _json(
        tmp_path / "local_server_manifest_comparison.json",
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

    audit = build_audit(module="observation_motivation", package_dir=tmp_path, expected_events=2)

    assert audit["ok"] is False
    assert "local_server_manifest_comparison_lacks_evidence:local_server_manifest_comparison.json" in audit["failures"]


def test_observation_package_audit_accepts_manifest_check_hash_equality(tmp_path):
    _complete_observation_package(tmp_path)
    _json(
        tmp_path / "local_server_manifest_comparison.json",
        {
            "ok": True,
            "manifest_checks": {
                "ranking_metrics.csv": {
                    "expected": TEST_SHA256,
                    "actual": TEST_SHA256,
                    "ok": True,
                }
            },
        },
    )

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
            "ok": True,
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


def test_component_ablation_package_audit_rejects_failed_builder_provenance(tmp_path):
    _complete_component_ablation_package(tmp_path)
    provenance = json.loads((tmp_path / "component_ablation_provenance.json").read_text(encoding="utf-8"))
    provenance["ok"] = False
    provenance["failures"] = ["valid_sweep_missing_main_ablation:full"]
    _json(tmp_path / "component_ablation_provenance.json", provenance)

    audit = build_audit(module="component_ablation", package_dir=tmp_path, expected_events=2)

    assert audit["ok"] is False
    assert "component_ablation:provenance_not_ok" in audit["failures"]


def test_component_ablation_package_audit_allows_preregistered_sweep_full_only(tmp_path):
    _complete_component_ablation_package(tmp_path)
    _write(
        tmp_path / "valid_ccrp_sweep.csv",
        "ablation,audit_ok,degeneracy_audit_ok,NDCG@10\n"
        "full,true,true,0.1\n",
    )

    audit = build_audit(module="component_ablation", package_dir=tmp_path, expected_events=2)

    assert audit["ok"] is True
    assert audit["paper_claim_ready"] is True


def test_component_ablation_package_audit_rejects_main_config_mismatch(tmp_path):
    _complete_component_ablation_package(tmp_path)
    text = (tmp_path / "component_ablation_summary.csv").read_text(encoding="utf-8")
    lines = text.splitlines()
    header = lines[0].split(",")
    insert_at = header.index("score_coverage_rate") + 1
    header.insert(insert_at, "eta")
    rows = []
    for line in lines[1:]:
        values = line.split(",")
        values.insert(insert_at, "0.5")
        rows.append(",".join(values))
    _write(tmp_path / "component_ablation_summary.csv", ",".join(header) + "\n" + "\n".join(rows) + "\n")
    _json(
        tmp_path / "ccrp_internal_provenance.json",
        {
            "status_label": "same_schema_internal_ablation",
            "audit_ok": True,
            "score_coverage_rate": 1.0,
            "eta": 1.0,
        },
    )

    audit = build_audit(module="component_ablation", package_dir=tmp_path, expected_events=2)

    assert audit["ok"] is False
    assert any(failure.startswith("component_ablation_config_mismatch:") for failure in audit["failures"])


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


def _complete_hyperparameter_package(root: Path) -> None:
    _general_files(root)
    required_manifest_files = (
        "valid_ccrp_hyperparameter_sweep.csv",
        "test_ccrp_hyperparameter_sweep.csv",
        "ccrp_hyperparameter_sweep_provenance.json",
        "ccrp_hyperparameter_curve_summary.csv",
        "ccrp_hyperparameter_curve_provenance.json",
        "fig_hyper_eta_curve.png",
        "fig_hyper_eta_curve.pdf",
        "fig_hyper_weight_simplex_or_lines.png",
        "fig_hyper_weight_simplex_or_lines.pdf",
        "run_config.json",
        "log_snippets.md",
    )
    _write(root / "valid_ccrp_hyperparameter_sweep.csv", "split,control\nvalid,eta\n")
    _write(root / "test_ccrp_hyperparameter_sweep.csv", "split,control\ntest,eta\n")
    _json(root / "ccrp_hyperparameter_sweep_provenance.json", {"ok": True})
    rows = []
    for split in ("valid", "test"):
        for control in ("eta", "weight_grid_label"):
            for value in ("0.5", "1.0", "2.0"):
                rows.append(
                    f"{split},main_control,{control},{value},full,NDCG@10,0.2,true,true,1.0,{DEFAULT_KEY_COUNT},1\n"
                )
    _write(
        root / "ccrp_hyperparameter_curve_summary.csv",
        "split,row_kind,control,control_value,score_mode,metric_name,metric_value,audit_ok,degeneracy_audit_ok,score_coverage_rate,candidate_key_count,candidate_rows_for_value\n"
        + "".join(rows),
    )
    figures = []
    for stem in ("fig_hyper_eta_curve", "fig_hyper_weight_simplex_or_lines"):
        for suffix, body in (("png", "png"), ("pdf", "%PDF")):
            name = f"{stem}.{suffix}"
            figures.append(name)
            _write(root / name, body)
    _json(
        root / "ccrp_hyperparameter_curve_provenance.json",
        {
            "artifact_class": "paper_critical_hyperparameter_analysis",
            "status_label": "paper_critical_hyperparameter_curve_ready",
            "paper_claim_scope": "valid_and_test_stability_curve_candidate",
            "reporting_mode": "valid_and_test",
            "git_commit": "abc123",
            "command": "python hyper.py",
            "sweep_sha256": TEST_SHA256,
            "test_sweep_sha256": TEST_SHA256_B,
            "test_not_used_for_selection": True,
            "metric": "NDCG@10",
            "min_values": 3,
            "controls": ["eta", "weight_grid_label"],
            "filters": {"score_mode": "full", "ablation": "full"},
            "audit_summary": {
                "require_audit_ok": True,
                "missing_audit_columns": [],
                "audited_rows": 12,
                "dropped_audit_rows": 0,
            },
            "control_reports": [
                {"split": split, "control": control, "curve_values": 3, "meets_min_values": True}
                for split in ("valid", "test")
                for control in ("eta", "weight_grid_label")
            ],
            "stability_report": _stability_report(),
            "sweep_source_provenance": {
                "status_label": "valid_test_saved_signal_hyperparameter_sweep_ready",
                "test_not_used_for_selection": True,
                "main_controls": ["eta", "weight_grid_label"],
                "eta_grid": [0.5, 1.0, 2.0],
                "weight_grid": ["0.5,0.3,0.2", "0.7,0.2,0.1", "0.4,0.4,0.2"],
                "expected_candidate_key_count": DEFAULT_KEY_COUNT,
                "tie_break_seed": 20260607,
                "row_counts": {
                    "valid_signal_rows": DEFAULT_KEY_COUNT,
                    "test_signal_rows": DEFAULT_KEY_COUNT,
                    "valid_candidate_rows": DEFAULT_KEY_COUNT,
                    "test_candidate_rows": DEFAULT_KEY_COUNT,
                    "valid_ranking_events": 10000,
                    "test_ranking_events": 10000,
                },
                "cleanup_status": {
                    "retained_bulk_scores_csv": False,
                    "retained_prediction_jsonl": False,
                    "retained_scored_temp_rows": False,
                    "retained_checkpoints": False,
                },
            },
            "figure_paths": figures,
        },
    )
    _write_manifest_checks(root, required_manifest_files)


def test_hyperparameter_package_audit_accepts_valid_and_test_package(tmp_path):
    _complete_hyperparameter_package(tmp_path)

    audit = build_audit(module="hyperparameter_analysis", package_dir=tmp_path)

    assert audit["ok"] is True
    assert audit["paper_claim_ready"] is True


def test_hyperparameter_package_audit_requires_manifest_coverage(tmp_path):
    _complete_hyperparameter_package(tmp_path)
    path = tmp_path / "local_server_manifest_comparison.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["manifest_checks"].pop("run_config.json")
    _json(path, payload)

    audit = build_audit(module="hyperparameter_analysis", package_dir=tmp_path)

    assert audit["ok"] is False
    assert "local_server_manifest_missing_required_file:run_config.json" in audit["failures"]


def test_hyperparameter_package_audit_requires_sweep_source_provenance(tmp_path):
    _complete_hyperparameter_package(tmp_path)
    path = tmp_path / "ccrp_hyperparameter_curve_provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance.pop("sweep_source_provenance")
    _json(path, provenance)

    audit = build_audit(module="hyperparameter_analysis", package_dir=tmp_path)

    assert audit["ok"] is False
    assert "hyperparameter:missing_sweep_source_provenance" in audit["failures"]


def test_hyperparameter_package_audit_rejects_retained_bulk_sweep_outputs(tmp_path):
    _complete_hyperparameter_package(tmp_path)
    path = tmp_path / "ccrp_hyperparameter_curve_provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance["sweep_source_provenance"]["cleanup_status"]["retained_scored_temp_rows"] = True
    _json(path, provenance)

    audit = build_audit(module="hyperparameter_analysis", package_dir=tmp_path)

    assert audit["ok"] is False
    assert "hyperparameter_cleanup_retained_bulk:retained_scored_temp_rows:True" in audit["failures"]


def test_hyperparameter_package_audit_rejects_confidence_weight_as_full_mode(tmp_path):
    _complete_hyperparameter_package(tmp_path)
    path = tmp_path / "ccrp_hyperparameter_curve_summary.csv"
    path.write_text(
        path.read_text(encoding="utf-8")
        + f"test,diagnostic_control,confidence_weight,0.7,full,NDCG@10,0.2,true,true,1.0,{DEFAULT_KEY_COUNT},1\n",
        encoding="utf-8",
    )

    audit = build_audit(module="hyperparameter_analysis", package_dir=tmp_path)

    assert audit["ok"] is False
    assert "hyperparameter_confidence_weight_full_mode:12" in audit["failures"]


def test_hyperparameter_package_audit_rejects_missing_row_audit_columns(tmp_path):
    _complete_hyperparameter_package(tmp_path)
    rows = []
    for split in ("valid", "test"):
        for control in ("eta", "weight_grid_label"):
            for value in ("0.5", "1.0", "2.0"):
                rows.append(f"{split},{control},{value},NDCG@10,0.2\n")
    _write(
        tmp_path / "ccrp_hyperparameter_curve_summary.csv",
        "split,control,control_value,metric_name,metric_value\n"
        + "".join(rows),
    )

    audit = build_audit(module="hyperparameter_analysis", package_dir=tmp_path)

    assert audit["ok"] is False
    assert "hyperparameter_summary_missing_audit_column:audit_ok" in audit["failures"]
    assert "hyperparameter_summary_missing_audit_column:score_coverage_rate" in audit["failures"]


def test_hyperparameter_package_audit_rejects_false_row_audit_and_coverage(tmp_path):
    _complete_hyperparameter_package(tmp_path)
    text = (tmp_path / "ccrp_hyperparameter_curve_summary.csv").read_text(encoding="utf-8")
    text = text.replace(f"0.2,true,true,1.0,{DEFAULT_KEY_COUNT}", f"0.2,false,false,0.5,{DEFAULT_KEY_COUNT - 1}", 1)
    (tmp_path / "ccrp_hyperparameter_curve_summary.csv").write_text(text, encoding="utf-8")

    audit = build_audit(module="hyperparameter_analysis", package_dir=tmp_path)

    assert audit["ok"] is False
    assert "hyperparameter_summary_audit_false:0:audit_ok" in audit["failures"]
    assert "hyperparameter_summary_audit_false:0:degeneracy_audit_ok" in audit["failures"]
    assert "hyperparameter_summary_score_coverage_not_one:0:0.5" in audit["failures"]
    assert f"hyperparameter_summary_candidate_key_count:0:{DEFAULT_KEY_COUNT - 1}!={DEFAULT_KEY_COUNT}" in audit["failures"]


def test_hyperparameter_package_audit_rejects_same_valid_test_sweep_hash(tmp_path):
    _complete_hyperparameter_package(tmp_path)
    path = tmp_path / "ccrp_hyperparameter_curve_provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance["test_sweep_sha256"] = provenance["sweep_sha256"]
    _json(path, provenance)

    audit = build_audit(module="hyperparameter_analysis", package_dir=tmp_path)

    assert audit["ok"] is False
    assert "hyperparameter:valid_test_sweep_hash_equal" in audit["failures"]


def test_hyperparameter_package_audit_rejects_missing_valid_sweep_hash(tmp_path):
    _complete_hyperparameter_package(tmp_path)
    path = tmp_path / "ccrp_hyperparameter_curve_provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance["sweep_sha256"] = ""
    _json(path, provenance)

    audit = build_audit(module="hyperparameter_analysis", package_dir=tmp_path)

    assert audit["ok"] is False
    assert "hyperparameter:missing_sweep_sha256" in audit["failures"]


def test_hyperparameter_package_audit_rejects_missing_stability_report(tmp_path):
    _complete_hyperparameter_package(tmp_path)
    path = tmp_path / "ccrp_hyperparameter_curve_provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance.pop("stability_report")
    _json(path, provenance)

    audit = build_audit(module="hyperparameter_analysis", package_dir=tmp_path)

    assert audit["ok"] is False
    assert "hyperparameter:missing_stability_report" in audit["failures"]


def test_hyperparameter_package_audit_rejects_unstable_valid_test_report(tmp_path):
    _complete_hyperparameter_package(tmp_path)
    path = tmp_path / "ccrp_hyperparameter_curve_provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance["stability_report"][0]["stable_within_tolerance"] = False
    provenance["stability_report"][0]["relative_drop_from_test_best"] = 0.2
    _json(path, provenance)

    audit = build_audit(module="hyperparameter_analysis", package_dir=tmp_path)

    assert audit["ok"] is False
    assert "hyperparameter_stability_report_not_stable:eta:0.2" in audit["failures"]
    assert "hyperparameter_stability_drop_exceeds_tolerance:eta:0.2>0.05" in audit["failures"]


def test_hyperparameter_package_audit_rejects_stability_report_mismatch(tmp_path):
    _complete_hyperparameter_package(tmp_path)
    path = tmp_path / "ccrp_hyperparameter_curve_provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance["stability_report"][0]["valid_best_value"] = "2.0"
    provenance["stability_report"][0]["test_rank_of_valid_best"] = 3
    _json(path, provenance)

    audit = build_audit(module="hyperparameter_analysis", package_dir=tmp_path)

    assert audit["ok"] is False
    assert "hyperparameter_stability_report_mismatch:eta:valid_best_value:2.0!=0.5" in audit["failures"]
    assert "hyperparameter_stability_report_mismatch:eta:test_rank_of_valid_best:3!=1" in audit["failures"]


def test_hyperparameter_package_audit_rejects_duplicate_and_extra_stability_controls(tmp_path):
    _complete_hyperparameter_package(tmp_path)
    path = tmp_path / "ccrp_hyperparameter_curve_provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance["stability_report"].append(dict(provenance["stability_report"][0]))
    extra = dict(provenance["stability_report"][0])
    extra["control"] = "temperature"
    provenance["stability_report"].append(extra)
    _json(path, provenance)

    audit = build_audit(module="hyperparameter_analysis", package_dir=tmp_path)

    assert audit["ok"] is False
    assert "hyperparameter_stability_duplicate_control:eta" in audit["failures"]
    assert "hyperparameter_stability_unexpected_control:temperature" in audit["failures"]


def test_hyperparameter_package_audit_rejects_test_best_of_many_curve_points(tmp_path):
    _complete_hyperparameter_package(tmp_path)
    text = (tmp_path / "ccrp_hyperparameter_curve_summary.csv").read_text(encoding="utf-8")
    text = text.replace(
        f"test,main_control,eta,0.5,full,NDCG@10,0.2,true,true,1.0,{DEFAULT_KEY_COUNT},1",
        f"test,main_control,eta,0.5,full,NDCG@10,0.2,true,true,1.0,{DEFAULT_KEY_COUNT},2",
        1,
    )
    (tmp_path / "ccrp_hyperparameter_curve_summary.csv").write_text(text, encoding="utf-8")

    audit = build_audit(module="hyperparameter_analysis", package_dir=tmp_path)

    assert audit["ok"] is False
    assert "hyperparameter_summary_candidate_rows_for_value:6:2!=1" in audit["failures"]


def test_hyperparameter_package_audit_rejects_duplicate_curve_points(tmp_path):
    _complete_hyperparameter_package(tmp_path)
    path = tmp_path / "ccrp_hyperparameter_curve_summary.csv"
    text = path.read_text(encoding="utf-8")
    duplicate = f"valid,main_control,eta,0.5,full,NDCG@10,0.2,true,true,1.0,{DEFAULT_KEY_COUNT},1\n"
    path.write_text(text + duplicate, encoding="utf-8")

    audit = build_audit(module="hyperparameter_analysis", package_dir=tmp_path)

    assert audit["ok"] is False
    assert "hyperparameter_summary_duplicate_curve_point:valid:eta:0.5" in audit["failures"]
    assert "hyperparameter_control_report_value_mismatch:valid:eta:3!=3" not in audit["failures"]


def test_hyperparameter_package_audit_rejects_unknown_metric_name(tmp_path):
    _complete_hyperparameter_package(tmp_path)
    path = tmp_path / "ccrp_hyperparameter_curve_provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance["metric"] = "AUC"
    _json(path, provenance)

    audit = build_audit(module="hyperparameter_analysis", package_dir=tmp_path)

    assert audit["ok"] is False
    assert "hyperparameter:unsupported_metric:AUC" in audit["failures"]


def test_hyperparameter_package_audit_rejects_summary_missing_test_rows(tmp_path):
    _complete_hyperparameter_package(tmp_path)
    rows = []
    for control in ("eta", "weight_grid_label"):
        for value in ("0.5", "1.0", "2.0"):
            rows.append(f"valid,{control},{value},NDCG@10,0.2,true,true,1.0,{DEFAULT_KEY_COUNT},1\n")
    _write(
        tmp_path / "ccrp_hyperparameter_curve_summary.csv",
        "split,control,control_value,metric_name,metric_value,audit_ok,degeneracy_audit_ok,score_coverage_rate,candidate_key_count,candidate_rows_for_value\n"
        + "".join(rows),
    )

    audit = build_audit(module="hyperparameter_analysis", package_dir=tmp_path)

    assert audit["ok"] is False
    assert "hyperparameter_summary_missing_split_control:test:eta" in audit["failures"]
    assert "hyperparameter_control_report_value_mismatch:test:eta:3!=0" in audit["failures"]


def test_hyperparameter_package_audit_rejects_too_short_summary_curve(tmp_path):
    _general_files(tmp_path)
    _write(
        tmp_path / "ccrp_hyperparameter_curve_summary.csv",
        "split,control,control_value,metric_name,metric_value,audit_ok,degeneracy_audit_ok,score_coverage_rate,candidate_key_count,candidate_rows_for_value\n"
        f"valid,eta,0.5,NDCG@10,0.2,true,true,1.0,{DEFAULT_KEY_COUNT},1\n"
        f"valid,eta,1.0,NDCG@10,0.3,true,true,1.0,{DEFAULT_KEY_COUNT},1\n"
        f"test,eta,0.5,NDCG@10,0.19,true,true,1.0,{DEFAULT_KEY_COUNT},1\n"
        f"test,eta,1.0,NDCG@10,0.29,true,true,1.0,{DEFAULT_KEY_COUNT},1\n",
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
            "sweep_sha256": TEST_SHA256,
            "test_sweep_sha256": TEST_SHA256_B,
            "metric": "NDCG@10",
            "min_values": 3,
            "controls": ["eta"],
            "filters": {"score_mode": "full", "ablation": "full"},
            "audit_summary": {
                "require_audit_ok": True,
                "missing_audit_columns": [],
                "audited_rows": 4,
                "dropped_audit_rows": 0,
            },
            "control_reports": [
                {"split": "valid", "control": "eta", "curve_values": 2, "meets_min_values": True},
                {"split": "test", "control": "eta", "curve_values": 2, "meets_min_values": True},
            ],
            "stability_report": _stability_report(("eta",)),
            "figure_paths": ["fig_hyper_eta_curve.png", "fig_hyper_eta_curve.pdf"],
        },
    )

    audit = build_audit(module="hyperparameter_analysis", package_dir=tmp_path, expected_controls=("eta",))

    assert audit["ok"] is False
    assert "hyperparameter_summary_control_too_short:valid:eta:2<3" in audit["failures"]
    assert "hyperparameter_summary_control_too_short:test:eta:2<3" in audit["failures"]


def test_hyperparameter_package_audit_requires_expected_controls(tmp_path):
    _general_files(tmp_path)
    _write(
        tmp_path / "ccrp_hyperparameter_curve_summary.csv",
        "split,control,control_value,metric_name,metric_value,audit_ok,degeneracy_audit_ok,score_coverage_rate,candidate_key_count,candidate_rows_for_value\n"
        f"valid,eta,0.5,NDCG@10,0.2,true,true,1.0,{DEFAULT_KEY_COUNT},1\n"
        f"valid,eta,1.0,NDCG@10,0.3,true,true,1.0,{DEFAULT_KEY_COUNT},1\n"
        f"valid,eta,2.0,NDCG@10,0.25,true,true,1.0,{DEFAULT_KEY_COUNT},1\n"
        f"test,eta,0.5,NDCG@10,0.19,true,true,1.0,{DEFAULT_KEY_COUNT},1\n"
        f"test,eta,1.0,NDCG@10,0.29,true,true,1.0,{DEFAULT_KEY_COUNT},1\n"
        f"test,eta,2.0,NDCG@10,0.24,true,true,1.0,{DEFAULT_KEY_COUNT},1\n",
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
            "sweep_sha256": TEST_SHA256,
            "test_sweep_sha256": TEST_SHA256_B,
            "metric": "NDCG@10",
            "min_values": 3,
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
            "stability_report": _stability_report(("eta",)),
            "figure_paths": ["fig_hyper_eta_curve.png", "fig_hyper_eta_curve.pdf"],
        },
    )

    audit = build_audit(module="hyperparameter_analysis", package_dir=tmp_path)

    assert audit["ok"] is False
    assert "hyperparameter_missing_expected_control:weight_grid_label" in audit["failures"]


def test_hyperparameter_package_audit_rejects_out_of_range_metric_values(tmp_path):
    _general_files(tmp_path)
    _write(
        tmp_path / "ccrp_hyperparameter_curve_summary.csv",
        "split,control,control_value,metric_name,metric_value,audit_ok,degeneracy_audit_ok,score_coverage_rate,candidate_key_count,candidate_rows_for_value\n"
        f"valid,eta,0.5,NDCG@10,1.5,true,true,1.0,{DEFAULT_KEY_COUNT},1\n"
        f"valid,eta,1.0,NDCG@10,0.3,true,true,1.0,{DEFAULT_KEY_COUNT},1\n"
        f"valid,eta,2.0,NDCG@10,0.25,true,true,1.0,{DEFAULT_KEY_COUNT},1\n"
        f"test,eta,0.5,NDCG@10,0.19,true,true,1.0,{DEFAULT_KEY_COUNT},1\n"
        f"test,eta,1.0,NDCG@10,0.29,true,true,1.0,{DEFAULT_KEY_COUNT},1\n"
        f"test,eta,2.0,NDCG@10,0.24,true,true,1.0,{DEFAULT_KEY_COUNT},1\n",
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
            "sweep_sha256": TEST_SHA256,
            "test_sweep_sha256": TEST_SHA256_B,
            "metric": "NDCG@10",
            "min_values": 3,
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
            "stability_report": _stability_report(("eta",)),
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
