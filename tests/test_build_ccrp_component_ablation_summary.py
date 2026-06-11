import json

from scripts.analysis.main_build_ccrp_component_ablation_summary import build_component_ablation_package
from scripts.audit.main_audit_phase2_5_module_package import build_audit


TEST_SHA256 = "a" * 64


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(path, payload):
    return _write(path, json.dumps(payload, indent=2) + "\n")


def _seed_selector_package(root):
    ranking = _write(
        root / "ranking_task.jsonl",
        '{"source_event_id":"e1","user_id":"u1","candidate_item_ids":["i1","i2"],"positive_item_id":"i1"}\n'
        '{"source_event_id":"e2","user_id":"u2","candidate_item_ids":["i3","i4"],"positive_item_id":"i4"}\n',
    )
    candidates = _write(
        root / "candidate_items.csv",
        "source_event_id,user_id,item_id\n"
        "e1,u1,i1\n"
        "e1,u1,i2\n"
        "e2,u2,i3\n"
        "e2,u2,i4\n",
    )
    signal = _write(
        root / "test_signal.csv",
        "source_event_id,user_id,item_id,relevance_probability,calibrated_confidence,evidence_support,counterevidence_strength\n"
        "e1,u1,i1,0.9,0.8,0.9,0.1\n"
        "e1,u1,i2,0.2,0.3,0.2,0.5\n"
        "e2,u2,i3,0.3,0.4,0.3,0.4\n"
        "e2,u2,i4,0.8,0.7,0.8,0.1\n",
    )
    _write_json(
        root / "selected_valid_config.json",
        {
            "domain": "toy",
            "split": "valid",
            "score_mode": "full",
            "ablation": "full",
            "eta": 1.0,
            "confidence_weight": 0.7,
            "weight_boundary": 0.5,
            "weight_calibration_gap": 0.3,
            "weight_evidence": 0.2,
        },
    )
    _write_json(
        root / "ccrp_internal_provenance.json",
        {
            "domain": "toy",
            "main_config_mode": "valid_selected",
            "selected_on": "valid",
            "selection_metric": "NDCG@10",
            "selected_valid_NDCG@10": 0.8,
            "score_mode": "full",
            "ablation": "full",
            "eta": 1.0,
            "confidence_weight": 0.7,
            "weight_boundary": 0.5,
            "weight_calibration_gap": 0.3,
            "weight_evidence": 0.2,
            "weight_grid_label": "0.5,0.3,0.2",
            "test_ranking_path": str(ranking),
            "test_candidate_items_path": str(candidates),
            "test_signal_path": str(signal),
        },
    )
    _write(
        root / "valid_ccrp_sweep.csv",
        "domain,split,score_mode,ablation,eta,confidence_weight,weight_grid_label,audit_ok,degeneracy_audit_ok,NDCG@10\n"
        "toy,valid,full,full,1.0,0.7,0.5;0.3;0.2,true,true,0.8\n"
        "toy,valid,full,without_risk_penalty,1.0,0.7,0.5;0.3;0.2,true,true,0.7\n",
    )
    return root


def _seed_preregistered_selector_package(root):
    _seed_selector_package(root)
    _write_json(
        root / "selected_valid_config.json",
        {
            "domain": "toy",
            "split": "valid",
            "score_mode": "full",
            "ablation": "full",
            "eta": 0.5,
            "confidence_weight": 0.7,
            "weight_boundary": 0.5,
            "weight_calibration_gap": 0.3,
            "weight_evidence": 0.2,
        },
    )
    provenance = json.loads((root / "ccrp_internal_provenance.json").read_text(encoding="utf-8"))
    provenance.update(
        {
            "main_config_mode": "preregistered",
            "selected_on": "preregistered",
            "eta": 1.0,
            "confidence_weight": 0.7,
            "weight_boundary": 0.5,
            "weight_calibration_gap": 0.3,
            "weight_evidence": 0.2,
            "weight_grid_label": "0.5,0.3,0.2",
        }
    )
    _write_json(root / "ccrp_internal_provenance.json", provenance)
    _write(
        root / "valid_ccrp_sweep.csv",
        "domain,split,score_mode,ablation,eta,confidence_weight,weight_grid_label,audit_ok,degeneracy_audit_ok,NDCG@10\n"
        "toy,valid,full,full,0.5,0.7,0.5;0.3;0.2,true,true,0.8\n",
    )
    return root


def _seed_package_audit_extras(root):
    _write(root / "log_snippets.md", "completed without hard failure markers\n")
    _write_json(root / "run_config.json", {"seed": 13})
    _write_json(
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
    _write(
        root / "selected_test_metrics.csv",
        "domain,split,audit_ok,degeneracy_audit_ok,score_coverage_rate,candidate_key_count,MRR,HR@5,HR@10,HR@20,NDCG@5,NDCG@10,NDCG@20\n"
        "toy,test,true,true,1.0,4,1.0,1.0,1.0,1.0,1.0,1.0,1.0\n",
    )
    _write(root / "tables" / "ranking_metrics.csv", "baseline_name,MRR,HR@5,HR@10,HR@20,NDCG@5,NDCG@10,NDCG@20\nccrp,1,1,1,1,1,1,1\n")
    _write(root / "tables" / "external_score_coverage.csv", "baseline_name,ranking_events,total_candidates,matched_candidates,score_coverage_rate\nccrp,2,4,4,1.0\n")
    _write(root / "tables" / "same_candidate_external_baseline_summary.csv", "baseline_name,status_label\nccrp,same_schema_internal_ablation\n")
    _write(root / "tables" / "ranking_eval_records.csv", "source_event_id,positive_rank\n1,1\n2,1\n")


def test_component_ablation_builder_uses_valid_selected_config_and_full_metrics(tmp_path):
    selector = _seed_selector_package(tmp_path)

    provenance = build_component_ablation_package(
        selector_dir=selector,
        ablations=["full", "without_risk_penalty"],
        expected_events=2,
        expected_candidates_per_event=2,
    )

    assert provenance["ok"] is True
    assert provenance["selection_source"]["selected_on"] == "valid"
    assert "component_ablation_summary.csv" in provenance["summary_path"]
    summary = (selector / "component_ablation_summary.csv").read_text(encoding="utf-8")
    for metric in ("HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR"):
        assert metric in summary
    assert "without_risk_penalty" in summary


def test_component_ablation_builder_uses_preregistered_main_config(tmp_path):
    selector = _seed_preregistered_selector_package(tmp_path)

    provenance = build_component_ablation_package(
        selector_dir=selector,
        ablations=["full", "without_risk_penalty"],
        expected_events=2,
        expected_candidates_per_event=2,
    )

    assert provenance["ok"] is True
    assert provenance["selection_source"]["selected_on"] == "preregistered"
    assert provenance["selection_source"]["main_config_mode"] == "preregistered"
    assert provenance["selected_config"]["eta"] == 1.0
    summary = (selector / "component_ablation_summary.csv").read_text(encoding="utf-8")
    assert ",1.0,0.7," in summary
    assert ",0.5,0.7," not in summary


def test_component_ablation_builder_outputs_package_audit_compatible_files(tmp_path):
    selector = _seed_selector_package(tmp_path)
    build_component_ablation_package(
        selector_dir=selector,
        ablations=["full", "without_risk_penalty"],
        expected_events=2,
        expected_candidates_per_event=2,
    )
    _seed_package_audit_extras(selector)

    audit = build_audit(
        module="component_ablation",
        package_dir=selector,
        expected_events=2,
        expected_candidates_per_event=2,
        expected_ablations=("full", "without_risk_penalty"),
    )

    assert audit["ok"] is True
    assert audit["paper_claim_ready"] is True
