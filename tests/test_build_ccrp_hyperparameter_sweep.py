import json

import pandas as pd
import pytest

from scripts.analysis.main_build_ccrp_hyperparameter_sweep import _validate_rows, build_sweep


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _seed_split(root, split):
    ranking = _write(
        root / f"{split}_ranking.jsonl",
        '{"source_event_id":"e1","user_id":"u1","candidate_item_ids":["i1","i2","i3"],"positive_item_id":"i1"}\n'
        '{"source_event_id":"e2","user_id":"u2","candidate_item_ids":["i4","i5","i6"],"positive_item_id":"i5"}\n'
        '{"source_event_id":"e3","user_id":"u3","candidate_item_ids":["i7","i8","i9"],"positive_item_id":"i9"}\n',
    )
    candidates = _write(
        root / f"{split}_candidate_items.csv",
        "source_event_id,user_id,item_id\n"
        "e1,u1,i1\n"
        "e1,u1,i2\n"
        "e1,u1,i3\n"
        "e2,u2,i4\n"
        "e2,u2,i5\n"
        "e2,u2,i6\n"
        "e3,u3,i7\n"
        "e3,u3,i8\n"
        "e3,u3,i9\n",
    )
    signal = _write(
        root / f"{split}_signal.csv",
        "source_event_id,user_id,item_id,relevance_probability,calibrated_confidence,evidence_support,counterevidence_strength\n"
        "e1,u1,i1,0.91,0.84,0.90,0.02\n"
        "e1,u1,i2,0.42,0.35,0.25,0.30\n"
        "e1,u1,i3,0.18,0.20,0.12,0.40\n"
        "e2,u2,i4,0.40,0.38,0.20,0.25\n"
        "e2,u2,i5,0.88,0.79,0.85,0.03\n"
        "e2,u2,i6,0.24,0.21,0.18,0.35\n"
        "e3,u3,i7,0.21,0.19,0.15,0.40\n"
        "e3,u3,i8,0.47,0.42,0.28,0.24\n"
        "e3,u3,i9,0.86,0.81,0.82,0.04\n",
    )
    return ranking, candidates, signal


def test_build_sweep_writes_valid_test_inputs_and_provenance(tmp_path):
    valid_ranking, valid_candidates, valid_signal = _seed_split(tmp_path, "valid")
    test_ranking, test_candidates, test_signal = _seed_split(tmp_path, "test")
    out = tmp_path / "hyper"

    provenance = build_sweep(
        domain="sports",
        valid_ranking_path=valid_ranking,
        test_ranking_path=test_ranking,
        valid_candidate_items_path=valid_candidates,
        test_candidate_items_path=test_candidates,
        valid_signal_path=valid_signal,
        test_signal_path=test_signal,
        output_dir=out,
        eta_grid="0,1,2",
        weight_grid="0.5,0.3,0.2;0.7,0.2,0.1;0.4,0.4,0.2",
        diagnostic_confidence_weights="0.1,0.5,0.9",
        expected_events=3,
        expected_candidates_per_event=3,
        max_tie_pair_rate=1.0,
        max_constant_event_rate=1.0,
    )

    valid = pd.read_csv(out / "valid_ccrp_hyperparameter_sweep.csv")
    test = pd.read_csv(out / "test_ccrp_hyperparameter_sweep.csv")
    saved_provenance = json.loads((out / "ccrp_hyperparameter_sweep_provenance.json").read_text(encoding="utf-8"))

    assert provenance["status_label"] == "valid_test_saved_signal_hyperparameter_sweep_ready"
    assert saved_provenance["test_not_used_for_selection"] is True
    assert saved_provenance["main_controls"] == ["eta", "weight_grid_label"]
    assert saved_provenance["diagnostic_controls"] == ["confidence_weight"]
    assert saved_provenance["cleanup_status"]["retained_bulk_scores_csv"] is False
    assert saved_provenance["row_counts"]["valid_signal_rows"] == 9
    assert saved_provenance["row_counts"]["test_ranking_events"] == 3
    assert len(valid) == 9
    assert len(test) == 9
    assert set(valid["split"]) == {"valid"}
    assert set(test["split"]) == {"test"}
    assert set(valid["control"]) == {"eta", "weight_grid_label", "confidence_weight"}
    assert set(valid.loc[valid["row_kind"] == "main_control", "control"]) == {"eta", "weight_grid_label"}
    assert set(valid.loc[valid["row_kind"] == "diagnostic_control", "control"]) == {"confidence_weight"}
    assert not ((valid["control"] == "confidence_weight") & (valid["score_mode"] == "full")).any()
    assert valid["score_coverage_rate"].eq(1.0).all()
    assert valid["candidate_key_count"].eq(9).all()
    assert valid["audit_ok"].eq(True).all()
    assert valid["degeneracy_audit_ok"].eq(True).all()
    assert (out / "valid_ccrp_hyperparameter_sweep.csv").is_file()
    assert (out / "test_ccrp_hyperparameter_sweep.csv").is_file()


def test_validate_rows_rejects_failed_degeneracy_audit():
    row = {
        "split": "valid",
        "control": "eta",
        "score_mode": "full",
        "candidate_key_count": 9,
        "score_key_count": 9,
        "score_coverage_rate": 1.0,
        "missing_score_keys": 0,
        "extra_score_keys": 0,
        "duplicate_score_keys": 0,
        "invalid_scores": 0,
        "blank_score_keys": 0,
        "audit_ok": True,
        "degeneracy_audit_ok": False,
        "MRR": 1.0,
        "HR@5": 1.0,
        "HR@10": 1.0,
        "HR@20": 1.0,
        "NDCG@5": 1.0,
        "NDCG@10": 1.0,
        "NDCG@20": 1.0,
    }

    with pytest.raises(ValueError, match="degeneracy_audit_ok"):
        _validate_rows([row], expected_keys=9, expected_main_controls=("eta",))
