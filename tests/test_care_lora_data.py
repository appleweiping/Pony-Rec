"""Tests for CARE-LoRA training data construction (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cli.build_care_lora_data import main as build_main
from src.methods.care_lora_data import (
    POLICIES,
    build_training_sample,
    evaluate_policies,
    summarize_policy_run,
    teacher_listwise_response,
)
from src.methods.care_rerank import sanitize_listwise_ranking


def _row() -> dict:
    return {
        "user_id": "U1",
        "target_item_id": "T1",
        "candidate_item_ids": ["A", "T1", "B"],
        "candidate_popularity_buckets": ["head", "tail", "mid"],
        "target_popularity_bucket": "tail",
        "_split_source": "valid",
    }


def _pred_wrong_hc() -> dict:
    return sanitize_listwise_ranking(
        {
            "user_id": "U1",
            "candidate_item_ids": ["A", "T1", "B"],
            "candidate_popularity_buckets": ["head", "tail", "mid"],
            "predicted_ranking": ["A", "B", "T1"],
            "raw_confidence": 0.9,
            "missing_confidence": False,
            "is_valid": True,
            "correctness": False,
        }
    )


def _feat_hc() -> dict:
    return {
        "care_features": {
            "high_confidence_wrong": True,
            "missing_confidence": False,
            "head_top1_wrong": True,
        }
    }


def test_teacher_response_is_valid_json() -> None:
    s = teacher_listwise_response("T1", ["A", "T1", "B"])
    d = json.loads(s)
    assert d["ranked_item_ids"][0] == "T1"
    assert len(d["ranked_item_ids"]) == 3


def test_vanilla_policy_always_keep_weight_one() -> None:
    ev = evaluate_policies(_row(), _pred_wrong_hc(), _feat_hc(), care_yaml=None)
    v = ev["vanilla_lora_baseline"]
    assert v.keep and v.sample_weight == 1.0


def test_prune_drops_invalid() -> None:
    bad_pred = dict(_pred_wrong_hc())
    bad_pred["is_valid"] = False
    ev = evaluate_policies(_row(), bad_pred, _feat_hc(), care_yaml=None)
    assert not ev["prune_high_uncertainty"].keep


def test_sample_weight_bounds() -> None:
    ev = evaluate_policies(_row(), _pred_wrong_hc(), _feat_hc(), care_yaml=None)
    for pol in POLICIES:
        o = ev[pol]
        if o.keep:
            assert 0.05 <= o.sample_weight <= 2.0
        else:
            assert o.sample_weight == 0.0


def test_no_target_leakage_in_teacher_order() -> None:
    row = _row()
    tid = str(row["target_item_id"])
    order = json.loads(teacher_listwise_response(tid, [str(x) for x in row["candidate_item_ids"]]))["ranked_item_ids"]
    negs = set(order) - {tid}
    hist = set(row.get("history_item_ids", []))
    assert not (negs & hist)
    assert tid in order


def test_vanilla_vs_care_full_can_differ() -> None:
    ev = evaluate_policies(_row(), _pred_wrong_hc(), _feat_hc(), care_yaml=None)
    assert ev["CARE_full_training"].keep
    assert ev["vanilla_lora_baseline"].sample_weight == 1.0
    assert ev["CARE_full_training"].sample_weight != 1.0


def test_build_training_sample_manifest_flags() -> None:
    ev = evaluate_policies(_row(), _pred_wrong_hc(), _feat_hc(), care_yaml=None)
    s = build_training_sample(
        row=_row(),
        prompt="P",
        pred=_pred_wrong_hc(),
        feat=_feat_hc(),
        policy="CARE_full_training",
        outcome=ev["CARE_full_training"],
        source_paths={"x": "y"},
    )
    assert s["run_type"] == "pilot"
    assert s["is_paper_result"] is False
    assert "care_risk_features" in s


def test_bucket_summary_keys() -> None:
    rows = [_row()]
    ev0 = evaluate_policies(rows[0], _pred_wrong_hc(), _feat_hc(), care_yaml=None)
    parallel = {pol: [ev0[pol]] for pol in POLICIES}
    s = summarize_policy_run(rows, parallel)
    assert "bucket_before" in s and "bucket_after" in s


def test_build_care_lora_data_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = tmp_path / "amazon_beauty"
    pred = d / "valid" / "predictions"
    pred.mkdir(parents=True)
    rep = tmp_path / "rep" / "amazon_beauty"
    rep.mkdir(parents=True)
    row = _row()
    pr = _pred_wrong_hc()
    pr["user_id"] = "U1"
    feat = _feat_hc()
    feat["user_id"] = "U1"
    (pred / "rank_predictions.jsonl").write_text(json.dumps(pr) + "\n", encoding="utf-8")
    (pred / "uncertainty_features.jsonl").write_text(json.dumps(feat) + "\n", encoding="utf-8")
    (rep / "valid_candidates.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "items.csv").write_text("item_id,title\nA,a\nT1,t\nB,b\n", encoding="utf-8")

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    build_main(
        [
            "--domain",
            "amazon_beauty",
            "--split",
            "valid",
            "--reprocess_dir",
            str(tmp_path / "rep"),
            "--deepseek_root",
            str(tmp_path),
            "--processed_dir",
            str(proc),
            "--output_root",
            str(tmp_path / "out"),
            "--care_rerank_config",
            str(root / "configs/methods/care_rerank_pilot.yaml"),
        ]
    )
    data_dir = tmp_path / "out" / "data"
    assert (data_dir / "vanilla_lora_baseline_train.jsonl").is_file()
    man = json.loads((data_dir / "data_manifest.json").read_text(encoding="utf-8"))
    assert man["run_type"] == "pilot"
    assert man["is_paper_result"] is False
