from __future__ import annotations

import json
import sys
from pathlib import Path

from src.cli.run_care_rerank_pilot import main as care_pilot_main
from src.data.protocol import write_jsonl
from src.methods.care_rerank import (
    VARIANT_ORDER,
    load_care_rerank_config,
    rerank_candidates_for_user,
    sanitize_listwise_ranking,
    stress_unreliability,
    tail_recovery_bonus,
)
from src.utils.manifest import build_manifest


def _pred(
    *,
    ranking: list[str],
    conf: float,
    correct: bool,
    buckets: list[str] | None = None,
) -> dict:
    cands = list(ranking)
    if buckets is None:
        buckets = ["tail"] * len(cands)
    return {
        "user_id": "u1",
        "candidate_item_ids": cands,
        "candidate_popularity_buckets": buckets,
        "predicted_ranking": list(ranking),
        "target_item_id": ranking[0],
        "correctness": correct,
        "raw_confidence": conf,
        "missing_confidence": False,
        "is_valid": True,
    }


def test_care_manifest_flags_pilot_non_paper() -> None:
    m = build_manifest(
        config={"run_type": "pilot"},
        dataset="d",
        domain="d",
        method="care_rerank_care_full",
        backend="rerank",
        model="m",
        prompt_template="p",
        seed=42,
    )
    assert m["backend_type"] == "rerank"
    assert m["is_paper_result"] is False


def test_base_score_monotonicity_identity_variant() -> None:
    """With equal auxiliary terms (identity), care_total equals base_score ∝ 1/log2(rank+1)."""
    cfg = load_care_rerank_config(Path("configs/methods/care_rerank_pilot.yaml"))
    ranking = ["b", "a", "c"]
    pred = _pred(ranking=ranking, conf=0.6, correct=True, buckets=["tail", "tail", "tail"])
    feat: dict = {"care_features": {}}
    _, rows = rerank_candidates_for_user(pred, feat, "original_deepseek", cfg)
    by_item = {r["item_id"]: r["care_total"] for r in rows}
    assert by_item["b"] > by_item["a"] > by_item["c"]


def test_head_echo_risk_exceeds_tail_under_stress() -> None:
    from src.methods.care_rerank import echo_risk_item

    s = 0.9
    assert echo_risk_item("head", s, blind_popularity=False) > echo_risk_item("tail", s, blind_popularity=False)


def test_tail_recovery_requires_safe_conf_and_rank() -> None:
    assert tail_recovery_bonus("tail", 3, 0.5) > 0
    assert tail_recovery_bonus("tail", 8, 0.5) == 0
    assert tail_recovery_bonus("tail", 3, 0.2) == 0


def test_rerank_preserves_candidate_set_no_dupes() -> None:
    cfg = load_care_rerank_config(Path("configs/methods/care_rerank_pilot.yaml"))
    ranking = [f"i{x}" for x in range(5)]
    pred = _pred(ranking=ranking, conf=0.55, correct=False, buckets=["mid", "tail", "tail", "head", "tail"])
    feat = {"care_features": {"high_confidence_wrong": True}}
    for v in VARIANT_ORDER:
        new_r, _ = rerank_candidates_for_user(pred, feat, v, cfg)
        assert set(new_r) == set(ranking)
        assert len(new_r) == len(set(new_r))


def test_sanitize_drops_out_of_slate_and_dupes() -> None:
    pred = {
        "user_id": "u",
        "candidate_item_ids": ["a", "b", "c"],
        "candidate_popularity_buckets": ["tail", "tail", "tail"],
        "predicted_ranking": ["b", "b", "z", "a"],
    }
    s = sanitize_listwise_ranking(pred)
    assert s["predicted_ranking"] == ["b", "a", "c"]


def test_stress_high_when_high_conf_wrong() -> None:
    pred = _pred(ranking=["a", "b"], conf=0.95, correct=False, buckets=["tail", "tail"])
    feat = {"care_features": {"high_confidence_wrong": True}}
    assert stress_unreliability(pred, feat) >= 0.85


def test_cli_smoke_care_rerank_pilot(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot"
    (pilot / "amazon_beauty" / "valid" / "predictions").mkdir(parents=True)
    preds = [
        {
            "user_id": "u1",
            "dataset": "amazon_beauty",
            "domain": "amazon_beauty",
            "split": "valid",
            "seed": 42,
            "candidate_item_ids": ["a", "b", "c"],
            "candidate_popularity_buckets": ["tail", "head", "tail"],
            "predicted_ranking": ["b", "a", "c"],
            "predicted_item_id": "b",
            "target_item_id": "a",
            "correctness": False,
            "raw_confidence": 0.9,
            "missing_confidence": False,
            "is_valid": True,
        }
    ]
    write_jsonl(preds, str(pilot / "amazon_beauty" / "valid" / "predictions" / "rank_predictions.jsonl"))
    write_jsonl(
        [{"user_id": "u1", "care_features": {"high_confidence_wrong": True, "verbalized_uncertainty_score": 0.2}}],
        str(pilot / "amazon_beauty" / "valid" / "predictions" / "uncertainty_features.jsonl"),
    )
    rep = tmp_path / "reprocess" / "amazon_beauty"
    rep.mkdir(parents=True)
    write_jsonl(
        [
            {
                "user_id": "u1",
                "target_item_id": "a",
                "candidate_item_ids": ["a", "b", "c"],
                "history_item_ids": [],
            }
        ],
        str(rep / "valid_candidates.jsonl"),
    )
    out_root = tmp_path / "out"
    old = sys.argv
    sys.argv = [
        "run_care_rerank_pilot",
        "--pilot_root",
        str(pilot),
        "--output_root",
        str(out_root),
        "--config",
        "configs/methods/care_rerank_pilot.yaml",
        "--reprocess_dir",
        str(tmp_path / "reprocess"),
    ]
    try:
        care_pilot_main()
    finally:
        sys.argv = old
    agg = out_root / "care_rerank_aggregate.csv"
    assert agg.is_file()
    man = json.loads((out_root / "original_deepseek" / "amazon_beauty" / "valid" / "care_manifest.json").read_text())
    assert man["run_type"] == "pilot"
    assert man["is_paper_result"] is False
