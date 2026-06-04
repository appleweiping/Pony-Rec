import json

import numpy as np
import pytest

from scripts.analysis.main_build_uncertainty_observation_study import (
    _write_json,
    build_observation_tables,
    load_event_uncertainty,
)


def test_build_observation_tables_bins_uncertainty_and_methods(tmp_path):
    uncertainty_path = tmp_path / "uncertainty_scores.csv"
    uncertainty_path.write_text(
        "source_event_id,user_id,item_id,ccrp_uncertainty\n"
        "e1,u1,i1,0.1\n"
        "e1,u1,i2,0.3\n"
        "e2,u2,i1,0.2\n"
        "e2,u2,i2,0.4\n"
        "e3,u3,i1,0.7\n"
        "e3,u3,i2,0.9\n"
        "e4,u4,i1,0.6\n"
        "e4,u4,i2,0.8\n",
        encoding="utf-8",
    )
    ccrp_eval = tmp_path / "ccrp_eval.csv"
    ccrp_eval.write_text(
        "source_event_id,positive_rank,num_candidates\n"
        "e1,1,21\n"
        "e2,3,21\n"
        "e3,11,21\n"
        "e4,20,21\n",
        encoding="utf-8",
    )
    baseline_eval = tmp_path / "baseline_eval.csv"
    baseline_eval.write_text(
        "source_event_id,positive_rank,num_candidates\n"
        "e1,2,21\n"
        "e2,2,21\n"
        "e3,8,21\n"
        "e4,21,21\n",
        encoding="utf-8",
    )

    event_bins, summary, provenance = build_observation_tables(
        domain="toy",
        uncertainty_scores_path=str(uncertainty_path),
        ccrp_eval_path=str(ccrp_eval),
        method_eval_specs=[f"baseline={baseline_eval}"],
        uncertainty_col="",
        event_agg="mean",
        n_bins=2,
        ks=[5, 10, 20],
        expected_events=4,
        expected_candidates_per_event=21,
        min_join_rate=1.0,
    )

    assert set(event_bins["method"]) == {"ccrp_v3", "baseline"}
    assert len(event_bins) == 8
    assert set(summary["uncertainty_bin"]) >= {"ALL"}
    assert {"MRR", "HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20"} <= set(summary.columns)
    assert len(summary) == 6
    assert summary.loc[
        (summary["method"] == "ccrp_v3") & (summary["uncertainty_bin"] == "ALL"),
        "HR@10",
    ].item() == pytest.approx(0.5)
    assert provenance["artifact_class"] == "paper_critical_observation_motivation"
    assert provenance["paper_claim_scope"] == "motivation_only_not_main_table_sota"
    assert provenance["expected_candidates_per_event"] == 21
    assert provenance["required_metrics"] == ["MRR", "HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20"]
    assert provenance["uncertainty_summary"]["uncertainty_col"] == "ccrp_uncertainty"
    assert all(row["join_rate"] == 1.0 for row in provenance["join_report"])
    assert all(row["exact_event_match"] for row in provenance["join_report"])


def test_uncertainty_observation_rejects_final_score_only_file(tmp_path):
    score_path = tmp_path / "scores.csv"
    score_path.write_text(
        "source_event_id,user_id,item_id,score\n"
        "e1,u1,i1,0.9\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No uncertainty column"):
        load_event_uncertainty(score_path)


def test_observation_rejects_duplicate_method_eval_events(tmp_path):
    uncertainty_path = tmp_path / "uncertainty_scores.csv"
    uncertainty_path.write_text(
        "source_event_id,ccrp_uncertainty\n"
        "e1,0.1\n"
        "e2,0.2\n",
        encoding="utf-8",
    )
    ccrp_eval = tmp_path / "ccrp_eval.csv"
    ccrp_eval.write_text(
        "source_event_id,positive_rank,num_candidates\n"
        "e1,1,101\n"
        "e1,2,101\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate event_id"):
        build_observation_tables(
            domain="toy",
            uncertainty_scores_path=str(uncertainty_path),
            ccrp_eval_path=str(ccrp_eval),
            method_eval_specs=[],
            uncertainty_col="",
            event_agg="mean",
            n_bins=2,
            ks=[5, 10, 20],
            expected_events=2,
            expected_candidates_per_event=101,
            min_join_rate=1.0,
        )


def test_observation_rejects_extra_method_eval_events(tmp_path):
    uncertainty_path = tmp_path / "uncertainty_scores.csv"
    uncertainty_path.write_text(
        "source_event_id,ccrp_uncertainty\n"
        "e1,0.1\n"
        "e2,0.2\n",
        encoding="utf-8",
    )
    ccrp_eval = tmp_path / "ccrp_eval.csv"
    ccrp_eval.write_text(
        "source_event_id,positive_rank,num_candidates\n"
        "e1,1,101\n"
        "e2,2,101\n"
        "e3,3,101\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not present in the C-CRP uncertainty input"):
        build_observation_tables(
            domain="toy",
            uncertainty_scores_path=str(uncertainty_path),
            ccrp_eval_path=str(ccrp_eval),
            method_eval_specs=[],
            uncertainty_col="",
            event_agg="mean",
            n_bins=2,
            ks=[5, 10, 20],
            expected_events=2,
            expected_candidates_per_event=101,
            min_join_rate=1.0,
        )


def test_observation_rejects_candidate_count_mismatch(tmp_path):
    uncertainty_path = tmp_path / "uncertainty_scores.csv"
    uncertainty_path.write_text(
        "source_event_id,ccrp_uncertainty\n"
        "e1,0.1\n",
        encoding="utf-8",
    )
    ccrp_eval = tmp_path / "ccrp_eval.csv"
    ccrp_eval.write_text(
        "source_event_id,positive_rank,num_candidates\n"
        "e1,1,100\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected 101"):
        build_observation_tables(
            domain="toy",
            uncertainty_scores_path=str(uncertainty_path),
            ccrp_eval_path=str(ccrp_eval),
            method_eval_specs=[],
            uncertainty_col="",
            event_agg="mean",
            n_bins=1,
            ks=[5, 10, 20],
            expected_events=1,
            expected_candidates_per_event=101,
            min_join_rate=1.0,
        )


def test_observation_json_writer_handles_numpy_scalars(tmp_path):
    path = tmp_path / "provenance.json"

    _write_json(path, {"count": np.int64(4), "score": np.float64(0.25)})

    assert json.loads(path.read_text(encoding="utf-8")) == {"count": 4, "score": 0.25}
