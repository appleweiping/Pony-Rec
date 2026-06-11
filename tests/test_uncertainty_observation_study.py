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
    uncertainty_rows = ["source_event_id,user_id,item_id,ccrp_uncertainty"]
    for event_id, user_id, low, high in (
        ("e1", "u1", 0.1, 0.3),
        ("e2", "u2", 0.2, 0.4),
        ("e3", "u3", 0.7, 0.9),
        ("e4", "u4", 0.6, 0.8),
    ):
        for idx, value in enumerate(np.linspace(low, high, 21), start=1):
            uncertainty_rows.append(f"{event_id},{user_id},i{idx},{value}")
    uncertainty_path.write_text("\n".join(uncertainty_rows) + "\n", encoding="utf-8")
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
    assert provenance["expected_uncertainty_rows_per_event"] == 21
    assert provenance["required_metrics"] == ["MRR", "HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20"]
    assert provenance["uncertainty_summary"]["uncertainty_col"] == "ccrp_uncertainty"
    assert provenance["uncertainty_summary"]["finite_uncertainty_rows"] == 84
    assert provenance["uncertainty_summary"]["candidate_rows_min"] == 21
    assert provenance["uncertainty_summary"]["candidate_rows_max"] == 21
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
            expected_uncertainty_rows_per_event=0,
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
            expected_uncertainty_rows_per_event=0,
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
            expected_uncertainty_rows_per_event=0,
            min_join_rate=1.0,
        )


def test_observation_rejects_incomplete_uncertainty_candidate_rows(tmp_path):
    uncertainty_path = tmp_path / "uncertainty_scores.csv"
    uncertainty_path.write_text(
        "source_event_id,ccrp_uncertainty\n"
        "e1,0.1\n"
        "e1,0.2\n"
        "e2,0.3\n",
        encoding="utf-8",
    )
    ccrp_eval = tmp_path / "ccrp_eval.csv"
    ccrp_eval.write_text(
        "source_event_id,positive_rank,num_candidates\n"
        "e1,1,2\n"
        "e2,2,2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Expected exactly 2 finite uncertainty rows per event"):
        build_observation_tables(
            domain="toy",
            uncertainty_scores_path=str(uncertainty_path),
            ccrp_eval_path=str(ccrp_eval),
            method_eval_specs=[],
            uncertainty_col="",
            event_agg="mean",
            n_bins=1,
            ks=[5, 10, 20],
            expected_events=2,
            expected_candidates_per_event=2,
            min_join_rate=1.0,
        )


def test_observation_json_writer_handles_numpy_scalars(tmp_path):
    path = tmp_path / "provenance.json"

    _write_json(path, {"count": np.int64(4), "score": np.float64(0.25)})

    assert json.loads(path.read_text(encoding="utf-8")) == {"count": 4, "score": 0.25}
