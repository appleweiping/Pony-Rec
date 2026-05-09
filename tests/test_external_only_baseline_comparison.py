from __future__ import annotations

import csv

from main_build_external_only_baseline_comparison import build_external_only_rows


def _write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "baseline_name",
        "domain",
        "status_label",
        "artifact_class",
        "sample_count",
        "HR@5",
        "NDCG@5",
        "HR@10",
        "NDCG@10",
        "HR@20",
        "NDCG@20",
        "MRR",
        "coverage@5",
        "coverage@10",
        "coverage@20",
        "head_exposure_ratio@10",
        "longtail_coverage@10",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_external_only_comparison_orders_completed_methods(tmp_path):
    _write_summary(
        tmp_path / "outputs" / "books_large" / "tables" / "same_candidate_external_baseline_summary.csv",
        [
            {
                "baseline_name": "rlmrec_style_qwen3_graphcl",
                "domain": "books",
                "status_label": "same_schema_external_baseline",
                "artifact_class": "completed_result",
                "sample_count": "10",
                "HR@10": "1.0",
                "NDCG@10": "0.7",
                "MRR": "0.6",
                "coverage@10": "1.0",
            },
            {
                "baseline_name": "sasrec",
                "domain": "books",
                "status_label": "same_schema_external_baseline",
                "artifact_class": "completed_result",
                "sample_count": "10",
                "HR@10": "1.0",
                "NDCG@10": "0.5",
                "MRR": "0.4",
                "coverage@10": "1.0",
            },
            {
                "baseline_name": "bert4rec",
                "domain": "books",
                "status_label": "same_schema_external_baseline",
                "artifact_class": "adapter_scaffold_score",
                "sample_count": "10",
                "NDCG@10": "0.9",
            },
        ],
    )

    rows = build_external_only_rows(
        external_summary_glob=str(tmp_path / "outputs" / "*" / "tables" / "same_candidate_external_baseline_summary.csv"),
        domains=["books"],
        methods=["sasrec", "rlmrec_style_qwen3_graphcl", "bert4rec"],
    )

    assert [row["method"] for row in rows] == ["sasrec", "rlmrec_style_qwen3_graphcl"]
    assert rows[1]["display_method"] == "RLMRec-style Qwen3-8B GraphCL"


def test_external_only_comparison_includes_official_methods_by_default(tmp_path):
    _write_summary(
        tmp_path / "outputs" / "books_llm2rec" / "tables" / "same_candidate_external_baseline_summary.csv",
        [
            {
                "baseline_name": "llm2rec_official_qwen3base_sasrec",
                "domain": "books",
                "status_label": "same_schema_external_baseline",
                "artifact_class": "completed_result",
                "sample_count": "10",
                "HR@5": "0.4",
                "NDCG@5": "0.3",
                "HR@10": "0.5",
                "NDCG@10": "0.35",
                "HR@20": "0.6",
                "NDCG@20": "0.4",
                "MRR": "0.2",
                "coverage@5": "0.7",
                "coverage@10": "0.8",
                "coverage@20": "0.9",
            }
        ],
    )

    rows = build_external_only_rows(
        external_summary_glob=str(tmp_path / "outputs" / "*" / "tables" / "same_candidate_external_baseline_summary.csv"),
        domains=["books"],
        methods=[
            "sasrec",
            "gru4rec",
            "bert4rec",
            "lightgcn",
            "llm2rec_official_qwen3base_sasrec",
        ],
    )

    assert rows[0]["method"] == "llm2rec_official_qwen3base_sasrec"
    assert rows[0]["display_method"] == "LLM2Rec official Qwen3-8B + SASRec"
    assert rows[0]["NDCG@20"] == "0.4"
