from __future__ import annotations

from main_build_paper_ready_baseline_comparison import build_paper_ready_rows


def test_paper_ready_rows_select_expected_methods_and_best_srpd():
    matrix_rows = [
        {
            "domain": "books",
            "comparison_scope": "week7_7_four_domain_export",
            "evidence_family": "week7_7_reference",
            "method": "direct_candidate_ranking",
            "method_variant": "direct_candidate_ranking",
            "NDCG@10": "0.63",
            "MRR": "0.52",
        },
        {
            "domain": "books",
            "comparison_scope": "week7_7_four_domain_export",
            "evidence_family": "week7_7_reference",
            "method": "structured_risk_rerank",
            "method_variant": "structured_risk_rerank",
            "NDCG@10": "0.64",
            "MRR": "0.53",
        },
        {
            "domain": "books",
            "comparison_scope": "week7_7_four_domain_export",
            "evidence_family": "srpd_trainable_framework",
            "method": "srpd_lora_ranker",
            "method_variant": "srpd_v1",
            "NDCG@10": "0.65",
            "MRR": "0.54",
        },
        {
            "domain": "books",
            "comparison_scope": "week7_7_four_domain_export",
            "evidence_family": "srpd_trainable_framework",
            "method": "srpd_lora_ranker",
            "method_variant": "srpd_v2",
            "NDCG@10": "0.70",
            "MRR": "0.60",
        },
        {
            "domain": "books",
            "comparison_scope": "week8_same_candidate_external_baseline",
            "evidence_family": "external_same_candidate_baseline",
            "method": "sasrec",
            "method_variant": "sasrec",
            "NDCG@10": "0.43",
            "artifact_class": "completed_result",
        },
        {
            "domain": "books",
            "comparison_scope": "week8_same_candidate_external_baseline",
            "evidence_family": "external_same_candidate_baseline",
            "method": "lightgcn",
            "method_variant": "lightgcn",
            "NDCG@10": "0.51",
            "artifact_class": "completed_result",
        },
        {
            "domain": "books",
            "comparison_scope": "week8_same_candidate_external_baseline",
            "evidence_family": "external_same_candidate_baseline",
            "method": "llm2rec_style_qwen3_sasrec",
            "method_variant": "llm2rec_style_qwen3_sasrec",
            "NDCG@10": "0.55",
            "artifact_class": "completed_result",
            "paper_role_hint": "same_schema_external_baseline",
        },
        {
            "domain": "books",
            "comparison_scope": "week8_same_candidate_external_baseline",
            "evidence_family": "external_same_candidate_baseline",
            "method": "llmesr_style_qwen3_sasrec",
            "method_variant": "llmesr_style_qwen3_sasrec",
            "NDCG@10": "0.56",
            "artifact_class": "completed_result",
            "paper_role_hint": "same_schema_external_baseline",
        },
        {
            "domain": "books",
            "comparison_scope": "week8_same_candidate_adapter_diagnostic",
            "evidence_family": "paper_adapter_scaffold",
            "method": "llmesr_scaffold",
            "method_variant": "llmesr_scaffold_hf_mean_pool_Qwen3-8B",
            "NDCG@10": "0.69",
            "artifact_class": "adapter_scaffold_score",
            "paper_role_hint": "adapter_scaffold_diagnostic_not_completed_result",
        },
    ]

    rows = build_paper_ready_rows(matrix_rows, domains=["books"])

    assert [row["table_group"] for row in rows] == [
        "week77_direct_reference",
        "week77_structured_risk_reference",
        "srpd_best_self_trained",
        "completed_external_baseline",
        "completed_external_baseline",
        "paper_project_same_backbone_baseline",
        "paper_project_same_backbone_baseline",
        "paper_adapter_scaffold_diagnostic",
    ]
    assert rows[2]["display_method"] == "SRPD best (srpd_v2)"
    assert rows[-3]["display_method"] == "LLM2Rec-style Qwen3-8B Emb. + SASRec"
    assert rows[-2]["display_method"] == "LLM-ESR-style Qwen3-8B Emb. + LLMESR-SASRec"
    assert rows[-1]["display_method"] == "llmesr_scaffold_hf_mean_pool_Qwen3-8B"
    assert rows[-1]["artifact_class"] == "adapter_scaffold_score"
