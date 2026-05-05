from __future__ import annotations

import csv
import json

from main_build_unified_method_matrix import _external_baseline_rows


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_external_rows_include_adapter_scaffold_as_diagnostic(tmp_path):
    output_root = tmp_path / "outputs"
    summary_path = (
        output_root
        / "movies_llmesr_scaffold_same_candidate"
        / "tables"
        / "same_candidate_external_baseline_summary.csv"
    )
    _write_csv(
        summary_path,
        [
            {
                "baseline_name": "llmesr_scaffold",
                "domain": "movies",
                "status_label": "llmesr_adapter_scaffold_score",
                "artifact_class": "adapter_scaffold_score",
                "sample_count": "500",
                "NDCG@10": "0.68",
                "MRR": "0.58",
            }
        ],
        ["baseline_name", "domain", "status_label", "artifact_class", "sample_count", "NDCG@10", "MRR"],
    )
    metadata_path = (
        output_root
        / "baselines"
        / "paper_adapters"
        / "movies_llmesr_same_candidate_adapter"
        / "llmesr_embedding_metadata.json"
    )
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps({"backend": "hf_mean_pool", "model_name": "/home/ajifang/models/Qwen/Qwen3-8B"}),
        encoding="utf-8",
    )

    rows = _external_baseline_rows(str(output_root / "*" / "tables" / "same_candidate_external_baseline_summary.csv"))

    assert len(rows) == 1
    assert rows[0]["comparison_scope"] == "week8_same_candidate_adapter_diagnostic"
    assert rows[0]["evidence_family"] == "paper_adapter_scaffold"
    assert rows[0]["method_variant"] == "llmesr_scaffold_hf_mean_pool_Qwen3-8B"
    assert rows[0]["paper_role_hint"] == "adapter_scaffold_diagnostic_not_completed_result"
