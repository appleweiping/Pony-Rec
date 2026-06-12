import csv
from pathlib import Path

from scripts.analysis.main_build_paper_result_tables import build_paper_result_tables


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _method_rows() -> list[dict[str, str]]:
    domains = ("sports", "toys", "home", "tools")
    rows: list[dict[str, str]] = []
    for domain in domains:
        rows.append(
            {
                "domain": domain,
                "rank_by_NDCG@10": "1",
                "method": "ccrp_v3_qwen3base_pointwise",
                "kind": "internal_method",
                "HR@10": "0.3",
                "NDCG@10": "0.2",
                "MRR": "0.19",
            }
        )
        for rank, method in enumerate(
            (
                "llmemb",
                "irllrec_intent",
                "rlmrec_graphcl",
                "llm2rec_sasrec",
                "llmesr_sasrec",
                "proex_profile",
                "promax_profile",
                "elmrec_graph",
            ),
            start=2,
        ):
            rows.append(
                {
                    "domain": domain,
                    "rank_by_NDCG@10": str(rank),
                    "method": method,
                    "kind": "official_baseline",
                    "HR@10": "0.1",
                    "NDCG@10": "0.05",
                    "MRR": "0.04",
                }
            )
    return rows


def _domain_summary_rows() -> list[dict[str, str]]:
    return [
        {
            "domain": domain,
            "paired_positive_holm_significant_count": "56",
            "paired_test_count": "56",
            "min_delta": "0.01",
            "min_ci_low": "0.005",
            "max_holm_p_value": "0.0001",
        }
        for domain in ("sports", "toys", "home", "tools")
    ]


def test_build_paper_result_tables_outputs_full_tables_and_audit(tmp_path: Path) -> None:
    method_rows_csv = _write_csv(tmp_path / "method_rows.csv", _method_rows())
    domain_summary_csv = _write_csv(tmp_path / "domain_summary.csv", _domain_summary_rows())
    full_table = tmp_path / "full.tex"
    sig_table = tmp_path / "sig.tex"
    audit_json = tmp_path / "audit.json"
    audit_md = tmp_path / "audit.md"

    audit = build_paper_result_tables(
        method_rows_csv=method_rows_csv,
        domain_summary_csv=domain_summary_csv,
        full_table_tex=full_table,
        significance_table_tex=sig_table,
        audit_json=audit_json,
        audit_md=audit_md,
    )

    assert audit["ok"] is True
    assert audit["method_row_count"] == 36
    assert audit["domain_counts"]["sports"]["official_rows"] == 8
    assert audit["domain_counts"]["tools"]["ccrp_rank1"] is True
    assert "LLMEmb" in full_table.read_text(encoding="utf-8")
    assert r"\method{}" in full_table.read_text(encoding="utf-8")
    assert "56/56" in sig_table.read_text(encoding="utf-8")
    assert audit_json.exists()
    assert audit_md.exists()
