from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_METHOD_ROWS = Path("outputs/summary/new_domains_official_ccrp_cross_domain_20260605_method_rows.csv")
DEFAULT_DOMAIN_SUMMARY = Path("outputs/summary/new_domains_official_ccrp_cross_domain_20260605_domain_summary.csv")
DEFAULT_FULL_TABLE = Path("Paper/tables/full_official_ndcg10_ranking.tex")
DEFAULT_SIGNIFICANCE_TABLE = Path("Paper/tables/significance_summary.tex")
DEFAULT_AUDIT_JSON = Path("outputs/summary/paper_critical/paper_result_tables_audit_20260612.json")
DEFAULT_AUDIT_MD = Path("outputs/summary/paper_critical/paper_result_tables_audit_20260612.md")

DOMAIN_ORDER = ("sports", "toys", "home", "tools")
DISPLAY_DOMAIN = {
    "sports": "Sports",
    "toys": "Toys",
    "home": "Home",
    "tools": "Tools",
}
DISPLAY_METHOD = {
    "ccrp_v3_qwen3base_pointwise": r"\method{}",
    "llmemb": "LLMEmb",
    "irllrec_intent": "IRLLRec",
    "rlmrec_graphcl": "RLMRec",
    "llm2rec_sasrec": "LLM2Rec",
    "llmesr_sasrec": "LLM-ESR",
    "proex_profile": "ProEx",
    "promax_profile": "ProMax",
    "elmrec_graph": "ELMRec",
}
EXPECTED_METHOD_ROWS = 36
EXPECTED_OFFICIAL_PER_DOMAIN = 8
EXPECTED_PAIRED_PER_DOMAIN = 56


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _as_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def _fmt_metric(value: float) -> str:
    return f"{value:.4f}"


def _fmt_p(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _bold_if_ccrp(row: dict[str, str], value: str) -> str:
    return rf"\textbf{{{value}}}" if row["kind"] == "internal_method" else value


def build_full_ndcg10_table(rows: list[dict[str, str]]) -> str:
    ordered: list[dict[str, str]] = []
    for domain in DOMAIN_ORDER:
        domain_rows = [row for row in rows if row["domain"] == domain]
        ordered.extend(sorted(domain_rows, key=lambda row: _as_int(row, "rank_by_NDCG@10")))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Complete same-candidate NDCG@10 ranking for the four new domains. Each domain contains 10,000 users, 101 candidates per user, one C-CRP row, and all eight official-code-level baseline rows.}",
        r"\label{tab:full_official_ndcg10}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Domain & Method & Rank & NDCG@10 & HR@10 & MRR \\",
        r"\midrule",
    ]

    first_domain = True
    for domain in DOMAIN_ORDER:
        if not first_domain:
            lines.append(r"\midrule")
        first_domain = False
        for row in [item for item in ordered if item["domain"] == domain]:
            method = DISPLAY_METHOD.get(row["method"], row["method"].replace("_", r"\_"))
            rank = str(_as_int(row, "rank_by_NDCG@10"))
            ndcg10 = _bold_if_ccrp(row, _fmt_metric(_as_float(row, "NDCG@10")))
            hr10 = _bold_if_ccrp(row, _fmt_metric(_as_float(row, "HR@10")))
            mrr = _bold_if_ccrp(row, _fmt_metric(_as_float(row, "MRR")))
            lines.append(
                f"{DISPLAY_DOMAIN[domain]} & {method} & {rank} & {ndcg10} & {hr10} & {mrr} " + r"\\"
            )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_significance_table(rows: list[dict[str, str]]) -> str:
    domain_rows = {row["domain"]: row for row in rows}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Paired-test gate summary for C-CRP versus the eight official baselines. The Holm family contains seven metrics for each of eight baselines per domain.}",
        r"\label{tab:significance_summary}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Domain & Positive Holm & Min $\Delta$ & Min CI low & Max Holm $p$ \\",
        r"\midrule",
    ]
    for domain in DOMAIN_ORDER:
        row = domain_rows[domain]
        positive = _as_int(row, "paired_positive_holm_significant_count")
        total = _as_int(row, "paired_test_count")
        min_delta = _fmt_metric(_as_float(row, "min_delta"))
        min_ci_low = _fmt_metric(_as_float(row, "min_ci_low"))
        max_p = _fmt_p(_as_float(row, "max_holm_p_value"))
        lines.append(f"{DISPLAY_DOMAIN[domain]} & {positive}/{total} & {min_delta} & {min_ci_low} & {max_p} " + r"\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def build_audit(method_rows: list[dict[str, str]], domain_summary_rows: list[dict[str, str]]) -> dict[str, Any]:
    domains = sorted({row["domain"] for row in method_rows})
    domain_counts = {
        domain: {
            "method_rows": sum(1 for row in method_rows if row["domain"] == domain),
            "official_rows": sum(
                1 for row in method_rows if row["domain"] == domain and row["kind"] == "official_baseline"
            ),
            "ccrp_rows": sum(1 for row in method_rows if row["domain"] == domain and row["kind"] == "internal_method"),
            "ccrp_rank1": any(
                row["domain"] == domain
                and row["kind"] == "internal_method"
                and _as_int(row, "rank_by_NDCG@10") == 1
                for row in method_rows
            ),
        }
        for domain in DOMAIN_ORDER
    }
    significance_counts = {
        row["domain"]: {
            "paired_positive_holm_significant_count": _as_int(row, "paired_positive_holm_significant_count"),
            "paired_test_count": _as_int(row, "paired_test_count"),
            "min_delta": _as_float(row, "min_delta"),
            "min_ci_low": _as_float(row, "min_ci_low"),
            "max_holm_p_value": _as_float(row, "max_holm_p_value"),
        }
        for row in domain_summary_rows
    }
    failures: list[str] = []
    if len(method_rows) != EXPECTED_METHOD_ROWS:
        failures.append(f"expected {EXPECTED_METHOD_ROWS} method rows, found {len(method_rows)}")
    if set(domains) != set(DOMAIN_ORDER):
        failures.append(f"domain mismatch: {domains}")
    for domain, counts in domain_counts.items():
        if counts["method_rows"] != EXPECTED_OFFICIAL_PER_DOMAIN + 1:
            failures.append(f"{domain}: expected 9 method rows, found {counts['method_rows']}")
        if counts["official_rows"] != EXPECTED_OFFICIAL_PER_DOMAIN:
            failures.append(f"{domain}: expected 8 official rows, found {counts['official_rows']}")
        if counts["ccrp_rows"] != 1:
            failures.append(f"{domain}: expected 1 C-CRP row, found {counts['ccrp_rows']}")
        if not counts["ccrp_rank1"]:
            failures.append(f"{domain}: C-CRP is not rank 1 by NDCG@10")
        sig = significance_counts.get(domain)
        if not sig:
            failures.append(f"{domain}: missing significance summary")
        elif sig["paired_positive_holm_significant_count"] != EXPECTED_PAIRED_PER_DOMAIN:
            failures.append(f"{domain}: expected 56 positive Holm tests, found {sig['paired_positive_holm_significant_count']}")

    return {
        "schema_version": "2026-06-12.paper_result_tables.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_inputs": True,
        "will_ssh": False,
        "will_start_experiment": False,
        "ok": not failures,
        "method_row_count": len(method_rows),
        "domain_counts": domain_counts,
        "significance_counts": significance_counts,
        "generated_tables": {
            "full_official_ndcg10": str(DEFAULT_FULL_TABLE),
            "significance_summary": str(DEFAULT_SIGNIFICANCE_TABLE),
        },
        "failures": failures,
    }


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Paper Result Tables Audit",
        "",
        f"- ok: `{audit['ok']}`",
        f"- method rows: `{audit['method_row_count']}`",
        f"- generated full ranking table: `{audit['generated_tables']['full_official_ndcg10']}`",
        f"- generated significance table: `{audit['generated_tables']['significance_summary']}`",
        "",
        "| Domain | Rows | Official | C-CRP rank 1 | Positive Holm | Min delta | Max Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for domain in DOMAIN_ORDER:
        counts = audit["domain_counts"][domain]
        sig = audit["significance_counts"][domain]
        lines.append(
            f"| {DISPLAY_DOMAIN[domain]} | {counts['method_rows']} | {counts['official_rows']} | "
            f"{counts['ccrp_rank1']} | {sig['paired_positive_holm_significant_count']}/{sig['paired_test_count']} | "
            f"{sig['min_delta']:.4f} | {_fmt_p(sig['max_holm_p_value'])} |"
        )
    if audit["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in audit["failures"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_paper_result_tables(
    *,
    method_rows_csv: Path = DEFAULT_METHOD_ROWS,
    domain_summary_csv: Path = DEFAULT_DOMAIN_SUMMARY,
    full_table_tex: Path = DEFAULT_FULL_TABLE,
    significance_table_tex: Path = DEFAULT_SIGNIFICANCE_TABLE,
    audit_json: Path = DEFAULT_AUDIT_JSON,
    audit_md: Path = DEFAULT_AUDIT_MD,
) -> dict[str, Any]:
    method_rows = _read_csv(method_rows_csv)
    domain_summary_rows = _read_csv(domain_summary_csv)
    audit = build_audit(method_rows, domain_summary_rows)
    full_table_tex.parent.mkdir(parents=True, exist_ok=True)
    full_table_tex.write_text(build_full_ndcg10_table(method_rows), encoding="utf-8")
    significance_table_tex.parent.mkdir(parents=True, exist_ok=True)
    significance_table_tex.write_text(build_significance_table(domain_summary_rows), encoding="utf-8")
    audit["generated_tables"] = {
        "full_official_ndcg10": str(full_table_tex),
        "significance_summary": str(significance_table_tex),
    }
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    write_markdown(audit_md, audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-rows-csv", type=Path, default=DEFAULT_METHOD_ROWS)
    parser.add_argument("--domain-summary-csv", type=Path, default=DEFAULT_DOMAIN_SUMMARY)
    parser.add_argument("--full-table-tex", type=Path, default=DEFAULT_FULL_TABLE)
    parser.add_argument("--significance-table-tex", type=Path, default=DEFAULT_SIGNIFICANCE_TABLE)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--audit-md", type=Path, default=DEFAULT_AUDIT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_paper_result_tables(
        method_rows_csv=args.method_rows_csv,
        domain_summary_csv=args.domain_summary_csv,
        full_table_tex=args.full_table_tex,
        significance_table_tex=args.significance_table_tex,
        audit_json=args.audit_json,
        audit_md=args.audit_md,
    )
    print(json.dumps({"ok": audit["ok"], "failures": audit["failures"]}, indent=2))
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
