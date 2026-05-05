from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


CLASSICAL_METHOD_ORDER = ["sasrec", "gru4rec", "bert4rec", "lightgcn"]
PAPER_PROJECT_METHOD_ORDER = ["llm2rec_style_qwen3_sasrec", "llmesr_style_qwen3_sasrec"]

OUTPUT_FIELDS = [
    "domain",
    "table_group",
    "display_method",
    "method",
    "method_variant",
    "sample_count",
    "NDCG@10",
    "MRR",
    "delta_vs_domain_direct",
    "delta_vs_week77_structured_risk",
    "status_label",
    "artifact_class",
    "paper_role_hint",
    "source_scope",
    "source_file",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact paper/report-ready comparison table from the unified "
            "method matrix. The table keeps adapter scaffold rows separate from "
            "completed external baselines."
        )
    )
    parser.add_argument(
        "--unified_matrix_path",
        default="outputs/summary/unified_method_matrix_week77_shadow_external_qwen_llmesr.csv",
    )
    parser.add_argument("--domains", default="beauty,books,electronics,movies")
    parser.add_argument("--output_root", default="outputs/summary")
    parser.add_argument("--output_name", default="paper_ready_baseline_comparison_week77_qwen_llmesr")
    parser.add_argument("--include_shadow_v6", action="store_true")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDS})


def _safe_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _fmt_metric(value: Any) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return str(value or "")
    return f"{numeric:.6f}"


def _method_label(row: dict[str, str]) -> str:
    method = row.get("method", "")
    variant = row.get("method_variant", "")
    if method == "direct_candidate_ranking":
        return "Direct ranking"
    if method == "structured_risk_rerank":
        return "Structured-risk rerank"
    if method == "srpd_lora_ranker":
        return f"SRPD best ({variant})"
    if method == "llmesr_scaffold":
        return variant or "LLM-ESR Qwen3 scaffold"
    if method == "llm2rec_style_qwen3_sasrec":
        return "LLM2Rec-style Qwen3-8B Emb. + SASRec"
    if method == "llmesr_style_qwen3_sasrec":
        return "LLM-ESR-style Qwen3-8B Emb. + LLMESR-SASRec"
    if method == "shadow_v6_decision_bridge":
        return "Shadow-v6 diagnostic"
    return method.upper() if method in CLASSICAL_METHOD_ORDER else method


def _table_group(row: dict[str, str]) -> str:
    method = row.get("method", "")
    evidence_family = row.get("evidence_family", "")
    if method == "direct_candidate_ranking":
        return "week77_direct_reference"
    if method == "structured_risk_rerank":
        return "week77_structured_risk_reference"
    if method == "srpd_lora_ranker":
        return "srpd_best_self_trained"
    if evidence_family == "external_same_candidate_baseline":
        if method in PAPER_PROJECT_METHOD_ORDER:
            return "paper_project_same_backbone_baseline"
        return "completed_external_baseline"
    if evidence_family == "paper_adapter_scaffold":
        return "paper_adapter_scaffold_diagnostic"
    if method == "shadow_v6_decision_bridge":
        return "shadow_v6_diagnostic"
    return evidence_family


def _comparison_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "domain": row.get("domain", ""),
        "table_group": _table_group(row),
        "display_method": _method_label(row),
        "method": row.get("method", ""),
        "method_variant": row.get("method_variant", ""),
        "sample_count": row.get("sample_count", ""),
        "NDCG@10": row.get("NDCG@10", ""),
        "MRR": row.get("MRR", ""),
        "delta_vs_domain_direct": row.get("delta_vs_domain_direct", ""),
        "delta_vs_week77_structured_risk": row.get("delta_vs_week77_structured_risk", ""),
        "status_label": row.get("status_label", ""),
        "artifact_class": row.get("artifact_class", ""),
        "paper_role_hint": row.get("paper_role_hint", ""),
        "source_scope": row.get("comparison_scope", ""),
        "source_file": row.get("source_file", ""),
        "notes": row.get("notes", ""),
    }


def _best_by_ndcg(rows: list[dict[str, str]]) -> dict[str, str] | None:
    scored = [(row, _safe_float(row.get("NDCG@10"))) for row in rows]
    scored = [(row, score) for row, score in scored if score is not None]
    if not scored:
        return None
    return max(scored, key=lambda item: item[1])[0]


def _first_matching(rows: list[dict[str, str]], *, method: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("method") == method:
            return row
    return None


def _classical_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_method = {
        row.get("method", ""): row
        for row in rows
        if row.get("evidence_family") == "external_same_candidate_baseline"
    }
    return [by_method[method] for method in CLASSICAL_METHOD_ORDER if method in by_method]


def _paper_project_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_method = {
        row.get("method", ""): row
        for row in rows
        if row.get("evidence_family") == "external_same_candidate_baseline"
    }
    return [by_method[method] for method in PAPER_PROJECT_METHOD_ORDER if method in by_method]


def build_paper_ready_rows(
    matrix_rows: list[dict[str, str]],
    *,
    domains: list[str],
    include_shadow_v6: bool = False,
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    by_domain = {domain: [row for row in matrix_rows if row.get("domain") == domain] for domain in domains}

    for domain in domains:
        rows = by_domain.get(domain, [])
        selected: list[dict[str, str]] = []

        direct = _first_matching(rows, method="direct_candidate_ranking")
        if direct:
            selected.append(direct)
        structured = _first_matching(rows, method="structured_risk_rerank")
        if structured:
            selected.append(structured)
        srpd_best = _best_by_ndcg([row for row in rows if row.get("evidence_family") == "srpd_trainable_framework"])
        if srpd_best:
            selected.append(srpd_best)
        selected.extend(_classical_rows(rows))
        selected.extend(_paper_project_rows(rows))
        scaffold = _best_by_ndcg([row for row in rows if row.get("evidence_family") == "paper_adapter_scaffold"])
        if scaffold:
            selected.append(scaffold)
        if include_shadow_v6:
            shadow = _first_matching(rows, method="shadow_v6_decision_bridge")
            if shadow:
                selected.append(shadow)

        output_rows.extend(_comparison_row(row) for row in selected)

    return output_rows


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    fields = [
        "domain",
        "table_group",
        "display_method",
        "sample_count",
        "NDCG@10",
        "MRR",
        "delta_vs_domain_direct",
        "delta_vs_week77_structured_risk",
        "artifact_class",
        "paper_role_hint",
    ]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if field in {"NDCG@10", "MRR", "delta_vs_domain_direct", "delta_vs_week77_structured_risk"}:
                value = _fmt_metric(value)
            values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    domains = [domain.strip() for domain in args.domains.split(",") if domain.strip()]
    matrix_path = Path(args.unified_matrix_path).expanduser()
    matrix_rows = _read_csv(matrix_path)
    rows = build_paper_ready_rows(matrix_rows, domains=domains, include_shadow_v6=args.include_shadow_v6)

    output_root = Path(args.output_root).expanduser()
    csv_path = output_root / f"{args.output_name}.csv"
    md_path = output_root / f"{args.output_name}.md"
    _write_csv(rows, csv_path)
    md_path.write_text(_to_markdown(rows), encoding="utf-8")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
