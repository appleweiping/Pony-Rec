from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path
from typing import Any


METHOD_ORDER = [
    "sasrec",
    "gru4rec",
    "bert4rec",
    "lightgcn",
    "llm2rec_official_qwen3base_sasrec",
    "llmesr_official_qwen3base_sasrec",
    "llmemb_official_qwen3base",
    "rlmrec_official_qwen3base_graphcl",
    "irllrec_official_qwen3base_intent",
    "setrec_official_qwen3base_identifier",
    "llm2rec_style_qwen3_sasrec",
    "llmesr_style_qwen3_sasrec",
    "llmemb_style_qwen3_sasrec",
    "rlmrec_style_qwen3_graphcl",
    "irllrec_style_qwen3_intent",
    "setrec_style_qwen3_identifier",
]

OUTPUT_FIELDS = [
    "domain",
    "display_method",
    "method",
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
    "status_label",
    "artifact_class",
    "source_file",
]

METRIC_FIELDS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact comparison table from completed external same-candidate baselines only."
    )
    parser.add_argument("--external_summary_glob", required=True)
    parser.add_argument("--domains", default="books,electronics,movies")
    parser.add_argument("--methods", default=",".join(METHOD_ORDER))
    parser.add_argument("--output_root", default="outputs/summary")
    parser.add_argument("--output_name", default="external_only_baseline_comparison")
    return parser.parse_args()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
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


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _method_label(method: str) -> str:
    return {
        "sasrec": "SASRec",
        "gru4rec": "GRU4Rec",
        "bert4rec": "BERT4Rec",
        "lightgcn": "LightGCN",
        "llm2rec_official_qwen3base_sasrec": "LLM2Rec official Qwen3-8B + SASRec",
        "llmesr_official_qwen3base_sasrec": "LLM-ESR official Qwen3-8B + LLMESR-SASRec",
        "llmemb_official_qwen3base": "LLMEmb official Qwen3-8B",
        "rlmrec_official_qwen3base_graphcl": "RLMRec official Qwen3-8B GraphCL",
        "irllrec_official_qwen3base_intent": "IRLLRec official Qwen3-8B IntentRep",
        "setrec_official_qwen3base_identifier": "SETRec official Qwen3-8B Identifier",
        "llm2rec_style_qwen3_sasrec": "LLM2Rec-style Qwen3-8B Emb. + SASRec",
        "llmesr_style_qwen3_sasrec": "LLM-ESR-style Qwen3-8B Emb. + LLMESR-SASRec",
        "llmemb_style_qwen3_sasrec": "LLMEmb-style Qwen3-8B Emb. + SASRec",
        "rlmrec_style_qwen3_graphcl": "RLMRec-style Qwen3-8B GraphCL",
        "irllrec_style_qwen3_intent": "IRLLRec-style Qwen3-8B IntentRep",
        "setrec_style_qwen3_identifier": "SETRec-style Qwen3-8B Identifier",
    }.get(method, method)


def _metric_completeness(row: dict[str, Any]) -> tuple[int, int]:
    populated = sum(1 for field in METRIC_FIELDS if str(row.get(field, "")).strip())
    source_specificity = 1 if "large10000_100neg" in str(row.get("source_file", "")) or "supplementary_smallerN_100neg" in str(row.get("source_file", "")) else 0
    return populated, source_specificity


def build_external_only_rows(
    *,
    external_summary_glob: str,
    domains: list[str],
    methods: list[str],
) -> list[dict[str, Any]]:
    rows_by_domain_method: dict[tuple[str, str], dict[str, Any]] = {}
    for path_text in sorted(glob.glob(external_summary_glob)):
        path = Path(path_text).expanduser()
        if not path.exists():
            continue
        for record in _read_csv_rows(path):
            if record.get("status_label") != "same_schema_external_baseline":
                continue
            if record.get("artifact_class") != "completed_result":
                continue
            domain = record.get("domain", "").strip()
            method = record.get("baseline_name", "").strip()
            if domain not in domains or method not in methods:
                continue
            row = {
                "domain": domain,
                "display_method": _method_label(method),
                "method": method,
                "sample_count": record.get("sample_count", ""),
                "HR@5": record.get("HR@5", ""),
                "NDCG@5": record.get("NDCG@5", ""),
                "HR@10": record.get("HR@10", ""),
                "NDCG@10": record.get("NDCG@10", ""),
                "HR@20": record.get("HR@20", ""),
                "NDCG@20": record.get("NDCG@20", ""),
                "MRR": record.get("MRR", ""),
                "coverage@5": record.get("coverage@5", ""),
                "coverage@10": record.get("coverage@10", ""),
                "coverage@20": record.get("coverage@20", ""),
                "head_exposure_ratio@10": record.get("head_exposure_ratio@10", ""),
                "longtail_coverage@10": record.get("longtail_coverage@10", ""),
                "status_label": record.get("status_label", ""),
                "artifact_class": record.get("artifact_class", ""),
                "source_file": str(path),
            }
            key = (domain, method)
            current = rows_by_domain_method.get(key)
            if current is None or _metric_completeness(row) >= _metric_completeness(current):
                rows_by_domain_method[key] = row

    output: list[dict[str, Any]] = []
    for domain in domains:
        domain_rows = [
            rows_by_domain_method[(domain, method)]
            for method in methods
            if (domain, method) in rows_by_domain_method
        ]
        output.extend(domain_rows)
    return output


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    fields = [
        "domain",
        "display_method",
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
        "artifact_class",
    ]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if field in {"HR@5", "NDCG@5", "HR@10", "NDCG@10", "HR@20", "NDCG@20", "MRR", "coverage@5", "coverage@10", "coverage@20"}:
                value = _fmt_metric(value)
            values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = build_external_only_rows(
        external_summary_glob=args.external_summary_glob,
        domains=_parse_csv_list(args.domains),
        methods=_parse_csv_list(args.methods),
    )
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
