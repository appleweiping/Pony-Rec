from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


FIELDS = [
    "domain",
    "comparison_scope",
    "evidence_family",
    "method",
    "method_variant",
    "sample_count",
    "NDCG@10",
    "MRR",
    "delta_vs_domain_direct",
    "delta_vs_week77_structured_risk",
    "status_label",
    "artifact_class",
    "is_paper_result",
    "paper_role_hint",
    "source_file",
    "notes",
]

DOMAIN_SUFFIX = {
    "beauty": "full973",
    "books": "small500",
    "electronics": "small500",
    "movies": "small500",
}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "\n"
    lines = [
        "| " + " | ".join(FIELDS) + " |",
        "| " + " | ".join(["---"] * len(FIELDS)) + " |",
    ]
    for row in rows:
        values = [str(row.get(field, "")).replace("\n", " ") for field in FIELDS]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _fmt_delta(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.10f}"


def _metric(row: dict[str, str], key: str) -> str:
    return row.get(key, "") or row.get(key.lower(), "")


def _line_count(path: Path) -> str:
    if not path.exists():
        return ""
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return str(count)


def _pick_method(rows: list[dict[str, str]], method: str) -> dict[str, str]:
    for row in rows:
        if row.get("method") == method:
            return row
    return {}


def _pick_uncertainty_rerank(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows:
        if "uncertainty_aware" in str(row.get("method", "")):
            return row
    return rows[-1] if rows else {}


def _add_delta_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    direct_by_domain: dict[str, float] = {}
    structured_by_domain: dict[str, float] = {}
    for row in rows:
        ndcg = _safe_float(row.get("NDCG@10"))
        if ndcg is None:
            continue
        if row.get("method") == "direct_candidate_ranking":
            direct_by_domain[str(row.get("domain"))] = ndcg
        if row.get("method") == "structured_risk_rerank":
            structured_by_domain[str(row.get("domain"))] = ndcg

    for row in rows:
        ndcg = _safe_float(row.get("NDCG@10"))
        domain = str(row.get("domain"))
        if ndcg is not None and domain in direct_by_domain:
            row["delta_vs_domain_direct"] = _fmt_delta(ndcg - direct_by_domain[domain])
        if ndcg is not None and domain in structured_by_domain:
            row["delta_vs_week77_structured_risk"] = _fmt_delta(ndcg - structured_by_domain[domain])
    return rows


def _week77_direct_and_structured_rows(week77_root: Path, domain: str) -> list[dict[str, Any]]:
    suffix = DOMAIN_SUFFIX[domain]
    exp = f"{domain}_qwen3_local_structured_risk_{suffix}"
    path = week77_root / exp / "tables" / "rerank_results.csv"
    records = _read_csv_rows(path)
    direct = _pick_method(records, "direct_candidate_ranking")
    structured = _pick_uncertainty_rerank(records)
    rows: list[dict[str, Any]] = []
    for method, family, record, role in [
        ("direct_candidate_ranking", "week7_7_reference", direct, "same_task_direct_reference"),
        ("structured_risk_rerank", "week7_7_reference", structured, "handcrafted_uncertainty_reference"),
    ]:
        rows.append(
            {
                "domain": domain,
                "comparison_scope": "week7_7_four_domain_export",
                "evidence_family": family,
                "method": method,
                "method_variant": record.get("method", method),
                "sample_count": record.get("sample_count", ""),
                "NDCG@10": _metric(record, "NDCG@10"),
                "MRR": _metric(record, "MRR"),
                "status_label": "server_export_verified" if record else "missing",
                "artifact_class": "paper_candidate_export",
                "is_paper_result": "",
                "paper_role_hint": role,
                "source_file": str(path),
                "notes": "Week7.7 direct/structured-risk row from server export.",
            }
        )
    return rows


def _week77_srpd_rows(week77_root: Path, domain: str) -> list[dict[str, Any]]:
    pattern = f"{domain}_qwen3_rank_srpd_v*_{DOMAIN_SUFFIX[domain]}"
    rows: list[dict[str, Any]] = []
    for exp_dir in sorted(week77_root.glob(pattern)):
        if not exp_dir.is_dir():
            continue
        result_path = exp_dir / "tables" / "framework_eval_summary.csv"
        result_rows = _read_csv_rows(result_path)
        record = result_rows[0] if result_rows else {}
        variant = exp_dir.name.split("_rank_")[-1].replace(f"_{DOMAIN_SUFFIX[domain]}", "")
        prediction_path = exp_dir / "predictions" / "rank_predictions.jsonl"
        rows.append(
            {
                "domain": domain,
                "comparison_scope": "week7_7_four_domain_export",
                "evidence_family": "srpd_trainable_framework",
                "method": "srpd_lora_ranker",
                "method_variant": variant,
                "sample_count": _line_count(prediction_path),
                "NDCG@10": _metric(record, "NDCG@10"),
                "MRR": _metric(record, "MRR"),
                "status_label": "server_export_verified" if record else "missing",
                "artifact_class": "paper_candidate_export",
                "is_paper_result": "",
                "paper_role_hint": "self_trained_lora_framework_candidate",
                "source_file": str(result_path),
                "notes": "SRPD is the current self-trained ranking framework line; compare against external baselines only under same-schema protocol.",
            }
        )
    return rows


def _shadow_rows(shadow_matrix_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _read_csv_rows(shadow_matrix_path):
        scenario = record.get("scenario", "")
        variant = record.get("shadow_variant", "")
        if scenario != "full_replay" or variant not in {"shadow_v1", "shadow_v6"}:
            continue
        is_v6 = variant == "shadow_v6"
        rows.append(
            {
                "domain": record.get("domain", ""),
                "comparison_scope": "week7_9_shadow_diagnostic",
                "evidence_family": "shadow_decision_bridge" if is_v6 else "shadow_signal_candidate",
                "method": "shadow_v6_decision_bridge" if is_v6 else "shadow_v1_signal_rerank",
                "method_variant": variant,
                "sample_count": record.get("sample_count", ""),
                "NDCG@10": record.get("rerank_ndcg_at_10", ""),
                "MRR": record.get("rerank_mrr", ""),
                "status_label": record.get("status", ""),
                "artifact_class": record.get("artifact_class", "diagnostic" if is_v6 else ""),
                "is_paper_result": record.get("is_paper_result", ""),
                "paper_role_hint": "diagnostic_bridge_not_promoted" if is_v6 else "winner_signal_source",
                "source_file": record.get("source_file", str(shadow_matrix_path)),
                "notes": "Shadow rows are diagnostic until aligned with the same paper-result protocol and statistical gates.",
            }
        )
    return rows


def build_unified_method_matrix(
    *,
    week77_root: Path,
    shadow_matrix_path: Path,
    domains: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in domains:
        rows.extend(_week77_direct_and_structured_rows(week77_root, domain))
        rows.extend(_week77_srpd_rows(week77_root, domain))
    rows.extend(_shadow_rows(shadow_matrix_path))
    rows = _add_delta_fields(rows)
    rows.sort(
        key=lambda row: (
            str(row.get("domain", "")),
            str(row.get("comparison_scope", "")),
            str(row.get("evidence_family", "")),
            str(row.get("method_variant", "")),
        )
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a unified matrix over Week7.7 direct/structured-risk/SRPD and shadow_v1/v6 diagnostics."
    )
    parser.add_argument("--week77_root", required=True)
    parser.add_argument("--shadow_matrix_path", default="outputs/summary/shadow_v1_to_v6_status_matrix.csv")
    parser.add_argument("--domains", default="beauty,books,electronics,movies")
    parser.add_argument("--output_root", default="outputs/summary")
    parser.add_argument("--output_name", default="unified_method_matrix_week77_shadow")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    rows = build_unified_method_matrix(
        week77_root=Path(args.week77_root).expanduser(),
        shadow_matrix_path=Path(args.shadow_matrix_path).expanduser(),
        domains=domains,
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
