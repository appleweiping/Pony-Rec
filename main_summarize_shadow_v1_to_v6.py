from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


FIELDS = [
    "evidence_layer",
    "scenario",
    "domain",
    "shadow_variant",
    "winner_signal_variant",
    "status",
    "pointwise_status",
    "calibration_status",
    "rerank_status",
    "noisy_pointwise_status",
    "noisy_rerank_status",
    "sample_count",
    "pointwise_auroc",
    "pointwise_ece",
    "calibrated_ece",
    "rerank_ndcg_at_10",
    "rerank_mrr",
    "noisy_pointwise_auroc",
    "noisy_rerank_ndcg_at_10",
    "ndcg_drop_noisy",
    "direct_ndcg_at_10",
    "direct_mrr",
    "delta_ndcg_at_10",
    "delta_mrr",
    "changed_ranking_fraction",
    "avg_position_shift",
    "matched_signal_rate",
    "fallback_rate",
    "mean_correction_gate",
    "mean_pair_weight",
    "bridge_rows",
    "artifact_class",
    "is_paper_result",
    "source_file",
]


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


def _status_from_row(row: dict[str, str]) -> str:
    statuses = [
        row.get("pointwise_status", ""),
        row.get("calibration_status", ""),
        row.get("rerank_status", ""),
    ]
    noisy_statuses = [
        row.get("noisy_pointwise_status", ""),
        row.get("noisy_rerank_status", ""),
    ]
    if all(status == "ready" for status in statuses + noisy_statuses):
        return "ready_with_noisy"
    if all(status == "ready" for status in statuses):
        return "ready_clean"
    if any(status == "ready" for status in statuses + noisy_statuses):
        return "partial"
    return "missing"


def _normalize_shadow_summary_row(row: dict[str, str], source_file: Path) -> dict[str, Any]:
    return {
        "evidence_layer": "signal_candidate",
        "scenario": row.get("scenario", ""),
        "domain": row.get("domain", ""),
        "shadow_variant": row.get("shadow_variant", ""),
        "winner_signal_variant": "",
        "status": _status_from_row(row),
        "pointwise_status": row.get("pointwise_status", ""),
        "calibration_status": row.get("calibration_status", ""),
        "rerank_status": row.get("rerank_status", ""),
        "noisy_pointwise_status": row.get("noisy_pointwise_status", ""),
        "noisy_rerank_status": row.get("noisy_rerank_status", ""),
        "sample_count": "",
        "pointwise_auroc": row.get("pointwise_auroc", ""),
        "pointwise_ece": row.get("pointwise_ece", ""),
        "calibrated_ece": row.get("calibrated_ece", ""),
        "rerank_ndcg_at_10": row.get("rerank_ndcg_at_10", ""),
        "rerank_mrr": row.get("rerank_mrr", ""),
        "noisy_pointwise_auroc": row.get("noisy_pointwise_auroc", ""),
        "noisy_rerank_ndcg_at_10": row.get("noisy_rerank_ndcg_at_10", ""),
        "ndcg_drop_noisy": row.get("ndcg_drop_noisy", ""),
        "source_file": str(source_file),
    }


def _pick_method(rows: list[dict[str, str]], method: str) -> dict[str, str]:
    for row in rows:
        if row.get("method") == method:
            return row
    return {}


def _build_v6_row(domain: str, v6_output_root: Path) -> dict[str, Any]:
    exp_name = f"{domain}_qwen3_shadow_v6_full_replay_structured_risk"
    result_path = v6_output_root / exp_name / "tables" / "rerank_results.csv"
    summary_path = v6_output_root / exp_name / "tables" / "shadow_v6_bridge_summary.csv"
    result_rows = _read_csv_rows(result_path)
    summary_rows = _read_csv_rows(summary_path)
    direct = _pick_method(result_rows, "direct_candidate_ranking")
    v6 = _pick_method(result_rows, "shadow_v6_decision_bridge")
    bridge = summary_rows[0] if summary_rows else {}

    direct_ndcg = _safe_float(direct.get("NDCG@10"))
    direct_mrr = _safe_float(direct.get("MRR"))
    v6_ndcg = _safe_float(v6.get("NDCG@10"))
    v6_mrr = _safe_float(v6.get("MRR"))
    status = "ready" if direct and v6 and bridge else "missing"

    return {
        "evidence_layer": "decision_bridge",
        "scenario": "full_replay",
        "domain": domain,
        "shadow_variant": "shadow_v6",
        "winner_signal_variant": v6.get("winner_signal_variant") or bridge.get("winner_signal_variant", ""),
        "status": status,
        "pointwise_status": "",
        "calibration_status": "",
        "rerank_status": "ready" if v6 else "missing",
        "noisy_pointwise_status": "",
        "noisy_rerank_status": "",
        "sample_count": v6.get("sample_count", ""),
        "pointwise_auroc": "",
        "pointwise_ece": "",
        "calibrated_ece": "",
        "rerank_ndcg_at_10": v6.get("NDCG@10", ""),
        "rerank_mrr": v6.get("MRR", ""),
        "noisy_pointwise_auroc": "",
        "noisy_rerank_ndcg_at_10": "",
        "ndcg_drop_noisy": "",
        "direct_ndcg_at_10": direct.get("NDCG@10", ""),
        "direct_mrr": direct.get("MRR", ""),
        "delta_ndcg_at_10": _fmt_delta(v6_ndcg - direct_ndcg) if v6_ndcg is not None and direct_ndcg is not None else "",
        "delta_mrr": _fmt_delta(v6_mrr - direct_mrr) if v6_mrr is not None and direct_mrr is not None else "",
        "changed_ranking_fraction": v6.get("changed_ranking_fraction", ""),
        "avg_position_shift": v6.get("avg_position_shift", ""),
        "matched_signal_rate": v6.get("matched_signal_rate") or bridge.get("matched_signal_rate", ""),
        "fallback_rate": v6.get("fallback_rate") or bridge.get("fallback_rate", ""),
        "mean_correction_gate": v6.get("mean_correction_gate") or bridge.get("mean_correction_gate", ""),
        "mean_pair_weight": v6.get("mean_pair_weight") or bridge.get("mean_pair_weight", ""),
        "bridge_rows": v6.get("bridge_rows") or bridge.get("bridge_rows", ""),
        "artifact_class": v6.get("artifact_class") or bridge.get("artifact_class", ""),
        "is_paper_result": v6.get("is_paper_result") or bridge.get("is_paper_result", ""),
        "source_file": str(result_path),
    }


def build_shadow_v1_to_v6_matrix(
    *,
    shadow_summary_root: Path,
    v6_output_root: Path,
    domains: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for filename in [
        "week7_9_shadow_small_prior_summary.csv",
        "week7_9_shadow_full_replay_summary.csv",
    ]:
        path = shadow_summary_root / filename
        for row in _read_csv_rows(path):
            rows.append(_normalize_shadow_summary_row(row, path))

    for domain in domains:
        rows.append(_build_v6_row(domain, v6_output_root))

    rows.sort(
        key=lambda row: (
            str(row.get("scenario", "")),
            str(row.get("domain", "")),
            str(row.get("evidence_layer", "")),
            str(row.get("shadow_variant", "")),
        )
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize shadow v1-v5 signal candidates and shadow_v6 bridge diagnostics."
    )
    parser.add_argument(
        "--shadow_summary_root",
        default="outputs/summary",
        help="Directory containing week7_9_shadow_*_summary.csv files.",
    )
    parser.add_argument(
        "--v6_output_root",
        default="outputs",
        help="Directory containing shadow_v6 bridge output experiment folders.",
    )
    parser.add_argument("--domains", default="beauty,books,electronics,movies")
    parser.add_argument("--output_root", default="outputs/summary")
    parser.add_argument("--output_name", default="shadow_v1_to_v6_status_matrix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    rows = build_shadow_v1_to_v6_matrix(
        shadow_summary_root=Path(args.shadow_summary_root).expanduser(),
        v6_output_root=Path(args.v6_output_root).expanduser(),
        domains=domains,
    )
    output_root = Path(args.output_root).expanduser()
    csv_path = output_root / f"{args.output_name}.csv"
    md_path = output_root / f"{args.output_name}.md"
    _write_csv(rows, csv_path)
    md_path.write_text(_to_markdown(rows), encoding="utf-8")

    ready_rows = sum(1 for row in rows if str(row.get("status", "")).startswith("ready"))
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")
    print(f"rows={len(rows)} ready_rows={ready_rows}")


if __name__ == "__main__":
    main()
