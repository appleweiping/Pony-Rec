from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_METRICS = ("HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR")
OFFICIAL_METHOD_DIRS = {
    "llmemb": "{exp}_llmemb_official_qwen3base_same_candidate",
    "proex_profile": "{exp}_proex_profile_official_qwen3base_same_candidate",
    "promax_profile": "{exp}_promax_profile_official_qwen3base_same_candidate",
    "elmrec_graph": "{exp}_elmrec_graph_official_qwen3base_same_candidate",
    "irllrec_intent": "{exp}_irllrec_intent_official_qwen3base_same_candidate",
    "rlmrec_graphcl": "{exp}_rlmrec_graphcl_official_qwen3base_same_candidate",
    "llm2rec_sasrec": "{exp}_llm2rec_sasrec_official_qwen3base_same_candidate",
    "llmesr_sasrec": "{exp}_llmesr_sasrec_official_qwen3base_same_candidate",
}
PREDICTION_DELETION_MANIFEST = "prediction_deletion_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a full domain comparison table and paired significance tests "
            "for C-CRP v3 against the eight official same-candidate baselines."
        )
    )
    parser.add_argument("--root", default=".", help="Project root containing outputs/.")
    parser.add_argument("--domain", required=True, help="Domain name, e.g. sports.")
    parser.add_argument("--gate_json", default="", help="Optional prior domain gate JSON; must pass if provided.")
    parser.add_argument("--output_dir", default="outputs/summary", help="Output directory for compact artifacts.")
    parser.add_argument("--stamp", default="", help="Output stamp. Defaults to <domain>_official_ccrp.")
    parser.add_argument("--expected_users", type=int, default=10000)
    parser.add_argument("--expected_candidates_per_user", type=int, default=101)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap_chunk", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _read_single_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, got {len(rows)}")
    return rows[0]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _line_count(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def _certified_deleted_prediction_line_count(base_dir: Path) -> tuple[int | None, str]:
    manifest_path = base_dir / PREDICTION_DELETION_MANIFEST
    if not manifest_path.exists():
        return None, ""
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception:
        return None, ""
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    row = files.get("predictions/rank_predictions.jsonl")
    if not isinstance(row, dict):
        return None, ""
    if (
        manifest.get("mode") == "post_domain_gate_prediction_cleanup"
        and manifest.get("ok") is True
        and manifest.get("failures") == []
        and row.get("deleted") is True
        and row.get("lines") is not None
        and row.get("sha256")
        and row.get("size")
    ):
        return int(row["lines"]), "prediction_deletion_manifest"
    return None, ""


def _certified_prediction_line_count(
    base_dir: Path,
    *,
    allow_deletion_manifest: bool = False,
) -> tuple[int | None, str]:
    prediction_path = base_dir / "predictions" / "rank_predictions.jsonl"
    if prediction_path.exists() and prediction_path.is_file():
        return _line_count(prediction_path), "file"

    if allow_deletion_manifest:
        certified_lines, certified_source = _certified_deleted_prediction_line_count(base_dir)
        if certified_lines is not None:
            return certified_lines, certified_source

    audit_path = base_dir / "server_final_evidence_audit.json"
    if not audit_path.exists():
        return None, ""
    try:
        with audit_path.open("r", encoding="utf-8") as fh:
            audit = json.load(fh)
    except Exception:
        return None, ""
    files = audit.get("files") if isinstance(audit.get("files"), dict) else {}
    row = files.get("predictions/rank_predictions.jsonl")
    if not isinstance(row, dict):
        return None, ""
    if (
        audit.get("mode") == "server_final"
        and audit.get("ok") is True
        and audit.get("failures") == []
        and row.get("present") is True
        and row.get("lines") is not None
    ):
        return int(row["lines"]), "server_final_evidence_audit"
    return None, ""


def _metric_from_rank(rank: int, metric: str) -> float:
    if metric.startswith("HR@"):
        k = int(metric.split("@", 1)[1])
        return float(0 < rank <= k)
    if metric.startswith("NDCG@"):
        k = int(metric.split("@", 1)[1])
        return float(1.0 / math.log2(rank + 1)) if 0 < rank <= k else 0.0
    if metric == "MRR":
        return float(1.0 / rank) if rank > 0 else 0.0
    raise ValueError(f"Unsupported metric: {metric}")


def _read_eval_metric_matrix(path: Path, method: str, expected_users: int) -> tuple[list[str], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    event_ids: list[str] = []
    rows: list[list[float]] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            event_id = row.get("source_event_id") or row.get("event_id") or row.get("user_id") or str(idx)
            if event_id in seen:
                raise ValueError(f"{method}: duplicate event_id {event_id}")
            seen.add(event_id)
            rank = int(float(row["positive_rank"]))
            event_ids.append(event_id)
            rows.append([_metric_from_rank(rank, metric) for metric in REQUIRED_METRICS])
    if len(rows) != expected_users:
        raise ValueError(f"{method}: expected {expected_users} eval records, got {len(rows)}")
    return event_ids, np.asarray(rows, dtype=np.float64)


def _wilcoxon_or_fallback(diff: np.ndarray) -> tuple[float, str]:
    if diff.size == 0:
        return float("nan"), "none"
    if np.allclose(diff, 0.0):
        return 1.0, "all_zero"
    try:
        from scipy.stats import wilcoxon  # type: ignore

        result = wilcoxon(diff, zero_method="pratt", correction=False, alternative="two-sided", method="auto")
        return float(result.pvalue), "wilcoxon_pratt_two_sided"
    except Exception:
        nonzero = diff[np.abs(diff) > 0.0]
        if nonzero.size == 0:
            return 1.0, "all_zero"
        mean = float(np.mean(nonzero))
        std = float(np.std(nonzero, ddof=1)) if nonzero.size > 1 else 0.0
        if std <= 0.0:
            return 0.0 if mean != 0.0 else 1.0, "normal_approx_degenerate"
        z = abs(mean) / (std / math.sqrt(nonzero.size))
        return float(math.erfc(z / math.sqrt(2.0))), "normal_approx_fallback"


def _holm_bonferroni(rows: list[dict[str, Any]], alpha: float) -> None:
    indexed = [(idx, float(row["p_value"])) for idx, row in enumerate(rows) if math.isfinite(float(row["p_value"]))]
    m = len(indexed)
    running_max = 0.0
    for rank, (idx, p_value) in enumerate(sorted(indexed, key=lambda item: item[1]), start=1):
        adjusted = min(1.0, (m - rank + 1) * p_value)
        running_max = max(running_max, adjusted)
        rows[idx]["holm_p_value"] = running_max
        rows[idx]["significant_holm"] = running_max < alpha
    for row in rows:
        row.setdefault("holm_p_value", float("nan"))
        row.setdefault("significant_holm", False)


def _bootstrap_ci(
    diff_matrix: np.ndarray,
    *,
    n_bootstrap: int,
    chunk_size: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_bootstrap <= 0:
        observed = np.mean(diff_matrix, axis=0)
        return observed, observed
    rng = np.random.default_rng(random_state)
    n_events = diff_matrix.shape[0]
    samples: list[np.ndarray] = []
    remaining = n_bootstrap
    while remaining > 0:
        chunk = min(chunk_size, remaining)
        indices = rng.integers(0, n_events, size=(chunk, n_events), endpoint=False)
        samples.append(np.mean(diff_matrix[indices], axis=1))
        remaining -= chunk
    boot = np.vstack(samples)
    return np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def _method_paths(root: Path, domain: str) -> dict[str, dict[str, Any]]:
    exp = f"{domain}_large10000_100neg"
    rows: dict[str, dict[str, Any]] = {
        "ccrp_v3_qwen3base_pointwise": {
            "kind": "internal_method",
            "dir": root / "outputs" / f"{exp}_ccrp_v3_qwen3base_pointwise_same_candidate",
        }
    }
    for method, template in OFFICIAL_METHOD_DIRS.items():
        rows[method] = {
            "kind": "official_baseline",
            "dir": root / "outputs" / template.format(exp=exp),
        }
    for method, row in rows.items():
        base = Path(row["dir"])
        row["summary_path"] = base / "tables" / "same_candidate_external_baseline_summary.csv"
        row["eval_path"] = base / "tables" / "ranking_eval_records.csv"
        row["metrics_path"] = base / "tables" / "ranking_metrics.csv"
        row["coverage_path"] = base / "tables" / "external_score_coverage.csv"
        row["prediction_path"] = base / "predictions" / "rank_predictions.jsonl"
        if row["kind"] == "official_baseline":
            row["provenance_path"] = base / "fairness_provenance.json"
            row["scores_path"] = base / "scores.csv"
        else:
            raw = root / "outputs" / f"{exp}_ccrp_v3"
            row["provenance_path"] = ""
            row["scores_path"] = raw / "scores.csv"
            row["raw_user_ranks_path"] = raw / "user_ranks.jsonl"
    return rows


def _load_gate(gate_json: Path) -> dict[str, Any]:
    if not gate_json:
        return {}
    with gate_json.open("r", encoding="utf-8") as fh:
        gate = json.load(fh)
    if not gate.get("gate_ok"):
        raise RuntimeError(f"Domain gate did not pass: {gate_json}")
    return gate


def _build_comparison_rows(
    paths: dict[str, dict[str, Any]],
    *,
    domain: str,
    expected_users: int,
    expected_candidates_per_user: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, list[str]]]:
    rows: list[dict[str, Any]] = []
    frames: dict[str, np.ndarray] = {}
    event_ids: dict[str, list[str]] = {}
    expected_score_lines = expected_users * expected_candidates_per_user + 1

    for method, info in paths.items():
        summary = _read_single_csv(Path(info["summary_path"]))
        metrics = _read_single_csv(Path(info["metrics_path"]))
        ids, frame = _read_eval_metric_matrix(Path(info["eval_path"]), method, expected_users)
        frames[method] = frame
        event_ids[method] = ids

        row: dict[str, Any] = {
            "domain": domain,
            "method": method,
            "kind": info["kind"],
            "status_label": summary.get("status_label", ""),
            "artifact_class": summary.get("artifact_class", ""),
            "score_coverage_rate": summary.get("score_coverage_rate", ""),
            "sample_count": metrics.get("sample_count", summary.get("sample_count", "")),
            "avg_candidates": metrics.get("avg_candidates", summary.get("avg_candidates", "")),
            "ranking_events": summary.get("ranking_events", ""),
            "total_candidates": summary.get("total_candidates", ""),
            "matched_candidates": summary.get("matched_candidates", ""),
            "summary_path": str(info["summary_path"]),
            "eval_path": str(info["eval_path"]),
            "scores_path": str(info["scores_path"]),
            "scores_csv_lines": _line_count(Path(info["scores_path"])),
            "ranking_eval_records_csv_lines": _line_count(Path(info["eval_path"])),
        }
        prediction_lines, prediction_source = _certified_prediction_line_count(
            Path(info["dir"]),
            allow_deletion_manifest=info["kind"] == "internal_method",
        )
        row["predictions_jsonl_lines"] = prediction_lines
        row["predictions_jsonl_line_source"] = prediction_source
        if row["scores_csv_lines"] != expected_score_lines:
            raise ValueError(f"{method}: score line count mismatch {row['scores_csv_lines']}")
        if row["predictions_jsonl_lines"] != expected_users:
            raise ValueError(f"{method}: prediction line count mismatch {row['predictions_jsonl_lines']}")
        for metric in REQUIRED_METRICS:
            summary_value = _as_float(summary.get(metric))
            event_value = float(np.mean(frame[:, REQUIRED_METRICS.index(metric)]))
            if math.isfinite(summary_value) and abs(summary_value - event_value) > 1e-12:
                raise ValueError(f"{method}: {metric} summary/event mismatch {summary_value} vs {event_value}")
            row[metric] = event_value
        rows.append(row)

    rows.sort(key=lambda row: (-float(row["NDCG@10"]), row["method"]))
    for idx, row in enumerate(rows, start=1):
        row["rank_by_NDCG@10"] = idx
    return rows, frames, event_ids


def _paired_tests(
    frames: dict[str, np.ndarray],
    event_ids: dict[str, list[str]],
    *,
    method_name: str,
    baselines: list[str],
    n_bootstrap: int,
    bootstrap_chunk: int,
    random_state: int,
    alpha: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    method_ids = event_ids[method_name]
    method_frame = frames[method_name]
    for baseline_idx, baseline in enumerate(baselines):
        if event_ids[baseline] != method_ids:
            raise ValueError(f"Event id order mismatch: {method_name} vs {baseline}")
        baseline_frame = frames[baseline]
        diff_matrix = method_frame - baseline_frame
        ci_low, ci_high = _bootstrap_ci(
            diff_matrix,
            n_bootstrap=n_bootstrap,
            chunk_size=bootstrap_chunk,
            random_state=random_state + baseline_idx,
        )
        for metric_idx, metric in enumerate(REQUIRED_METRICS):
            diff = diff_matrix[:, metric_idx]
            p_value, test_name = _wilcoxon_or_fallback(diff)
            std = float(np.std(diff, ddof=1)) if diff.size > 1 else float("nan")
            rows.append(
                {
                    "baseline": baseline,
                    "method": method_name,
                    "metric": metric,
                    "n_paired_events": int(diff.size),
                    "baseline_mean": float(np.mean(baseline_frame[:, metric_idx])),
                    "method_mean": float(np.mean(method_frame[:, metric_idx])),
                    "delta": float(np.mean(diff)),
                    "ci_low": float(ci_low[metric_idx]),
                    "ci_high": float(ci_high[metric_idx]),
                    "p_value": p_value,
                    "test": test_name,
                    "effect_cohen_dz": float(np.mean(diff) / std) if std > 0.0 else float("nan"),
                    "win_rate": float(np.mean(diff > 0.0)),
                    "loss_rate": float(np.mean(diff < 0.0)),
                    "tie_rate": float(np.mean(diff == 0.0)),
                }
            )
    _holm_bonferroni(rows, alpha)
    for row in rows:
        row["winner_label"] = (
            "ccrp_significant_positive"
            if row["delta"] > 0.0 and row["ci_low"] > 0.0 and row["significant_holm"]
            else "not_significant_positive"
        )
    return rows


def _best_baseline_by_metric(comparison_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    baseline_rows = [row for row in comparison_rows if row["kind"] == "official_baseline"]
    best: dict[str, dict[str, Any]] = {}
    for metric in REQUIRED_METRICS:
        row = max(baseline_rows, key=lambda item: float(item[metric]))
        best[metric] = {"baseline": row["method"], "value": float(row[metric])}
    return best


def _write_markdown(
    path: Path,
    *,
    domain: str,
    comparison_rows: list[dict[str, Any]],
    stats_summary: dict[str, Any],
    paired_rows: list[dict[str, Any]],
) -> None:
    metric_cols = ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]
    lines = [
        f"# {domain} official baseline vs C-CRP comparison",
        "",
        "Read-only same-candidate comparison. This table does not start or rerun experiments.",
        "",
        "| Rank | Method | Kind | " + " | ".join(metric_cols) + " |",
        "|---:|---|---|" + "|".join(["---:"] * len(metric_cols)) + "|",
    ]
    for row in comparison_rows:
        values = " | ".join(f"{float(row[col]):.6f}" for col in metric_cols)
        lines.append(f"| {row['rank_by_NDCG@10']} | `{row['method']}` | {row['kind']} | {values} |")
    lines.extend(
        [
            "",
            "## Gate Summary",
            "",
            f"- Observed C-CRP best on all seven metrics: `{stats_summary['ccrp_observed_best_all_metrics']}`",
            f"- Holm-significant positive C-CRP deltas for all C-CRP-vs-official tests: `{stats_summary['paired_all_positive_significant']}`",
            f"- Number of paired tests: `{stats_summary['paired_test_count']}`",
            f"- Minimum delta across tests: `{stats_summary['min_delta']:.12f}`",
            f"- Maximum Holm-adjusted p-value across tests: `{stats_summary['max_holm_p_value']:.12g}`",
            "",
            "## Closest Official Baseline By Metric",
            "",
            "| Metric | Best official baseline | Baseline | C-CRP | Delta | Holm p | 95% CI |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in stats_summary["closest_baseline_tests"]:
        lines.append(
            f"| {item['metric']} | `{item['baseline']}` | {item['baseline_mean']:.6f} | "
            f"{item['method_mean']:.6f} | {item['delta']:.6f} | "
            f"{item['holm_p_value']:.6g} | [{item['ci_low']:.6f}, {item['ci_high']:.6f}] |"
        )
    lines.extend(
        [
            "",
            f"Claim note: this is a {domain}-domain statistical gate. Multi-domain paper-level SOTA wording still requires the declared domain set, aligned baselines, and ARIS review.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser()
    stamp = args.stamp or f"{args.domain}_official_ccrp"
    gate: dict[str, Any] = {}
    if args.gate_json:
        gate = _load_gate(Path(args.gate_json).expanduser())

    paths = _method_paths(root, args.domain)
    comparison_rows, frames, event_ids = _build_comparison_rows(
        paths,
        domain=args.domain,
        expected_users=args.expected_users,
        expected_candidates_per_user=args.expected_candidates_per_user,
    )
    method_name = "ccrp_v3_qwen3base_pointwise"
    baselines = list(OFFICIAL_METHOD_DIRS.keys())
    paired_rows = _paired_tests(
        frames,
        event_ids,
        method_name=method_name,
        baselines=baselines,
        n_bootstrap=args.n_bootstrap,
        bootstrap_chunk=args.bootstrap_chunk,
        random_state=args.random_state,
        alpha=args.alpha,
    )

    ccrp_row = next(row for row in comparison_rows if row["method"] == method_name)
    best_baseline = _best_baseline_by_metric(comparison_rows)
    observed_best = all(float(ccrp_row[metric]) > best_baseline[metric]["value"] for metric in REQUIRED_METRICS)
    all_positive_significant = all(row["winner_label"] == "ccrp_significant_positive" for row in paired_rows)
    closest_tests = []
    for metric, best in best_baseline.items():
        match = next(row for row in paired_rows if row["baseline"] == best["baseline"] and row["metric"] == metric)
        closest_tests.append(match)

    stats_summary = {
        "domain": args.domain,
        "gate_json": str(Path(args.gate_json).expanduser()) if args.gate_json else "",
        "prior_gate_ok": gate.get("gate_ok", "") if gate else "",
        "ccrp_observed_best_all_metrics": observed_best,
        "paired_all_positive_significant": all_positive_significant,
        "paired_test_count": len(paired_rows),
        "holm_family": "all_CCRP_vs_8_official_x_7_metrics",
        "alpha": args.alpha,
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_ci": "paired_event_bootstrap_percentile_95",
        "test": "wilcoxon_pratt_two_sided_with_fallback",
        "min_delta": min(float(row["delta"]) for row in paired_rows),
        "min_ci_low": min(float(row["ci_low"]) for row in paired_rows),
        "max_holm_p_value": max(float(row["holm_p_value"]) for row in paired_rows),
        "ccrp_rank_by_NDCG@10": ccrp_row["rank_by_NDCG@10"],
        "best_official_by_metric": best_baseline,
        "closest_baseline_tests": closest_tests,
        "claim_gate": (
            f"{args.domain}_domain_pass"
            if observed_best and all_positive_significant
            else f"{args.domain}_domain_incomplete_or_not_significant"
        ),
        "claim_scope_warning": (
            f"{args.domain}-only gate. Do not generalize to the paper's full domain set until all declared "
            "domains, comparison tables, paired tests, and ARIS review gates are complete."
        ),
    }

    comparison_path = output_dir / f"{stamp}_comparison.csv"
    paired_path = output_dir / f"{stamp}_paired_tests.csv"
    summary_path = output_dir / f"{stamp}_paired_summary.json"
    markdown_path = output_dir / f"{stamp}_comparison.md"

    comparison_fields = [
        "rank_by_NDCG@10",
        "domain",
        "method",
        "kind",
        *REQUIRED_METRICS,
        "sample_count",
        "avg_candidates",
        "score_coverage_rate",
        "ranking_events",
        "total_candidates",
        "matched_candidates",
        "scores_csv_lines",
        "predictions_jsonl_lines",
        "predictions_jsonl_line_source",
        "ranking_eval_records_csv_lines",
        "status_label",
        "artifact_class",
        "summary_path",
        "eval_path",
        "scores_path",
    ]
    paired_fields = [
        "baseline",
        "method",
        "metric",
        "n_paired_events",
        "baseline_mean",
        "method_mean",
        "delta",
        "ci_low",
        "ci_high",
        "p_value",
        "holm_p_value",
        "significant_holm",
        "winner_label",
        "test",
        "effect_cohen_dz",
        "win_rate",
        "loss_rate",
        "tie_rate",
    ]
    _write_csv(comparison_path, comparison_rows, comparison_fields)
    _write_csv(paired_path, paired_rows, paired_fields)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(stats_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(
        markdown_path,
        domain=args.domain,
        comparison_rows=comparison_rows,
        stats_summary=stats_summary,
        paired_rows=paired_rows,
    )

    if not args.quiet:
        print(json.dumps(stats_summary, indent=2, ensure_ascii=False))
    return 0 if stats_summary["claim_gate"] == f"{args.domain}_domain_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
