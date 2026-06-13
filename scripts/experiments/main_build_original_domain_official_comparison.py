from __future__ import annotations

"""Paired Holm-corrected bootstrap significance tests for C-CRP vs the 8 official
baselines on the ORIGINAL-4 domains (beauty, books, electronics, movies).

This GENERALIZES scripts/experiments/main_build_domain_official_comparison.py
(used for the new-4 domains sports/toys/home/tools) to the original-4. It MATCHES
that method exactly:

  * paired per-event bootstrap, 2000 resamples, 95% percentile CI
  * Wilcoxon signed-rank (Pratt, two-sided) p-value with normal-approx fallback
  * Holm correction at alpha=0.05 over the 56-test family per domain
  * 7 metrics (HR@5/10/20, NDCG@5/10/20, MRR) x 8 official baselines = 56 tests

Differences from the new-4 script, required by the original-4 layout:
  1. C-CRP per-event source = outputs/ccrp_v3_formal/<domain>/user_ranks.jsonl
     (pos_rank field), which is the AUTHORITATIVE source that matches the
     published outputs/ccrp_v3_formal/main_comparison_table.csv numbers. (The
     <domain>_ccrp_formal_selected_same_candidate dir is a stale variant whose
     NDCG@10 does NOT match the published table -> NOT used.)
  2. Per-domain canonical baseline dir templates (beauty uses the
     supplementary_smallerN layout with 973 users; books/electronics/movies use
     the large10000_100neg layout with 10000 users).
  3. CANONICAL-DIR VERIFICATION: for every domain x baseline, the selected dir's
     tables/ranking_metrics.csv NDCG@10 and HR@10 MUST match the published
     main_comparison_table.csv row (tol 1e-6). Any mismatch / missing baseline is
     REPORTED explicitly (never silently dropped or substituted).
  4. SIGNED reporting: C-CRP is rank-2 on beauty and rank-5 on movies, so for
     every domain we report how many of the 56 paired deltas are
     positive-&-Holm-significant, negative-&-Holm-significant (C-CRP
     significantly BEHIND), and non-significant.

CPU ONLY. Reads ~1GB ranking_eval_records.csv server-side; writes only small JSON.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_METRICS = ("HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR")

# Published-table method name -> baseline-dir variant suffix (the bit after
# official_qwen3base in the dir name). llmemb has no variant suffix.
OFFICIAL_METHODS = {
    "llmemb": "",
    "proex_profile": "profile",
    "promax_profile": "profile",
    "elmrec_graph": "graph",
    "irllrec_intent": "intent",
    "rlmrec_graphcl": "graphcl",
    "llm2rec_sasrec": "sasrec",
    "llmesr_sasrec": "sasrec",
}

# Per-domain exp-prefix and expected user count.
DOMAIN_CONFIG = {
    "beauty": {"prefix": "beauty_supplementary_smallerN_100neg", "expected_users": 973},
    "books": {"prefix": "books_large10000_100neg", "expected_users": 10000},
    "electronics": {"prefix": "electronics_large10000_100neg", "expected_users": 10000},
    "movies": {"prefix": "movies_large10000_100neg", "expected_users": 10000},
}

# Method name as it appears in main_comparison_table.csv -> our internal key.
PUBLISHED_NAME = {
    "llmemb": "llmemb",
    "proex_profile": "proex_profile",
    "promax_profile": "promax_profile",
    "elmrec_graph": "elmrec_graph",
    "irllrec_intent": "irllrec_intent",
    "rlmrec_graphcl": "rlmrec_graphcl",
    "llm2rec_sasrec": "llm2rec_sasrec",
    "llmesr_sasrec": "llmesr_sasrec",
}
CCRP_PUBLISHED_NAME = "C-CRP_v3_ours"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Project root containing outputs/.")
    parser.add_argument("--domain", required=True, choices=sorted(DOMAIN_CONFIG.keys()))
    parser.add_argument("--output_dir", default="outputs/summary/paper_critical/significance_all8")
    parser.add_argument("--stamp", default="", help="Defaults to <domain>_original_official_ccrp.")
    parser.add_argument("--expected_candidates_per_user", type=int, default=101)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap_chunk", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--match_tol", type=float, default=1e-6)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


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


def _read_published_table(root: Path, domain: str) -> dict[str, dict[str, float]]:
    path = root / "outputs" / "ccrp_v3_formal" / "main_comparison_table.csv"
    out: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("domain") != domain:
                continue
            out[row["method"]] = {m: _as_float(row.get(m)) for m in REQUIRED_METRICS}
    return out


def _read_eval_records_csv(path: Path) -> tuple[list[str], dict[str, int]]:
    """Return ordered user_ids and {user_id: positive_rank} from a baseline dir."""
    user_ids: list[str] = []
    ranks: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            uid = row.get("user_id") or row.get("source_event_id") or str(idx)
            if uid in ranks:
                raise ValueError(f"duplicate user_id {uid} in {path}")
            rank = int(float(row["positive_rank"]))
            user_ids.append(uid)
            ranks[uid] = rank
    return user_ids, ranks


def _read_ccrp_user_ranks(path: Path) -> dict[str, int]:
    """Read C-CRP per-event ranks from outputs/ccrp_v3_formal/<domain>/user_ranks.jsonl.

    IMPORTANT: this file stores pos_rank as a 0-INDEXED position (0..n_candidates-1),
    whereas the baseline ranking_eval_records.csv stores positive_rank as 1-INDEXED.
    Adding 1 here reproduces the published report.json / main_comparison_table.csv
    metrics EXACTLY for all four original domains (verified by _verify_ccrp_means).
    """
    ranks: dict[str, int] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = rec["user_id"]
            if uid in ranks:
                raise ValueError(f"duplicate ccrp user_id {uid}")
            ranks[uid] = int(rec["pos_rank"]) + 1  # 0-indexed -> 1-indexed
    return ranks


def _ranks_to_matrix(ranks: dict[str, int], order: list[str]) -> np.ndarray:
    rows = [[_metric_from_rank(ranks[uid], m) for m in REQUIRED_METRICS] for uid in order]
    return np.asarray(rows, dtype=np.float64)


def _read_ranking_metrics(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1:
        raise ValueError(f"expected 1 row in {path}, got {len(rows)}")
    return {m: _as_float(rows[0].get(m)) for m in REQUIRED_METRICS}


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


def _baseline_dir(root: Path, domain: str, method: str) -> Path:
    prefix = DOMAIN_CONFIG[domain]["prefix"]
    variant = OFFICIAL_METHODS[method]
    base_method = method.rsplit("_", 1)[0] if variant and method.endswith("_" + variant) else method
    # method internal keys: e.g. proex_profile -> base proex, variant profile.
    # Build "<base>_official_qwen3base[_<variant>]".
    if variant:
        tail = f"{base_method}_official_qwen3base_{variant}_same_candidate"
    else:
        tail = f"{base_method}_official_qwen3base_same_candidate"
    return root / "outputs" / f"{prefix}_{tail}"


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    domain = args.domain
    cfg = DOMAIN_CONFIG[domain]
    expected_users = cfg["expected_users"]
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = args.stamp or f"{domain}_original_official_ccrp"

    published = _read_published_table(root, domain)
    report: dict[str, Any] = {
        "domain": domain,
        "expected_users": expected_users,
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_ci": "paired_event_bootstrap_percentile_95",
        "test": "wilcoxon_pratt_two_sided_with_fallback",
        "holm_family": "all_CCRP_vs_8_official_x_7_metrics",
        "alpha": args.alpha,
        "metrics": list(REQUIRED_METRICS),
        "ccrp_source": f"outputs/ccrp_v3_formal/{domain}/user_ranks.jsonl",
        "canonical_dir_verification": [],
        "errors": [],
    }

    # ---- C-CRP source (authoritative, matches published table) ----
    ccrp_path = root / "outputs" / "ccrp_v3_formal" / domain / "user_ranks.jsonl"
    if not ccrp_path.exists():
        report["errors"].append(f"MISSING C-CRP source {ccrp_path}")
        _dump(output_dir, stamp, report)
        return 2
    ccrp_ranks = _read_ccrp_user_ranks(ccrp_path)
    if len(ccrp_ranks) != expected_users:
        report["errors"].append(f"C-CRP user count {len(ccrp_ranks)} != expected {expected_users}")

    # ---- baselines: select canonical dir, verify against published table ----
    baseline_ranks: dict[str, dict[str, int]] = {}
    common_order: list[str] | None = None
    for method in OFFICIAL_METHODS:
        if method not in published:
            report["errors"].append(f"{method}: not in published main_comparison_table for {domain}")
            continue
        bdir = _baseline_dir(root, domain, method)
        metrics_path = bdir / "tables" / "ranking_metrics.csv"
        eval_path = bdir / "tables" / "ranking_eval_records.csv"
        ver: dict[str, Any] = {"method": method, "dir": str(bdir)}
        if not metrics_path.exists() or not eval_path.exists():
            ver["status"] = "MISSING_DIR_OR_TABLES"
            report["canonical_dir_verification"].append(ver)
            report["errors"].append(f"{method}: missing {metrics_path} or {eval_path}")
            continue
        published_row = published[method]
        observed = _read_ranking_metrics(metrics_path)
        d_ndcg10 = abs(observed["NDCG@10"] - published_row["NDCG@10"])
        d_hr10 = abs(observed["HR@10"] - published_row["HR@10"])
        ver["published_NDCG@10"] = published_row["NDCG@10"]
        ver["observed_NDCG@10"] = observed["NDCG@10"]
        ver["published_HR@10"] = published_row["HR@10"]
        ver["observed_HR@10"] = observed["HR@10"]
        ver["delta_NDCG@10"] = d_ndcg10
        ver["delta_HR@10"] = d_hr10
        matched = d_ndcg10 <= args.match_tol and d_hr10 <= args.match_tol
        ver["matches_published"] = matched
        if not matched:
            ver["status"] = "PUBLISHED_NUMBER_MISMATCH"
            report["canonical_dir_verification"].append(ver)
            report["errors"].append(
                f"{method}: ranking_metrics NDCG@10/HR@10 do not match published "
                f"(dNDCG10={d_ndcg10:.2e}, dHR10={d_hr10:.2e})"
            )
            continue
        order, ranks = _read_eval_records_csv(eval_path)
        if len(ranks) != expected_users:
            ver["status"] = "USER_COUNT_MISMATCH"
            ver["n_users"] = len(ranks)
            report["canonical_dir_verification"].append(ver)
            report["errors"].append(f"{method}: {len(ranks)} eval records != expected {expected_users}")
            continue
        ver["status"] = "OK"
        ver["n_users"] = len(ranks)
        report["canonical_dir_verification"].append(ver)
        baseline_ranks[method] = ranks
        if common_order is None:
            common_order = order

    # ---- determine the common user set across C-CRP + all OK baselines ----
    if common_order is None or not baseline_ranks:
        report["errors"].append("No OK baselines; cannot run paired tests.")
        _dump(output_dir, stamp, report)
        return 2
    common_users = set(ccrp_ranks)
    for ranks in baseline_ranks.values():
        common_users &= set(ranks)
    order = [u for u in common_order if u in common_users]
    report["n_paired_events"] = len(order)
    if len(order) != expected_users:
        report["errors"].append(
            f"common paired user set {len(order)} != expected {expected_users} "
            f"(C-CRP {len(ccrp_ranks)}, intersection used)"
        )

    ccrp_matrix = _ranks_to_matrix(ccrp_ranks, order)

    # ---- self-check: per-event means MUST reproduce the published table ----
    ccrp_pub = published.get(CCRP_PUBLISHED_NAME, {})
    ccrp_event_means = {m: float(np.mean(ccrp_matrix[:, i])) for i, m in enumerate(REQUIRED_METRICS)}
    ccrp_mean_check: dict[str, Any] = {"method": "C-CRP_v3_ours"}
    ccrp_ok = True
    for m in REQUIRED_METRICS:
        pub = ccrp_pub.get(m)
        obs = ccrp_event_means[m]
        d = abs(obs - pub) if pub is not None else float("nan")
        ccrp_mean_check[m] = {"published": pub, "event_mean": obs, "delta": d}
        if pub is None or d > 1e-6:
            ccrp_ok = False
    ccrp_mean_check["matches_published"] = ccrp_ok
    report["ccrp_event_mean_verification"] = ccrp_mean_check
    if not ccrp_ok:
        report["errors"].append(
            "C-CRP per-event means do NOT reproduce published main_comparison_table "
            "(possible indexing/source mismatch)."
        )

    # ---- paired tests (identical math to the new-4 canonical script) ----
    paired_rows: list[dict[str, Any]] = []
    for baseline_idx, method in enumerate(OFFICIAL_METHODS):
        if method not in baseline_ranks:
            continue
        base_matrix = _ranks_to_matrix(baseline_ranks[method], order)
        # baseline per-event mean over the common user order must match published.
        for i, m in enumerate(REQUIRED_METRICS):
            pub = published[method][m]
            obs = float(np.mean(base_matrix[:, i]))
            if pub is not None and abs(obs - pub) > 1e-6:
                report["errors"].append(
                    f"{method}: per-event {m} mean {obs:.6f} != published {pub:.6f} over common users"
                )
        diff_matrix = ccrp_matrix - base_matrix
        ci_low, ci_high = _bootstrap_ci(
            diff_matrix,
            n_bootstrap=args.n_bootstrap,
            chunk_size=args.bootstrap_chunk,
            random_state=args.random_state + baseline_idx,
        )
        for metric_idx, metric in enumerate(REQUIRED_METRICS):
            diff = diff_matrix[:, metric_idx]
            p_value, test_name = _wilcoxon_or_fallback(diff)
            std = float(np.std(diff, ddof=1)) if diff.size > 1 else float("nan")
            paired_rows.append(
                {
                    "baseline": method,
                    "method": "C-CRP_v3_ours",
                    "metric": metric,
                    "n_paired_events": int(diff.size),
                    "baseline_mean": float(np.mean(base_matrix[:, metric_idx])),
                    "method_mean": float(np.mean(ccrp_matrix[:, metric_idx])),
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
    _holm_bonferroni(paired_rows, args.alpha)

    # ---- SIGNED labelling ----
    for row in paired_rows:
        sig = bool(row["significant_holm"])
        if sig and row["delta"] > 0.0 and row["ci_low"] > 0.0:
            row["signed_label"] = "ccrp_positive_significant"
        elif sig and row["delta"] < 0.0 and row["ci_high"] < 0.0:
            row["signed_label"] = "ccrp_negative_significant"
        elif sig:
            # Holm-significant but CI straddles 0 or delta sign disagrees with CI.
            row["signed_label"] = "significant_ambiguous_sign"
        else:
            row["signed_label"] = "not_significant"

    pos = sum(1 for r in paired_rows if r["signed_label"] == "ccrp_positive_significant")
    neg = sum(1 for r in paired_rows if r["signed_label"] == "ccrp_negative_significant")
    amb = sum(1 for r in paired_rows if r["signed_label"] == "significant_ambiguous_sign")
    ns = sum(1 for r in paired_rows if r["signed_label"] == "not_significant")

    report["paired_test_count"] = len(paired_rows)
    report["signed_counts"] = {
        "positive_significant": pos,
        "negative_significant": neg,
        "significant_ambiguous_sign": amb,
        "not_significant": ns,
    }
    report["max_holm_p_value"] = max((float(r["holm_p_value"]) for r in paired_rows), default=float("nan"))
    report["min_delta"] = min((float(r["delta"]) for r in paired_rows), default=float("nan"))
    report["max_delta"] = max((float(r["delta"]) for r in paired_rows), default=float("nan"))
    report["paired_rows"] = paired_rows

    # C-CRP published rank by NDCG@10 within this domain (from published table).
    ranking = sorted(
        ((m, v["NDCG@10"]) for m, v in published.items()),
        key=lambda kv: -kv[1],
    )
    ccrp_ndcg10 = published.get(CCRP_PUBLISHED_NAME, {}).get("NDCG@10")
    if ccrp_ndcg10 is not None:
        better = sum(1 for _, v in ranking if v > ccrp_ndcg10)
        report["ccrp_published_rank_by_NDCG@10"] = better + 1

    report["all_56_present"] = len(paired_rows) == 56 and not report["errors"]
    _dump(output_dir, stamp, report)
    if not args.quiet:
        small = {k: v for k, v in report.items() if k != "paired_rows"}
        print(json.dumps(small, indent=2, ensure_ascii=False))
    return 0 if report["all_56_present"] else 1


def _dump(output_dir: Path, stamp: str, report: dict[str, Any]) -> None:
    (output_dir / f"{stamp}_paired_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # paired tests CSV (one row per test)
    rows = report.get("paired_rows", [])
    if rows:
        fields = [
            "baseline", "method", "metric", "n_paired_events",
            "baseline_mean", "method_mean", "delta", "ci_low", "ci_high",
            "p_value", "holm_p_value", "significant_holm", "signed_label",
            "test", "effect_cohen_dz", "win_rate", "loss_rate", "tie_rate",
        ]
        with (output_dir / f"{stamp}_paired_tests.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
