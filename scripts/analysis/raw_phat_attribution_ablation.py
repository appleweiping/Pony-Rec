"""RAW-p̂ attribution ablation (Major-Revision blocker, Task 1).

Reviewer ask: provide a baseline that ranks the 101 sampled-negative candidates by
the RAW LLM relevance probability ALONE -- no calibration map, no uncertainty U,
no risk penalty (equivalently eta=0) -- to attribute C-CRP's reported win between
"LLM pointwise judging" and "the C-CRP machinery (calibration + uncertainty
decomposition + risk-adjusted ranking)".

This script is CPU-only and reuses the project's own ranking + metric code:
  * src.shadow.ccrp._tie_break logic via main_select_ccrp_variant_on_valid._predictions_from_scores
    (same seeded SHA256 stable tie-break, seed 20260607)
  * src.eval.ranking_task_metrics.build_ranking_eval_frame / compute_ranking_task_metrics

Per-candidate raw LLM probability is read from the locally-persisted Phase-2.5
signal rows:
  outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/
      ccrp_signal_rows_<domain>/test/test_ccrp_signal_rows.csv
column `relevance_probability` (pre-calibration LLM verbalized posterior).

The canonical event / candidate-set / positive-item frame is taken from any one
official baseline's ranking_eval_records.csv for the domain (candidate sets are
identical across all methods under the same-candidate protocol):
  outputs/baselines/official_adapters/<domain>_large10000_100neg_<baseline>_official_qwen3base_same_candidate/tables/ranking_eval_records.csv

Output: a results JSON written under outputs/summary/paper_critical/ comparing
raw-p̂ vs the reported main-table C-CRP per domain. Expectation (given the formal
C-CRP v3 runner already ranks by raw relevance_probability and the reported
main-table C-CRP row uses eta=0-equivalent pure-prob sorting): raw-p̂ ~= C-CRP.

NOTE: this is a temporary analysis artifact for the rebuttal. It does NOT touch
any GPU and re-uses already-persisted LLM outputs (no re-inference).
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.internal_scores import finite_float, text  # noqa: E402
from src.eval.ranking_task_metrics import (  # noqa: E402
    build_ranking_eval_frame,
    compute_ranking_task_metrics,
)
from src.shadow.scoring import _clamp01  # noqa: E402  (only for clamp parity)


def _load_predictions_fn():
    """Reuse the project's exact seeded tie-break ranker without making scripts/ a package."""
    import importlib.util

    sel_path = REPO_ROOT / "scripts" / "misc" / "main_select_ccrp_variant_on_valid.py"
    spec = importlib.util.spec_from_file_location("_ccrp_selector", sel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._predictions_from_scores


_predictions_from_scores = _load_predictions_fn()

FULL_REPORTING_KS = (5, 10, 20)
TIE_BREAK_SEED = 20260607  # identical to the project's selector default

SIGNAL_ROOT = (
    REPO_ROOT
    / "outputs"
    / "summary"
    / "paper_critical"
    / "ccrp_signal_generation_plan_post_performance_gate_20260606"
)

# Reported main-table C-CRP v3 numbers (ccrp_v3_qwen3base_pointwise) from
# outputs/summary/new_domains_official_ccrp_cross_domain_20260605_method_rows.csv
# (these are the headline numbers the rebuttal must attribute).
REPORTED_CCRP_MAIN = {
    "sports": {"HR@5": 0.2745, "HR@10": 0.3819, "HR@20": 0.5172,
               "NDCG@5": 0.19845140673482808, "NDCG@10": 0.23286197589860086,
               "NDCG@20": 0.2670062993241999, "MRR": 0.20756582350031022},
    "toys": {"HR@5": 0.3172, "HR@10": 0.3964, "HR@20": 0.5059,
             "NDCG@5": 0.2451904009717959, "NDCG@10": 0.27079859856897753,
             "NDCG@20": 0.298341205798594, "MRR": 0.2503049488607351},
    "home": {"HR@5": 0.1561, "HR@10": 0.2264, "HR@20": 0.3505,
             "NDCG@5": 0.10978017620744406, "NDCG@10": 0.13239420796539653,
             "NDCG@20": 0.16351113583680343, "MRR": 0.1259477884601691},
    "tools": {"HR@5": 0.1937, "HR@10": 0.2696, "HR@20": 0.3931,
              "NDCG@5": 0.14186375906171483, "NDCG@10": 0.16611553052793934,
              "NDCG@20": 0.19703986741872317, "MRR": 0.15585924577949772},
}


def _find_ranking_eval_records(domain: str) -> Path:
    pattern = str(
        REPO_ROOT
        / "outputs"
        / "baselines"
        / "official_adapters"
        / f"{domain}_large10000_100neg_*_official_qwen3base_same_candidate"
        / "tables"
        / "ranking_eval_records.csv"
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No ranking_eval_records.csv found locally for domain={domain} (pattern: {pattern})."
        )
    return Path(matches[0])


def _signal_rows_path(domain: str) -> Path:
    return SIGNAL_ROOT / f"ccrp_signal_rows_{domain}" / "test" / "test_ccrp_signal_rows.csv"


def _load_ranking_rows(rer_path: Path) -> list[dict[str, Any]]:
    """Reconstruct ranking_task-style rows from a ranking_eval_records.csv."""
    df = pd.read_csv(rer_path)
    rows: list[dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        candidate_ids = ast.literal_eval(rec["candidate_item_ids"]) if isinstance(rec["candidate_item_ids"], str) else rec["candidate_item_ids"]
        pop_groups_raw = rec.get("candidate_popularity_groups")
        pop_groups = (
            ast.literal_eval(pop_groups_raw)
            if isinstance(pop_groups_raw, str) and pop_groups_raw.strip().startswith("[")
            else []
        )
        rows.append(
            {
                "source_event_id": text(rec.get("source_event_id")),
                "user_id": text(rec.get("user_id")),
                "split_name": rec.get("split_name"),
                "timestamp": rec.get("timestamp"),
                "positive_item_id": text(rec.get("positive_item_id")),
                "candidate_item_ids": [text(c) for c in candidate_ids],
                "candidate_popularity_groups": [text(g) for g in pop_groups],
            }
        )
    return rows


def _load_raw_score_rows(
    signal_path: Path,
    *,
    prob_column: str,
    chunksize: int = 200_000,
) -> list[dict[str, Any]]:
    """Stream the (large) signal CSV and emit (event,user,item)->raw prob score rows."""
    score_rows: list[dict[str, Any]] = []
    usecols = ["source_event_id", "user_id", "candidate_item_id", "item_id", prob_column]
    reader = pd.read_csv(signal_path, usecols=lambda c: c in usecols, chunksize=chunksize)
    for chunk in reader:
        for rec in chunk.to_dict(orient="records"):
            item_id = text(rec.get("candidate_item_id")) or text(rec.get("item_id"))
            score_rows.append(
                {
                    "source_event_id": text(rec.get("source_event_id")),
                    "user_id": text(rec.get("user_id")),
                    "item_id": item_id,
                    "score": _clamp01(finite_float(rec.get(prob_column))),
                }
            )
    return score_rows


def rerank_domain(
    domain: str,
    *,
    prob_column: str = "relevance_probability",
    k: int = 10,
) -> dict[str, Any]:
    rer_path = _find_ranking_eval_records(domain)
    signal_path = _signal_rows_path(domain)
    if not signal_path.exists():
        return {
            "domain": domain,
            "status": "skipped_missing_local_test_signal_rows",
            "expected_signal_path": str(signal_path),
            "note": "Test signal rows not present locally for this domain; re-run requires server copy.",
            "reported_ccrp_main_metrics": REPORTED_CCRP_MAIN.get(domain, {}),
            "selector_route_same_tiebreak_reference": _selector_route_reference(domain),
        }

    ranking_rows = _load_ranking_rows(rer_path)
    score_rows = _load_raw_score_rows(signal_path, prob_column=prob_column)

    ranking_k = max(int(k), max(FULL_REPORTING_KS))
    predictions = _predictions_from_scores(
        ranking_rows,
        score_rows,
        method_name=f"raw_phat:{prob_column}",
        k=ranking_k,
        tie_break_seed=TIE_BREAK_SEED,
    )
    metrics = compute_ranking_task_metrics(
        build_ranking_eval_frame(pd.DataFrame(predictions)), k=ranking_k, ks=FULL_REPORTING_KS
    )

    reported = REPORTED_CCRP_MAIN.get(domain, {})
    deltas = {
        m: (float(metrics[m]) - float(reported[m]))
        for m in reported
        if m in metrics
    }
    return {
        "domain": domain,
        "status": "ok",
        "prob_column": prob_column,
        "tie_break_seed": TIE_BREAK_SEED,
        "n_events": len(ranking_rows),
        "n_candidate_scores": len(score_rows),
        "ranking_eval_records_source": str(rer_path.relative_to(REPO_ROOT)),
        "signal_rows_source": str(signal_path.relative_to(REPO_ROOT)),
        "raw_phat_metrics": {
            m: float(metrics[m])
            for m in ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "MRR"]
            if m in metrics
        },
        "reported_ccrp_main_metrics": reported,
        "raw_phat_minus_ccrp_main": deltas,
        "selector_route_same_tiebreak_reference": _selector_route_reference(domain),
    }


def _selector_route_reference(domain: str) -> dict[str, Any]:
    """Pull the same-seeded-tie-break selector-route C-CRP numbers already on disk:
    full (calibration+uncertainty+risk) vs confidence_only (calibrated prob, no U/risk).
    These give the apples-to-apples (identical tie-break) attribution alongside raw-p̂.
    """
    diag = (
        SIGNAL_ROOT
        / f"ccrp_ablation_{domain}"
        / "ccrp_ablation_diagnostics.csv"
    )
    if not diag.exists():
        return {"status": "no_selector_diagnostics", "path": str(diag)}
    df = pd.read_csv(diag)
    out: dict[str, Any] = {"status": "ok", "source": str(diag.relative_to(REPO_ROOT))}
    wanted = {
        "ccrp_full": ("full", "full"),
        "confidence_only_no_uncertainty_no_risk": ("confidence_only", "full"),
        "without_risk_penalty": ("full", "without_risk_penalty"),
        "without_calibration_gap": ("full", "without_calibration_gap"),
    }
    for label, (mode, abl) in wanted.items():
        sub = df[(df["score_mode"] == mode) & (df["ablation"] == abl)]
        if not sub.empty:
            r = sub.iloc[0]
            out[label] = {m: float(r[m]) for m in ["HR@10", "NDCG@10", "MRR"] if m in r}
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAW-p̂ attribution ablation (CPU re-rank).")
    parser.add_argument("--domains", default="sports,toys,home,tools")
    parser.add_argument("--prob_column", default="relevance_probability")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--output_json",
        default=str(
            REPO_ROOT
            / "outputs"
            / "summary"
            / "paper_critical"
            / "raw_phat_attribution_ablation_results.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    results = [rerank_domain(d, prob_column=args.prob_column, k=args.k) for d in domains]
    payload = {
        "experiment": "raw_phat_attribution_ablation",
        "description": (
            "Rank 101 sampled-negative candidates by RAW LLM relevance_probability alone "
            "(no calibration, no uncertainty, eta=0-equivalent), using the project's own "
            "seeded tie-break + ranking_task_metrics. Compares to reported main-table C-CRP v3."
        ),
        "tie_break_seed": TIE_BREAK_SEED,
        "domains": results,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
