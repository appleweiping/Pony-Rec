from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.baselines.internal_scores import (
    audit_score_rows_against_candidates,
    finite_float,
    read_csv_rows,
    text,
    write_json,
    write_score_rows,
)
from src.eval.ranking_task_metrics import build_ranking_eval_frame, compute_ranking_task_metrics
from src.shadow.ccrp import apply_ccrp_scores, parse_weights
from src.utils.io import load_jsonl, save_jsonl


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _load_signal(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    return pd.read_json(source, lines=True)


def _score_lookup(scored_df: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    lookup: dict[tuple[str, str, str], float] = {}
    for row in scored_df.to_dict(orient="records"):
        key = (text(row.get("source_event_id")), text(row.get("user_id")), text(row.get("candidate_item_id")))
        if all(key):
            lookup[key] = finite_float(row.get("ccrp_risk_adjusted_score"))
    return lookup


def _candidate_score_rows(
    *,
    candidate_rows: list[dict[str, Any]],
    scored_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    lookup = _score_lookup(scored_df)
    rows: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        key = (text(candidate.get("source_event_id")), text(candidate.get("user_id")), text(candidate.get("item_id")))
        rows.append(
            {
                "source_event_id": key[0],
                "user_id": key[1],
                "item_id": key[2],
                "score": lookup.get(key, float("nan")),
            }
        )
    return rows


def _predictions_from_scores(
    ranking_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    *,
    method_name: str,
    k: int,
) -> list[dict[str, Any]]:
    score_lookup = {
        (text(row.get("source_event_id")), text(row.get("user_id")), text(row.get("item_id"))): finite_float(row.get("score"))
        for row in score_rows
    }
    predictions: list[dict[str, Any]] = []
    for record in ranking_rows:
        source_event_id = text(record.get("source_event_id"))
        user_id = text(record.get("user_id"))
        candidate_ids = [text(item) for item in record.get("candidate_item_ids", []) if text(item)]
        scored = [
            (
                item_id,
                score_lookup.get((source_event_id, user_id, item_id), float("-inf")),
                idx,
            )
            for idx, item_id in enumerate(candidate_ids)
        ]
        ranked = [item_id for item_id, _, _ in sorted(scored, key=lambda item: (-item[1], item[2]))]
        prediction = dict(record)
        prediction.update(
            {
                "pred_ranked_item_ids": ranked,
                "topk_item_ids": ranked[:k],
                "parse_success": True,
                "latency": 0.0,
                "confidence": -1.0,
                "contains_out_of_candidate_item": False,
                "raw_response": method_name,
                "candidate_scores": {item_id: score for item_id, score, _ in scored},
            }
        )
        predictions.append(prediction)
    return predictions


def _evaluate_candidate_scores(
    *,
    ranking_path: str | Path,
    candidate_items_path: str | Path,
    signal_path: str | Path,
    score_mode: str,
    ablation: str,
    eta: float,
    confidence_weight: float,
    k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    ranking_rows = load_jsonl(ranking_path)
    candidate_rows = read_csv_rows(candidate_items_path)
    signal_df = _load_signal(signal_path)
    scored_df = apply_ccrp_scores(
        signal_df,
        weights=parse_weights(None),
        eta=eta,
        ablation=ablation,
        score_mode=score_mode,
        confidence_weight=confidence_weight,
    )
    score_rows = _candidate_score_rows(candidate_rows=candidate_rows, scored_df=scored_df)
    audit = audit_score_rows_against_candidates(candidate_rows=candidate_rows, score_rows=score_rows)
    if not audit["audit_ok"]:
        raise ValueError(f"C-CRP score coverage audit failed for {signal_path}: {audit}")
    predictions = _predictions_from_scores(
        ranking_rows,
        score_rows,
        method_name=f"ccrp:{score_mode}:{ablation}:eta={eta}:cw={confidence_weight}",
        k=k,
    )
    metrics = compute_ranking_task_metrics(build_ranking_eval_frame(pd.DataFrame(predictions)), k=k)
    return {**audit, **metrics}, score_rows, scored_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select C-CRP mode on validation, export exact test scores, and optionally import them.")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--valid_ranking_path", required=True)
    parser.add_argument("--test_ranking_path", required=True)
    parser.add_argument("--valid_candidate_items_path", required=True)
    parser.add_argument("--test_candidate_items_path", required=True)
    parser.add_argument("--valid_signal_path", required=True)
    parser.add_argument("--test_signal_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--score_modes", default="confidence_only,evidence_only,confidence_plus_evidence,full")
    parser.add_argument("--ablations", default="full,without_calibration_gap,without_evidence_support,without_counterevidence,without_risk_penalty")
    parser.add_argument("--etas", default="0.5,1.0,2.0")
    parser.add_argument("--confidence_weights", default="0.5,0.7,0.9")
    parser.add_argument("--selection_metric", default="NDCG@10")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--baseline_name", default=None)
    parser.add_argument("--import_exp_name", default=None)
    parser.add_argument("--import_scores", action="store_true")
    parser.add_argument("--status_label", default="same_schema_internal_method")
    parser.add_argument("--artifact_class", default="completed_result")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_modes = _parse_csv_list(args.score_modes)
    ablations = _parse_csv_list(args.ablations)
    etas = _parse_float_list(args.etas)
    confidence_weights = _parse_float_list(args.confidence_weights)

    valid_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for mode in score_modes:
        for ablation in ablations:
            for eta in etas:
                for confidence_weight in confidence_weights:
                    if mode != "confidence_plus_evidence" and confidence_weight != confidence_weights[0]:
                        continue
                    metrics, _, _ = _evaluate_candidate_scores(
                        ranking_path=args.valid_ranking_path,
                        candidate_items_path=args.valid_candidate_items_path,
                        signal_path=args.valid_signal_path,
                        score_mode=mode,
                        ablation=ablation,
                        eta=eta,
                        confidence_weight=confidence_weight,
                        k=args.k,
                    )
                    row = {
                        "domain": args.domain,
                        "split": "valid",
                        "score_mode": mode,
                        "ablation": ablation,
                        "eta": eta,
                        "confidence_weight": confidence_weight,
                        **metrics,
                    }
                    valid_rows.append(row)
                    if best is None or float(row[args.selection_metric]) > float(best[args.selection_metric]):
                        best = row

    if best is None:
        raise RuntimeError("No C-CRP validation rows were produced.")
    pd.DataFrame(valid_rows).to_csv(output_dir / "valid_ccrp_sweep.csv", index=False)
    write_json(best, output_dir / "selected_valid_config.json")

    test_metrics, test_score_rows, test_scored_df = _evaluate_candidate_scores(
        ranking_path=args.test_ranking_path,
        candidate_items_path=args.test_candidate_items_path,
        signal_path=args.test_signal_path,
        score_mode=str(best["score_mode"]),
        ablation=str(best["ablation"]),
        eta=float(best["eta"]),
        confidence_weight=float(best["confidence_weight"]),
        k=args.k,
    )
    score_summary = write_score_rows(test_score_rows, output_dir / "ccrp_selected_test_scores.csv")
    test_scored_df.to_csv(output_dir / "ccrp_selected_test_scored_rows.csv", index=False)
    test_row = {
        "domain": args.domain,
        "split": "test",
        "selected_on": "valid",
        "selection_metric": args.selection_metric,
        f"selected_valid_{args.selection_metric}": best[args.selection_metric],
        "score_mode": best["score_mode"],
        "ablation": best["ablation"],
        "eta": best["eta"],
        "confidence_weight": best["confidence_weight"],
        "status_label": args.status_label,
        "artifact_class": args.artifact_class,
        **score_summary,
        **test_metrics,
    }
    pd.DataFrame([test_row]).to_csv(output_dir / "selected_test_metrics.csv", index=False)
    write_json(test_row, output_dir / "ccrp_internal_provenance.json")

    if args.import_scores:
        baseline_name = args.baseline_name or f"{args.domain}_ccrp_{best['score_mode']}"
        exp_name = args.import_exp_name or f"{args.domain}_ccrp_selected_same_candidate"
        cmd = [
            sys.executable,
            "main_import_same_candidate_baseline_scores.py",
            "--baseline_name",
            baseline_name,
            "--exp_name",
            exp_name,
            "--domain",
            args.domain,
            "--ranking_input_path",
            args.test_ranking_path,
            "--scores_path",
            str(output_dir / "ccrp_selected_test_scores.csv"),
            "--artifact_class",
            args.artifact_class,
            "--status_label",
            args.status_label,
        ]
        subprocess.run(cmd, check=True)

    print(f"Saved C-CRP valid sweep: {output_dir / 'valid_ccrp_sweep.csv'}")
    print(f"Saved selected C-CRP scores: {output_dir / 'ccrp_selected_test_scores.csv'}")
    print(f"Saved selected test metrics: {output_dir / 'selected_test_metrics.csv'}")


if __name__ == "__main__":
    main()
