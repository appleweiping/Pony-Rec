from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
repo_root_text = str(REPO_ROOT)
if repo_root_text not in sys.path:
    sys.path.insert(0, repo_root_text)

from scripts.misc.main_select_ccrp_variant_on_valid import (  # noqa: E402
    FULL_REPORTING_KS,
    _candidate_score_rows,
    _parse_float_list,
    _parse_weight_grid,
    _predictions_from_scores,
    _weight_label,
)
from src.baselines.internal_scores import (  # noqa: E402
    audit_score_degeneracy,
    audit_score_rows_against_candidates,
    finite_float,
    read_csv_rows,
    sha256_file,
    text,
)
from src.eval.ranking_task_metrics import build_ranking_eval_frame, compute_ranking_task_metrics  # noqa: E402
from src.shadow.ccrp import apply_ccrp_scores, parse_weights  # noqa: E402
from src.utils.io import load_jsonl  # noqa: E402


DEFAULT_ETA_GRID = "0,0.25,0.5,1,2,4"
DEFAULT_WEIGHT_GRID = "0.5,0.3,0.2;0.7,0.2,0.1;0.4,0.4,0.2;0.4,0.2,0.4;0.33,0.33,0.34"
DEFAULT_DIAGNOSTIC_CONFIDENCE_GRID = "0.1,0.3,0.5,0.7,0.9"
MAIN_CONTROLS = ("eta", "weight_grid_label")
DIAGNOSTIC_CONTROLS = ("confidence_weight",)


@dataclass
class SplitResources:
    split: str
    ranking_path: Path
    candidate_items_path: Path
    signal_path: Path
    ranking_rows: list[dict[str, Any]]
    candidate_rows: list[dict[str, Any]]
    signal_df: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build valid/test C-CRP hyperparameter sweep CSVs from saved signal rows. "
            "This does not query an LLM, import scores, or retain per-grid score dumps."
        )
    )
    parser.add_argument("--domain", required=True)
    parser.add_argument("--valid_ranking_path", required=True)
    parser.add_argument("--test_ranking_path", required=True)
    parser.add_argument("--valid_candidate_items_path", required=True)
    parser.add_argument("--test_candidate_items_path", required=True)
    parser.add_argument("--valid_signal_path", required=True)
    parser.add_argument("--test_signal_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metric", default="NDCG@10")
    parser.add_argument("--eta_grid", default=DEFAULT_ETA_GRID)
    parser.add_argument("--weight_grid", default=DEFAULT_WEIGHT_GRID)
    parser.add_argument("--diagnostic_confidence_weights", default=DEFAULT_DIAGNOSTIC_CONFIDENCE_GRID)
    parser.add_argument("--skip_diagnostic_confidence", action="store_true")
    parser.add_argument("--expected_events", type=int, default=10000)
    parser.add_argument("--expected_candidates_per_event", type=int, default=101)
    parser.add_argument("--tie_break_seed", type=int, default=20260607)
    parser.add_argument("--max_tie_pair_rate", type=float, default=0.7)
    parser.add_argument("--max_constant_event_rate", type=float, default=0.02)
    return parser.parse_args()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _load_signal(path: str | Path) -> pd.DataFrame:
    signal_path = Path(path)
    if signal_path.suffix.lower() == ".csv":
        return pd.read_csv(signal_path)
    return pd.read_json(signal_path, lines=True)


def _read_resources(
    *,
    split: str,
    ranking_path: str | Path,
    candidate_items_path: str | Path,
    signal_path: str | Path,
    expected_events: int,
    expected_candidates_per_event: int,
) -> SplitResources:
    ranking = Path(ranking_path)
    candidates = Path(candidate_items_path)
    signal = Path(signal_path)
    ranking_rows = load_jsonl(ranking)
    candidate_rows = read_csv_rows(candidates)
    signal_df = _load_signal(signal)
    expected_keys = expected_events * expected_candidates_per_event if expected_events > 0 else 0
    if expected_events > 0 and len(ranking_rows) != expected_events:
        raise ValueError(f"{split}: ranking rows {len(ranking_rows)} != expected_events {expected_events}")
    if expected_keys > 0 and len(candidate_rows) != expected_keys:
        raise ValueError(f"{split}: candidate rows {len(candidate_rows)} != expected keys {expected_keys}")
    if expected_keys > 0 and len(signal_df) != expected_keys:
        raise ValueError(f"{split}: signal rows {len(signal_df)} != expected keys {expected_keys}")
    return SplitResources(
        split=split,
        ranking_path=ranking,
        candidate_items_path=candidates,
        signal_path=signal,
        ranking_rows=ranking_rows,
        candidate_rows=candidate_rows,
        signal_df=signal_df,
    )


def _evaluate_config(
    resources: SplitResources,
    *,
    domain: str,
    control: str,
    control_value: str,
    row_kind: str,
    score_mode: str,
    ablation: str,
    eta: float,
    confidence_weight: float,
    weights: tuple[float, float, float],
    expected_events: int,
    expected_candidates_per_event: int,
    tie_break_seed: int,
    max_tie_pair_rate: float,
    max_constant_event_rate: float,
) -> dict[str, Any]:
    scored_df = apply_ccrp_scores(
        resources.signal_df,
        weights=parse_weights(weights),
        eta=eta,
        ablation=ablation,
        score_mode=score_mode,
        confidence_weight=confidence_weight,
    )
    score_rows = _candidate_score_rows(candidate_rows=resources.candidate_rows, scored_df=scored_df)
    coverage_audit = audit_score_rows_against_candidates(candidate_rows=resources.candidate_rows, score_rows=score_rows)
    if not coverage_audit["audit_ok"]:
        raise ValueError(f"{resources.split}:{control}:{control_value} coverage audit failed: {coverage_audit}")
    degeneracy_audit = audit_score_degeneracy(
        score_rows,
        max_tie_pair_rate=max_tie_pair_rate,
        max_constant_event_rate=max_constant_event_rate,
    )
    predictions = _predictions_from_scores(
        resources.ranking_rows,
        score_rows,
        method_name=(
            f"ccrp_hyper:{resources.split}:{control}={control_value}:"
            f"{score_mode}:{ablation}:eta={eta}:cw={confidence_weight}:w={_weight_label(weights)}"
        ),
        k=max(FULL_REPORTING_KS),
        tie_break_seed=tie_break_seed,
    )
    metrics = compute_ranking_task_metrics(
        build_ranking_eval_frame(pd.DataFrame(predictions)),
        k=max(FULL_REPORTING_KS),
        ks=FULL_REPORTING_KS,
    )
    del scored_df, score_rows, predictions
    gc.collect()
    expected_keys = expected_events * expected_candidates_per_event if expected_events > 0 else 0
    return {
        "domain": domain,
        "split": resources.split,
        "row_kind": row_kind,
        "control": control,
        "control_value": str(control_value),
        "score_mode": score_mode,
        "ablation": ablation,
        "eta": float(eta),
        "confidence_weight": float(confidence_weight),
        "weight_boundary": float(weights[0]),
        "weight_calibration_gap": float(weights[1]),
        "weight_evidence": float(weights[2]),
        "weight_grid_label": _weight_label(weights),
        "expected_events": int(expected_events),
        "expected_candidates_per_event": int(expected_candidates_per_event),
        "expected_candidate_key_count": int(expected_keys),
        "tie_break_seed": int(tie_break_seed),
        **coverage_audit,
        **degeneracy_audit,
        **metrics,
    }


def _validate_rows(
    rows: list[dict[str, Any]],
    *,
    expected_keys: int,
    expected_main_controls: tuple[str, ...] = MAIN_CONTROLS,
) -> None:
    if not rows:
        raise ValueError("empty hyperparameter sweep rows")
    for idx, row in enumerate(rows):
        for key in ("MRR", "HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20"):
            value = finite_float(row.get(key), default=float("nan"))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"row {idx} metric {key} is nonfinite/out of range: {row.get(key)}")
        if int(row.get("candidate_key_count") or 0) != expected_keys:
            raise ValueError(f"row {idx} candidate_key_count mismatch: {row.get('candidate_key_count')} != {expected_keys}")
        if int(row.get("score_key_count") or 0) != expected_keys:
            raise ValueError(f"row {idx} score_key_count mismatch: {row.get('score_key_count')} != {expected_keys}")
        if abs(float(row.get("score_coverage_rate") or 0.0) - 1.0) > 1e-12:
            raise ValueError(f"row {idx} score_coverage_rate != 1.0: {row.get('score_coverage_rate')}")
        for key in ("missing_score_keys", "extra_score_keys", "duplicate_score_keys", "invalid_scores", "blank_score_keys"):
            if int(row.get(key) or 0) != 0:
                raise ValueError(f"row {idx} {key} not zero: {row.get(key)}")
        if row.get("audit_ok") is not True:
            raise ValueError(f"row {idx} audit_ok is not true")
        if row.get("degeneracy_audit_ok") is not True:
            raise ValueError(f"row {idx} degeneracy_audit_ok is not true")
        if str(row.get("control")) == "confidence_weight" and str(row.get("score_mode")) == "full":
            raise ValueError("confidence_weight cannot be emitted as a full-mode main-control row")
    splits = {str(row.get("split")) for row in rows}
    for split in splits:
        for control in expected_main_controls:
            if not any(str(row.get("split")) == split and str(row.get("control")) == control for row in rows):
                raise ValueError(f"missing main control {control} for split {split}")


def build_sweep(
    *,
    domain: str,
    valid_ranking_path: str | Path,
    test_ranking_path: str | Path,
    valid_candidate_items_path: str | Path,
    test_candidate_items_path: str | Path,
    valid_signal_path: str | Path,
    test_signal_path: str | Path,
    output_dir: str | Path,
    metric: str = "NDCG@10",
    eta_grid: str = DEFAULT_ETA_GRID,
    weight_grid: str = DEFAULT_WEIGHT_GRID,
    diagnostic_confidence_weights: str = DEFAULT_DIAGNOSTIC_CONFIDENCE_GRID,
    include_diagnostic_confidence: bool = True,
    expected_events: int = 10000,
    expected_candidates_per_event: int = 101,
    tie_break_seed: int = 20260607,
    max_tie_pair_rate: float = 0.7,
    max_constant_event_rate: float = 0.02,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    etas = _parse_float_list(eta_grid)
    weights = _parse_weight_grid(weight_grid)
    diagnostic_cws = _parse_float_list(diagnostic_confidence_weights) if include_diagnostic_confidence else []
    current_weights = (0.5, 0.3, 0.2)
    valid = _read_resources(
        split="valid",
        ranking_path=valid_ranking_path,
        candidate_items_path=valid_candidate_items_path,
        signal_path=valid_signal_path,
        expected_events=expected_events,
        expected_candidates_per_event=expected_candidates_per_event,
    )
    test = _read_resources(
        split="test",
        ranking_path=test_ranking_path,
        candidate_items_path=test_candidate_items_path,
        signal_path=test_signal_path,
        expected_events=expected_events,
        expected_candidates_per_event=expected_candidates_per_event,
    )
    rows_by_split: dict[str, list[dict[str, Any]]] = {"valid": [], "test": []}
    for resources in (valid, test):
        for eta in etas:
            rows_by_split[resources.split].append(
                _evaluate_config(
                    resources,
                    domain=domain,
                    control="eta",
                    control_value=f"{eta:g}",
                    row_kind="main_control",
                    score_mode="full",
                    ablation="full",
                    eta=eta,
                    confidence_weight=0.7,
                    weights=current_weights,
                    expected_events=expected_events,
                    expected_candidates_per_event=expected_candidates_per_event,
                    tie_break_seed=tie_break_seed,
                    max_tie_pair_rate=max_tie_pair_rate,
                    max_constant_event_rate=max_constant_event_rate,
                )
            )
        for weight in weights:
            rows_by_split[resources.split].append(
                _evaluate_config(
                    resources,
                    domain=domain,
                    control="weight_grid_label",
                    control_value=_weight_label(weight),
                    row_kind="main_control",
                    score_mode="full",
                    ablation="full",
                    eta=1.0,
                    confidence_weight=0.7,
                    weights=weight,
                    expected_events=expected_events,
                    expected_candidates_per_event=expected_candidates_per_event,
                    tie_break_seed=tie_break_seed,
                    max_tie_pair_rate=max_tie_pair_rate,
                    max_constant_event_rate=max_constant_event_rate,
                )
            )
        for confidence_weight in diagnostic_cws:
            rows_by_split[resources.split].append(
                _evaluate_config(
                    resources,
                    domain=domain,
                    control="confidence_weight",
                    control_value=f"{confidence_weight:g}",
                    row_kind="diagnostic_control",
                    score_mode="confidence_plus_evidence",
                    ablation="full",
                    eta=1.0,
                    confidence_weight=confidence_weight,
                    weights=current_weights,
                    expected_events=expected_events,
                    expected_candidates_per_event=expected_candidates_per_event,
                    tie_break_seed=tie_break_seed,
                    max_tie_pair_rate=max_tie_pair_rate,
                    max_constant_event_rate=max_constant_event_rate,
                )
            )
    expected_keys = expected_events * expected_candidates_per_event if expected_events > 0 else 0
    all_rows = rows_by_split["valid"] + rows_by_split["test"]
    _validate_rows(all_rows, expected_keys=expected_keys)
    valid_path = out / "valid_ccrp_hyperparameter_sweep.csv"
    test_path = out / "test_ccrp_hyperparameter_sweep.csv"
    pd.DataFrame(rows_by_split["valid"]).to_csv(valid_path, index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(rows_by_split["test"]).to_csv(test_path, index=False, quoting=csv.QUOTE_MINIMAL)
    provenance = {
        "artifact_class": "paper_critical_hyperparameter_sweep_inputs",
        "status_label": "valid_test_saved_signal_hyperparameter_sweep_ready",
        "ok": True,
        "domain": domain,
        "git_commit": _git_commit(),
        "command": " ".join(sys.argv),
        "metric": metric,
        "test_not_used_for_selection": True,
        "main_controls": list(MAIN_CONTROLS),
        "diagnostic_controls": list(DIAGNOSTIC_CONTROLS) if include_diagnostic_confidence else [],
        "eta_grid": [float(value) for value in etas],
        "weight_grid": [_weight_label(value) for value in weights],
        "diagnostic_confidence_weights": [float(value) for value in diagnostic_cws],
        "expected_events": int(expected_events),
        "expected_candidates_per_event": int(expected_candidates_per_event),
        "expected_candidate_key_count": int(expected_keys),
        "tie_break_seed": int(tie_break_seed),
        "max_tie_pair_rate": float(max_tie_pair_rate),
        "max_constant_event_rate": float(max_constant_event_rate),
        "paths": {
            "valid_ranking_path": str(valid_ranking_path),
            "test_ranking_path": str(test_ranking_path),
            "valid_candidate_items_path": str(valid_candidate_items_path),
            "test_candidate_items_path": str(test_candidate_items_path),
            "valid_signal_path": str(valid_signal_path),
            "test_signal_path": str(test_signal_path),
            "valid_sweep_csv": str(valid_path),
            "test_sweep_csv": str(test_path),
        },
        "sha256": {
            "valid_ranking": sha256_file(valid_ranking_path),
            "test_ranking": sha256_file(test_ranking_path),
            "valid_candidate_items": sha256_file(valid_candidate_items_path),
            "test_candidate_items": sha256_file(test_candidate_items_path),
            "valid_signal": sha256_file(valid_signal_path),
            "test_signal": sha256_file(test_signal_path),
            "valid_sweep_csv": sha256_file(valid_path),
            "test_sweep_csv": sha256_file(test_path),
        },
        "row_counts": {
            "valid_rows": len(rows_by_split["valid"]),
            "test_rows": len(rows_by_split["test"]),
            "valid_signal_rows": len(valid.signal_df),
            "test_signal_rows": len(test.signal_df),
            "valid_candidate_rows": len(valid.candidate_rows),
            "test_candidate_rows": len(test.candidate_rows),
            "valid_ranking_events": len(valid.ranking_rows),
            "test_ranking_events": len(test.ranking_rows),
        },
        "cleanup_status": {
            "retained_bulk_scores_csv": False,
            "retained_prediction_jsonl": False,
            "retained_scored_temp_rows": False,
            "retained_checkpoints": False,
            "policy": "metrics_only_no_per_grid_score_or_prediction_files",
        },
        "claim_limits": [
            "Hyperparameter curves are sensitivity/stability evidence only.",
            "Test sweep is reporting-only and cannot select or change the main method.",
            "confidence_weight is diagnostic under confidence_plus_evidence and not a main full-mode hyperparameter.",
        ],
    }
    (out / "ccrp_hyperparameter_sweep_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return provenance


def main() -> None:
    args = parse_args()
    provenance = build_sweep(
        domain=args.domain,
        valid_ranking_path=args.valid_ranking_path,
        test_ranking_path=args.test_ranking_path,
        valid_candidate_items_path=args.valid_candidate_items_path,
        test_candidate_items_path=args.test_candidate_items_path,
        valid_signal_path=args.valid_signal_path,
        test_signal_path=args.test_signal_path,
        output_dir=args.output_dir,
        metric=args.metric,
        eta_grid=args.eta_grid,
        weight_grid=args.weight_grid,
        diagnostic_confidence_weights=args.diagnostic_confidence_weights,
        include_diagnostic_confidence=not args.skip_diagnostic_confidence,
        expected_events=args.expected_events,
        expected_candidates_per_event=args.expected_candidates_per_event,
        tie_break_seed=args.tie_break_seed,
        max_tie_pair_rate=args.max_tie_pair_rate,
        max_constant_event_rate=args.max_constant_event_rate,
    )
    print(
        json.dumps(
            {
                "ok": provenance["ok"],
                "output_dir": args.output_dir,
                "valid_rows": provenance["row_counts"]["valid_rows"],
                "test_rows": provenance["row_counts"]["test_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
