from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_repo_root_text = str(_REPO_ROOT)
if _repo_root_text not in sys.path:
    sys.path.insert(0, _repo_root_text)

from src.baselines.internal_scores import (
    audit_score_degeneracy,
    audit_score_rows_against_candidates,
    read_csv_rows,
    sha256_file,
    text,
    write_json,
    write_score_rows,
)
from src.shadow.ccrp import apply_ccrp_scores, parse_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild C-CRP scored rows from saved signal rows and a fixed C-CRP config. "
            "This does not query an LLM or select on test data; use it only after the "
            "signal source has passed the uncertainty-source preflight audit."
        )
    )
    parser.add_argument("--signal_path", required=True, help="CSV or JSONL signal rows with relevance/evidence fields.")
    parser.add_argument("--candidate_items_path", required=True, help="Same-candidate candidate_items.csv.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--domain", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--selected_config_json",
        default="",
        help="Optional selected_valid_config.json or ccrp_internal_provenance.json containing score_mode/ablation/eta/weights.",
    )
    parser.add_argument("--score_mode", default="full")
    parser.add_argument("--ablation", default="full")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--confidence_weight", type=float, default=0.7)
    parser.add_argument("--weights", default="0.5,0.3,0.2", help="boundary,calibration_gap,evidence")
    parser.add_argument("--expected_rows", type=int, default=0, help="Optional exact expected candidate row count.")
    parser.add_argument("--allow_degenerate_scores", action="store_true")
    return parser.parse_args()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _load_signal(path: str | Path) -> pd.DataFrame:
    signal_path = Path(path)
    if not signal_path.exists():
        raise FileNotFoundError(signal_path)
    if signal_path.suffix.lower() == ".csv":
        return pd.read_csv(signal_path)
    return pd.read_json(signal_path, lines=True)


def _load_config(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object config: {config_path}")
    return data


def _float_config(config: dict[str, Any], key: str, fallback: float) -> float:
    try:
        value = config.get(key, fallback)
        return float(value)
    except Exception:
        return fallback


def _config_weights(config: dict[str, Any], fallback: str) -> list[float]:
    keys = ("weight_boundary", "weight_calibration_gap", "weight_evidence")
    if all(key in config for key in keys):
        return [float(config[key]) for key in keys]
    values = [item.strip() for item in str(fallback).split(",") if item.strip()]
    if len(values) != 3:
        raise ValueError("--weights must contain boundary,calibration_gap,evidence")
    return [float(item) for item in values]


def _score_lookup(scored_df: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    lookup: dict[tuple[str, str, str], float] = {}
    for row in scored_df.to_dict(orient="records"):
        item_id = text(row.get("candidate_item_id")) or text(row.get("item_id"))
        key = (text(row.get("source_event_id")), text(row.get("user_id")), item_id)
        if all(key):
            lookup[key] = float(row["ccrp_risk_adjusted_score"])
    return lookup


def _candidate_score_rows(candidate_rows: list[dict[str, Any]], scored_df: pd.DataFrame) -> list[dict[str, Any]]:
    lookup = _score_lookup(scored_df)
    rows: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        key = (
            text(candidate.get("source_event_id")),
            text(candidate.get("user_id")),
            text(candidate.get("item_id")),
        )
        rows.append(
            {
                "source_event_id": key[0],
                "user_id": key[1],
                "item_id": key[2],
                "score": lookup.get(key, float("nan")),
            }
        )
    return rows


def export_ccrp_scored_rows(
    *,
    signal_path: str | Path,
    candidate_items_path: str | Path,
    output_dir: str | Path,
    domain: str = "",
    split: str = "test",
    selected_config_json: str | Path = "",
    score_mode: str = "full",
    ablation: str = "full",
    eta: float = 1.0,
    confidence_weight: float = 0.7,
    weights: str = "0.5,0.3,0.2",
    expected_rows: int = 0,
    allow_degenerate_scores: bool = False,
) -> dict[str, Any]:
    config = _load_config(selected_config_json)
    chosen_score_mode = str(config.get("score_mode", score_mode))
    chosen_ablation = str(config.get("ablation", ablation))
    chosen_eta = _float_config(config, "eta", eta)
    chosen_confidence_weight = _float_config(config, "confidence_weight", confidence_weight)
    chosen_weights = _config_weights(config, weights)

    signal_df = _load_signal(signal_path)
    candidate_rows = read_csv_rows(candidate_items_path)
    if expected_rows and len(candidate_rows) != expected_rows:
        raise ValueError(f"candidate row count mismatch: expected {expected_rows}, got {len(candidate_rows)}")

    scored_df = apply_ccrp_scores(
        signal_df,
        weights=parse_weights(chosen_weights),
        eta=chosen_eta,
        ablation=chosen_ablation,
        score_mode=chosen_score_mode,
        confidence_weight=chosen_confidence_weight,
    )
    score_rows = _candidate_score_rows(candidate_rows, scored_df)
    coverage_audit = audit_score_rows_against_candidates(candidate_rows=candidate_rows, score_rows=score_rows)
    degeneracy_audit = audit_score_degeneracy(score_rows)
    if not coverage_audit["audit_ok"]:
        raise ValueError(f"C-CRP score coverage audit failed: {coverage_audit}")
    if not allow_degenerate_scores and not degeneracy_audit["degeneracy_audit_ok"]:
        raise ValueError(f"C-CRP score degeneracy audit failed: {degeneracy_audit}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scored_rows_path = out / "ccrp_scored_rows.csv"
    scored_df.to_csv(scored_rows_path, index=False)
    score_summary = write_score_rows(score_rows, out / "ccrp_scores.csv")
    provenance = {
        "status_label": "ccrp_scored_rows_rebuilt_from_saved_signal",
        "artifact_class": "paper_critical_input_reconstruction",
        "domain": domain,
        "split": split,
        "git_commit": _git_commit(),
        "signal_path": str(signal_path),
        "candidate_items_path": str(candidate_items_path),
        "selected_config_json": str(selected_config_json) if selected_config_json else "",
        "signal_sha256": sha256_file(signal_path),
        "candidate_items_sha256": sha256_file(candidate_items_path),
        "scored_rows_path": str(scored_rows_path),
        "scored_rows_sha256": sha256_file(scored_rows_path),
        "scored_rows": int(len(scored_df)),
        "score_mode": chosen_score_mode,
        "ablation": chosen_ablation,
        "eta": chosen_eta,
        "confidence_weight": chosen_confidence_weight,
        "weight_boundary": chosen_weights[0],
        "weight_calibration_gap": chosen_weights[1],
        "weight_evidence": chosen_weights[2],
        **score_summary,
        **coverage_audit,
        **degeneracy_audit,
    }
    write_json(provenance, out / "ccrp_scored_rows_provenance.json")
    return provenance


def main() -> None:
    args = parse_args()
    provenance = export_ccrp_scored_rows(
        signal_path=args.signal_path,
        candidate_items_path=args.candidate_items_path,
        output_dir=args.output_dir,
        domain=args.domain,
        split=args.split,
        selected_config_json=args.selected_config_json,
        score_mode=args.score_mode,
        ablation=args.ablation,
        eta=args.eta,
        confidence_weight=args.confidence_weight,
        weights=args.weights,
        expected_rows=args.expected_rows,
        allow_degenerate_scores=args.allow_degenerate_scores,
    )
    print(json.dumps({"ok": True, "provenance": provenance}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
