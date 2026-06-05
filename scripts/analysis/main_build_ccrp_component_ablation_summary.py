from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.misc.main_select_ccrp_variant_on_valid import _evaluate_candidate_scores, _weight_label
from src.baselines.internal_scores import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ABLATIONS = (
    "full",
    "without_boundary_uncertainty",
    "without_calibration_gap",
    "without_evidence_support",
    "without_counterevidence",
    "without_risk_penalty",
)
FULL_KS = (5, 10, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the paper-critical C-CRP leave-one-component-out ablation "
            "summary after validation selection. This uses the selected valid "
            "configuration and evaluates each requested ablation on the test split "
            "without selecting on test."
        )
    )
    parser.add_argument("--selector_dir", required=True, help="Directory produced by main_select_ccrp_variant_on_valid.py.")
    parser.add_argument("--output_dir", default="", help="Defaults to --selector_dir.")
    parser.add_argument("--domain", default="")
    parser.add_argument("--selected_config_json", default="", help="Defaults to <selector_dir>/selected_valid_config.json.")
    parser.add_argument("--selector_provenance_json", default="", help="Defaults to <selector_dir>/ccrp_internal_provenance.json.")
    parser.add_argument("--valid_sweep_csv", default="", help="Defaults to <selector_dir>/valid_ccrp_sweep.csv.")
    parser.add_argument("--test_ranking_path", default="")
    parser.add_argument("--test_candidate_items_path", default="")
    parser.add_argument("--test_signal_path", default="")
    parser.add_argument("--ablations", default=",".join(DEFAULT_ABLATIONS))
    parser.add_argument("--expected_events", type=int, default=10000)
    parser.add_argument("--expected_candidates_per_event", type=int, default=101)
    parser.add_argument("--metric", default="NDCG@10")
    parser.add_argument("--skip_plot", action="store_true")
    parser.add_argument(
        "--allow_non_full_score_mode",
        action="store_true",
        help="Allow non-full score modes. By default true component ablation claims require score_mode=full.",
    )
    parser.add_argument("--status_label", default="paper_critical_component_ablation_ready")
    parser.add_argument("--row_status_label", default="same_schema_internal_ablation")
    return parser.parse_args()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


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


def _selected_weights(selected: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(selected["weight_boundary"]),
        float(selected["weight_calibration_gap"]),
        float(selected["weight_evidence"]),
    )


def _path_arg(value: str, fallback: str | Path) -> str:
    return str(value or fallback)


def _input_sha256(paths: dict[str, str]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items() if path and Path(path).exists()}


def _valid_sweep_ablations(path: str | Path) -> set[str]:
    df = pd.read_csv(path)
    if "ablation" not in df.columns:
        return set()
    return {str(value).strip() for value in df["ablation"].dropna().tolist() if str(value).strip()}


def _plot_summary(rows: list[dict[str, Any]], output_dir: Path, *, metric: str) -> list[str]:
    if not rows:
        return []
    import matplotlib  # type: ignore

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore

    df = pd.DataFrame(rows)
    if metric not in df.columns:
        raise ValueError(f"metric={metric!r} not present in component ablation summary")
    order = df["ablation"].astype(str).tolist()
    values = df[metric].astype(float).tolist()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(range(len(order)), values, color="#4c78a8")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel(metric)
    ax.set_title("C-CRP leave-one-component-out ablation")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths: list[str] = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"fig_component_ablation.{suffix}"
        fig.savefig(path, dpi=200)
        paths.append(str(path))
    plt.close(fig)
    return paths


def build_component_ablation_package(
    *,
    selector_dir: str | Path,
    output_dir: str | Path | None = None,
    domain: str = "",
    selected_config_json: str | Path | None = None,
    selector_provenance_json: str | Path | None = None,
    valid_sweep_csv: str | Path | None = None,
    test_ranking_path: str | Path | None = None,
    test_candidate_items_path: str | Path | None = None,
    test_signal_path: str | Path | None = None,
    ablations: list[str] | tuple[str, ...] = DEFAULT_ABLATIONS,
    expected_events: int = 10000,
    expected_candidates_per_event: int = 101,
    metric: str = "NDCG@10",
    skip_plot: bool = False,
    require_full_score_mode: bool = True,
    status_label: str = "paper_critical_component_ablation_ready",
    row_status_label: str = "same_schema_internal_ablation",
) -> dict[str, Any]:
    selector = Path(selector_dir)
    out = Path(output_dir) if output_dir else selector
    out.mkdir(parents=True, exist_ok=True)
    selected_path = Path(selected_config_json) if selected_config_json else selector / "selected_valid_config.json"
    provenance_path = Path(selector_provenance_json) if selector_provenance_json else selector / "ccrp_internal_provenance.json"
    sweep_path = Path(valid_sweep_csv) if valid_sweep_csv else selector / "valid_ccrp_sweep.csv"
    selected = _read_json(selected_path)
    selector_provenance = _read_json(provenance_path)

    selected_domain = domain or str(selector_provenance.get("domain") or selected.get("domain") or "")
    if not selected_domain:
        raise ValueError("domain is required when it is absent from selector provenance")
    test_ranking = _path_arg(str(test_ranking_path or ""), selector_provenance.get("test_ranking_path", ""))
    test_candidates = _path_arg(str(test_candidate_items_path or ""), selector_provenance.get("test_candidate_items_path", ""))
    test_signal = _path_arg(str(test_signal_path or ""), selector_provenance.get("test_signal_path", ""))
    for label, path in {
        "selected_config_json": str(selected_path),
        "selector_provenance_json": str(provenance_path),
        "valid_sweep_csv": str(sweep_path),
        "test_ranking_path": test_ranking,
        "test_candidate_items_path": test_candidates,
        "test_signal_path": test_signal,
    }.items():
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"{label} is missing: {path}")

    weights = _selected_weights(selected)
    score_mode = str(selected["score_mode"])
    eta = float(selected["eta"])
    confidence_weight = float(selected["confidence_weight"])
    selected_on = str(selector_provenance.get("selected_on") or "").strip().lower()
    selected_split = str(selected.get("split") or "").strip().lower()
    sweep_ablations = _valid_sweep_ablations(sweep_path)
    rows: list[dict[str, Any]] = []
    k = max(FULL_KS)
    for ablation in ablations:
        metrics, _, _ = _evaluate_candidate_scores(
            ranking_path=test_ranking,
            candidate_items_path=test_candidates,
            signal_path=test_signal,
            score_mode=score_mode,
            ablation=ablation,
            eta=eta,
            confidence_weight=confidence_weight,
            weights=weights,
            k=k,
            ks=FULL_KS,
            fail_on_degeneracy=False,
        )
        n_events = int(metrics.get("sample_count") or 0)
        candidate_key_count = int(metrics.get("candidate_key_count") or 0)
        expected_candidate_keys = expected_events * expected_candidates_per_event if expected_events > 0 else candidate_key_count
        rows.append(
            {
                "domain": selected_domain,
                "split": "test",
                "ablation": ablation,
                "status_label": row_status_label,
                "selected_on": "valid",
                "selected_on_test": False,
                "score_mode": score_mode,
                "eta": eta,
                "confidence_weight": confidence_weight,
                "weight_boundary": weights[0],
                "weight_calibration_gap": weights[1],
                "weight_evidence": weights[2],
                "weight_grid_label": _weight_label(weights),
                "n_events": n_events,
                "expected_events": expected_events,
                "expected_candidates_per_event": expected_candidates_per_event,
                "expected_candidate_key_count": expected_candidate_keys,
                **metrics,
            }
        )
    summary_path = out / "component_ablation_summary.csv"
    _write_csv(summary_path, rows)
    _write_json(out / "component_ablation_summary.json", {"rows": rows})
    figure_paths = [] if skip_plot else _plot_summary(rows, out, metric=metric)
    input_paths = {
        "selected_config_json": str(selected_path),
        "selector_provenance_json": str(provenance_path),
        "valid_sweep_csv": str(sweep_path),
        "test_ranking_path": test_ranking,
        "test_candidate_items_path": test_candidates,
        "test_signal_path": test_signal,
    }
    failures: list[str] = []
    if selected_on != "valid":
        failures.append(f"selector_provenance_selected_on_not_valid:{selected_on or 'missing'}")
    if selected_split != "valid":
        failures.append(f"selected_config_split_not_valid:{selected_split or 'missing'}")
    if require_full_score_mode and score_mode != "full":
        failures.append(f"selected_score_mode_not_full_for_component_ablation:{score_mode}")
    missing_sweep_ablations = [ablation for ablation in ablations if ablation not in sweep_ablations]
    if missing_sweep_ablations:
        failures.append(f"valid_sweep_missing_ablation:{','.join(missing_sweep_ablations)}")
    for row in rows:
        if row["n_events"] != expected_events:
            failures.append(f"{row['ablation']}:n_events:{row['n_events']}!={expected_events}")
        if row["candidate_key_count"] != row["expected_candidate_key_count"]:
            failures.append(
                f"{row['ablation']}:candidate_key_count:{row['candidate_key_count']}!={row['expected_candidate_key_count']}"
            )
        if row["score_coverage_rate"] != 1.0:
            failures.append(f"{row['ablation']}:score_coverage_rate:{row['score_coverage_rate']}")
        if not row["audit_ok"]:
            failures.append(f"{row['ablation']}:audit_ok_false")
        if not row["degeneracy_audit_ok"]:
            failures.append(f"{row['ablation']}:degeneracy_audit_false")
    provenance = {
        "artifact_class": "paper_critical_component_ablation",
        "status_label": status_label,
        "paper_claim_scope": "leave_one_component_out_component_evidence_candidate",
        "claim_limits": [
            "Uses validation-selected score mode, eta, confidence weight, and weight triple.",
            "Does not select ablation variants on test.",
            "Requires the Phase 2.5 module package audit before paper claims.",
        ],
        "ok": not failures,
        "failures": failures,
        "domain": selected_domain,
        "git_commit": _git_commit(),
        "command": " ".join(sys.argv),
        "input_paths": input_paths,
        "input_sha256": _input_sha256(input_paths),
        "selected_config": {
            "score_mode": score_mode,
            "eta": eta,
            "confidence_weight": confidence_weight,
            "weight_boundary": weights[0],
            "weight_calibration_gap": weights[1],
            "weight_evidence": weights[2],
            "weight_grid_label": _weight_label(weights),
        },
        "selection_source": {
            "selected_on": selected_on,
            "selected_config_split": selected_split,
            "selection_metric": selector_provenance.get("selection_metric", ""),
            "selected_valid_metric": selector_provenance.get(
                f"selected_valid_{selector_provenance.get('selection_metric', '')}", ""
            ),
        },
        "require_full_score_mode": require_full_score_mode,
        "valid_sweep_ablations": sorted(sweep_ablations),
        "ablations": list(ablations),
        "required_metrics": ["MRR"] + [f"HR@{k}" for k in FULL_KS] + [f"NDCG@{k}" for k in FULL_KS],
        "expected_events": expected_events,
        "expected_candidates_per_event": expected_candidates_per_event,
        "summary_path": str(summary_path),
        "figure_paths": figure_paths,
    }
    _write_json(out / "component_ablation_provenance.json", provenance)
    return provenance


def main() -> None:
    args = parse_args()
    provenance = build_component_ablation_package(
        selector_dir=args.selector_dir,
        output_dir=args.output_dir or None,
        domain=args.domain,
        selected_config_json=args.selected_config_json or None,
        selector_provenance_json=args.selector_provenance_json or None,
        valid_sweep_csv=args.valid_sweep_csv or None,
        test_ranking_path=args.test_ranking_path or None,
        test_candidate_items_path=args.test_candidate_items_path or None,
        test_signal_path=args.test_signal_path or None,
        ablations=_parse_list(args.ablations),
        expected_events=args.expected_events,
        expected_candidates_per_event=args.expected_candidates_per_event,
        metric=args.metric,
        skip_plot=args.skip_plot,
        require_full_score_mode=not args.allow_non_full_score_mode,
        status_label=args.status_label,
        row_status_label=args.row_status_label,
    )
    print(
        json.dumps(
            {
                "ok": provenance["ok"],
                "output_dir": str(Path(args.output_dir or args.selector_dir)),
                "ablations": provenance["ablations"],
                "failures": provenance["failures"],
            },
            indent=2,
        )
    )
    if not provenance["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
