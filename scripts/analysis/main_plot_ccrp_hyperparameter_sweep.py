from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build paper-critical C-CRP hyperparameter curve tables and figures "
            "from valid_ccrp_sweep.csv produced by main_select_ccrp_variant_on_valid.py."
        )
    )
    parser.add_argument("--sweep_csv", required=True)
    parser.add_argument(
        "--test_sweep_csv",
        default="",
        help=(
            "Optional test-split sweep with the same schema. When provided, curves are "
            "reported separately for valid and test rather than validation-only."
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metric", default="NDCG@10")
    parser.add_argument("--domain", default="")
    parser.add_argument("--score_mode", default="full")
    parser.add_argument("--ablation", default="full")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--confidence_weight", type=float, default=0.5)
    parser.add_argument("--weight_grid_label", default="0.5,0.3,0.2")
    parser.add_argument(
        "--controls",
        default="eta,confidence_weight,weight_grid_label",
        help="Comma-separated controls to plot: eta, confidence_weight, weight_grid_label.",
    )
    parser.add_argument("--min_values", type=int, default=3)
    parser.add_argument("--allow_incomplete", action="store_true")
    parser.add_argument("--skip_plot", action="store_true")
    parser.add_argument(
        "--require_audit_ok",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require audit_ok and degeneracy_audit_ok columns to exist and be true for paper-facing curves.",
    )
    return parser.parse_args()


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _format_float(value: float) -> str:
    return f"{float(value):g}"


def _parse_controls(value: str) -> list[str]:
    allowed = {"eta", "confidence_weight", "weight_grid_label"}
    controls = [item.strip() for item in str(value or "").split(",") if item.strip()]
    unknown = sorted(set(controls) - allowed)
    if unknown:
        raise ValueError(f"Unsupported controls: {unknown}; expected subset of {sorted(allowed)}")
    return controls or ["eta", "confidence_weight", "weight_grid_label"]


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    h = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_sweeps(valid_path: str | Path, test_path: str | Path | None = None) -> pd.DataFrame:
    valid = pd.read_csv(valid_path)
    valid["split"] = "valid"
    frames = [valid]
    if test_path:
        test = pd.read_csv(test_path)
        test["split"] = "test"
        frames.append(test)
    return pd.concat(frames, ignore_index=True)


def _filter_audited_rows(df: pd.DataFrame, *, require_audit_ok: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    before = len(out)
    dropped = 0
    required_columns = ("audit_ok", "degeneracy_audit_ok")
    missing_columns = [column for column in required_columns if column not in out.columns]
    if require_audit_ok and missing_columns:
        raise ValueError(f"Sweep CSV missing required audit columns: {missing_columns}")
    if require_audit_ok:
        for column in required_columns:
            mask = out[column].map(_as_bool)
            dropped += int((~mask).sum())
            out = out[mask].copy()
    return out, {
        "input_rows": int(before),
        "audited_rows": int(len(out)),
        "dropped_audit_rows": int(dropped),
        "require_audit_ok": bool(require_audit_ok),
        "required_audit_columns": list(required_columns),
        "missing_audit_columns": missing_columns,
    }


def _filter_exact(df: pd.DataFrame, column: str, value: Any) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"Sweep CSV missing required column: {column}")
    if isinstance(value, float):
        return df[df[column].map(_as_float).sub(float(value)).abs() <= 1e-12].copy()
    return df[df[column].map(_text) == _text(value)].copy()


def _sort_curve(df: pd.DataFrame, control: str) -> pd.DataFrame:
    if control in {"eta", "confidence_weight"}:
        out = df.copy()
        out["_sort_value"] = out["control_value"].map(_as_float)
        return out.sort_values("_sort_value").drop(columns=["_sort_value"]).reset_index(drop=True)
    return df.sort_values("control_value").reset_index(drop=True)


def _select_best_per_value(df: pd.DataFrame, *, control: str, metric: str) -> pd.DataFrame:
    if metric not in df.columns:
        raise ValueError(f"Sweep CSV missing metric column: {metric}")
    rows: list[dict[str, Any]] = []
    for value, group in df.groupby(control, dropna=False, sort=False):
        metric_values = group[metric].map(_as_float)
        if metric_values.isna().all():
            continue
        best_idx = metric_values.idxmax()
        best = group.loc[best_idx].to_dict()
        best["control"] = control
        best["control_value"] = _format_float(float(value)) if control in {"eta", "confidence_weight"} else _text(value)
        best["metric_name"] = metric
        best["metric_value"] = float(best[metric])
        best["candidate_rows_for_value"] = int(len(group))
        rows.append(best)
    return _sort_curve(pd.DataFrame(rows), control) if rows else pd.DataFrame()


def _curve_source(
    df: pd.DataFrame,
    *,
    control: str,
    score_mode: str,
    ablation: str,
    eta: float,
    confidence_weight: float,
    weight_grid_label: str,
) -> pd.DataFrame:
    out = _filter_exact(df, "ablation", ablation)
    if control == "confidence_weight":
        out = _filter_exact(out, "score_mode", "confidence_plus_evidence")
        out = _filter_exact(out, "eta", eta)
        out = _filter_exact(out, "weight_grid_label", weight_grid_label)
    elif control == "eta":
        out = _filter_exact(out, "score_mode", score_mode)
        out = _filter_exact(out, "weight_grid_label", weight_grid_label)
        if score_mode == "confidence_plus_evidence":
            out = _filter_exact(out, "confidence_weight", confidence_weight)
    elif control == "weight_grid_label":
        out = _filter_exact(out, "score_mode", score_mode)
        out = _filter_exact(out, "eta", eta)
        if score_mode == "confidence_plus_evidence":
            out = _filter_exact(out, "confidence_weight", confidence_weight)
    else:
        raise ValueError(f"Unsupported control: {control}")
    return out


def _evidence_label(
    *,
    reporting_mode: str,
    allow_incomplete: bool,
    require_audit_ok: bool,
    control_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    incomplete_controls = [
        {"split": row["split"], "control": row["control"], "curve_values": row["curve_values"]}
        for row in control_reports
        if not bool(row.get("meets_min_values"))
    ]
    if not require_audit_ok:
        status_label = "diagnostic_hyperparameter_curve_audit_not_enforced"
        claim_scope = "diagnostic_only_not_paper_stability"
    elif incomplete_controls or allow_incomplete:
        status_label = "diagnostic_hyperparameter_curve_incomplete"
        claim_scope = "diagnostic_only_not_paper_stability"
    elif reporting_mode == "valid_only":
        status_label = "validation_only_hyperparameter_selection_curve"
        claim_scope = "validation_only_not_stability_claim"
    else:
        status_label = "paper_critical_hyperparameter_curve_ready"
        claim_scope = "valid_and_test_stability_curve_candidate"
    return {
        "status_label": status_label,
        "paper_claim_scope": claim_scope,
        "incomplete_controls": incomplete_controls,
        "claim_limits": [
            "Hyperparameter curves do not replace official-baseline, ablation, or paired-test gates.",
            "Validation-only curves support selection/protocol transparency, not paper-facing stability claims.",
            "Diagnostic or audit-not-enforced curves must not be used as main paper evidence.",
        ],
    }


def build_hyperparameter_summary(
    sweep_csv: str | Path,
    *,
    test_sweep_csv: str | Path | None = None,
    domain: str = "",
    metric: str = "NDCG@10",
    score_mode: str = "full",
    ablation: str = "full",
    eta: float = 1.0,
    confidence_weight: float = 0.5,
    weight_grid_label: str = "0.5,0.3,0.2",
    controls: list[str] | None = None,
    min_values: int = 3,
    allow_incomplete: bool = False,
    require_audit_ok: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(sweep_csv)
    if not path.exists():
        raise FileNotFoundError(path)
    test_path = Path(test_sweep_csv) if test_sweep_csv else None
    if test_path and not test_path.exists():
        raise FileNotFoundError(test_path)
    df = _load_sweeps(path, test_path)
    if df.empty:
        raise ValueError(f"Empty sweep CSV: {path}")
    audited, audit_summary = _filter_audited_rows(df, require_audit_ok=require_audit_ok)
    controls = controls or ["eta", "confidence_weight", "weight_grid_label"]

    summary_frames: list[pd.DataFrame] = []
    control_reports: list[dict[str, Any]] = []
    split_values = sorted(audited["split"].dropna().astype(str).unique().tolist())
    for split in split_values:
        split_df = audited[audited["split"].astype(str) == split].copy()
        for control in controls:
            source = _curve_source(
                split_df,
                control=control,
                score_mode=score_mode,
                ablation=ablation,
                eta=eta,
                confidence_weight=confidence_weight,
                weight_grid_label=weight_grid_label,
            )
            curve = _select_best_per_value(source, control=control, metric=metric)
            distinct_values = int(curve["control_value"].nunique()) if not curve.empty else 0
            control_reports.append(
                {
                    "split": split,
                    "control": control,
                    "source_rows": int(len(source)),
                    "curve_values": distinct_values,
                    "meets_min_values": bool(distinct_values >= min_values),
                }
            )
            if distinct_values < min_values and not allow_incomplete:
                raise ValueError(
                    f"Split {split} control {control} has {distinct_values} plotted values, below min_values={min_values}. "
                    "Use a larger sweep or --allow_incomplete for diagnostic-only output."
                )
            if not curve.empty:
                if "domain" in curve.columns:
                    curve["domain"] = domain or curve["domain"]
                else:
                    curve.insert(0, "domain", domain)
                curve["split"] = split
                summary_frames.append(curve)

    if not summary_frames:
        raise ValueError("No hyperparameter curves could be built from the sweep.")
    summary = pd.concat(summary_frames, ignore_index=True)
    reporting_mode = "valid_and_test" if test_path else "valid_only"
    evidence_label = _evidence_label(
        reporting_mode=reporting_mode,
        allow_incomplete=allow_incomplete,
        require_audit_ok=require_audit_ok,
        control_reports=control_reports,
    )
    provenance = {
        "artifact_class": "paper_critical_hyperparameter_analysis",
        **evidence_label,
        "domain": domain,
        "git_commit": _git_commit(),
        "command": " ".join(sys.argv),
        "sweep_csv": str(path),
        "sweep_sha256": sha256_file(path),
        "test_sweep_csv": str(test_path) if test_path else "",
        "test_sweep_sha256": sha256_file(test_path) if test_path else "",
        "reporting_mode": reporting_mode,
        "metric": metric,
        "filters": {
            "score_mode": score_mode,
            "ablation": ablation,
            "eta": eta,
            "confidence_weight": confidence_weight,
            "weight_grid_label": weight_grid_label,
        },
        "controls": controls,
        "min_values": min_values,
        "allow_incomplete": allow_incomplete,
        "audit_summary": audit_summary,
        "control_reports": control_reports,
    }
    return summary, provenance


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=_json_default)


def _plot_control(summary: pd.DataFrame, output_dir: Path, *, control: str, metric: str) -> list[str]:
    plot_df = summary[summary["control"] == control].copy()
    if plot_df.empty:
        return []
    import matplotlib.pyplot as plt  # type: ignore

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for split, group in plot_df.groupby("split", sort=True):
        group = group.sort_values("control_value", key=lambda s: s.map(_as_float) if control in {"eta", "confidence_weight"} else s)
        if control in {"eta", "confidence_weight"}:
            x = group["control_value"].map(_as_float)
            ax.plot(x, group["metric_value"].astype(float), marker="o", linewidth=1.8, label=split)
            ax.set_xlabel(control)
        else:
            x = np.arange(len(group))
            ax.plot(x, group["metric_value"].astype(float), marker="o", linewidth=1.8, label=split)
            ax.set_xticks(x)
            ax.set_xticklabels(group["control_value"].astype(str), rotation=25, ha="right")
            ax.set_xlabel("C-CRP weight triple")
    ax.set_ylabel(metric)
    ax.set_title(f"C-CRP hyperparameter sensitivity: {control}")
    ax.grid(True, alpha=0.25)
    if plot_df["split"].nunique() > 1:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    filenames = {
        "eta": "fig_hyper_eta_curve",
        "confidence_weight": "fig_hyper_confidence_weight_curve",
        "weight_grid_label": "fig_hyper_weight_simplex_or_lines",
    }
    stem = filenames[control]
    paths: list[str] = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=200)
        paths.append(str(path))
    plt.close(fig)
    return paths


def write_outputs(summary: pd.DataFrame, provenance: dict[str, Any], output_dir: str | Path, *, skip_plot: bool) -> None:
    out = Path(output_dir)
    _write_csv(out / "ccrp_hyperparameter_curve_summary.csv", summary)
    figure_paths: list[str] = []
    if not skip_plot:
        metric = str(provenance["metric"])
        for control in provenance["controls"]:
            figure_paths.extend(_plot_control(summary, out, control=control, metric=metric))
    provenance["figure_paths"] = figure_paths
    _write_json(out / "ccrp_hyperparameter_curve_provenance.json", provenance)


def main() -> None:
    args = parse_args()
    controls = _parse_controls(args.controls)
    summary, provenance = build_hyperparameter_summary(
        args.sweep_csv,
        test_sweep_csv=args.test_sweep_csv or None,
        domain=args.domain,
        metric=args.metric,
        score_mode=args.score_mode,
        ablation=args.ablation,
        eta=args.eta,
        confidence_weight=args.confidence_weight,
        weight_grid_label=args.weight_grid_label,
        controls=controls,
        min_values=args.min_values,
        allow_incomplete=args.allow_incomplete,
        require_audit_ok=args.require_audit_ok,
    )
    write_outputs(summary, provenance, args.output_dir, skip_plot=args.skip_plot)
    print(json.dumps({"ok": True, "output_dir": args.output_dir, "rows": int(len(summary))}, indent=2))


if __name__ == "__main__":
    main()
