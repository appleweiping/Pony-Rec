from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[2]
_repo_root_text = str(_REPO_ROOT)
if _repo_root_text not in sys.path:
    sys.path.insert(0, _repo_root_text)


DEFAULT_UNCERTAINTY_COLUMNS = (
    "ccrp_uncertainty",
    "uncertainty",
    "shadow_uncertainty",
    "risk_uncertainty",
)
DEFAULT_KS = (5, 10, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the paper-critical uncertainty observation/motivation tables "
            "from event-level C-CRP uncertainty and same-candidate ranking eval records."
        )
    )
    parser.add_argument("--domain", required=True)
    parser.add_argument(
        "--uncertainty_scores_path",
        required=True,
        help=(
            "CSV or JSONL with source_event_id/user_id/item_id and an uncertainty column. "
            "Do not pass final C-CRP scores.csv unless it also contains uncertainty fields."
        ),
    )
    parser.add_argument("--ccrp_eval_path", required=True, help="C-CRP ranking_eval_records.csv.")
    parser.add_argument(
        "--method_eval",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="Representative method ranking_eval_records.csv. Repeat for official baselines.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--uncertainty_col", default="", help="Defaults to auto-detecting a C-CRP uncertainty column.")
    parser.add_argument("--event_agg", choices=("mean", "max", "median", "p90"), default="mean")
    parser.add_argument("--n_bins", type=int, default=5)
    parser.add_argument("--ks", default="5,10,20")
    parser.add_argument("--primary_metric", default="NDCG@10")
    parser.add_argument("--expected_events", type=int, default=10000)
    parser.add_argument(
        "--expected_candidates_per_event",
        type=int,
        default=101,
        help="If ranking eval records include num_candidates, require this exact candidate count. Use 0 to disable.",
    )
    parser.add_argument("--min_join_rate", type=float, default=0.999)
    parser.add_argument("--skip_plot", action="store_true")
    return parser.parse_args()


def _read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(table_path)
    if table_path.suffix.lower() == ".jsonl":
        rows = []
        with table_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if text:
                    rows.append(json.loads(text))
        return pd.DataFrame(rows)
    return pd.read_csv(table_path)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _event_col(df: pd.DataFrame) -> str:
    for col in ("source_event_id", "event_id", "user_id"):
        if col in df.columns:
            return col
    raise ValueError("Expected one of source_event_id, event_id, or user_id.")


def _finite_float(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


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


def _parse_ks(value: str) -> list[int]:
    out: list[int] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        k = int(item)
        if k <= 0:
            raise ValueError(f"Metric cutoff must be positive: {item}")
        if k not in out:
            out.append(k)
    return sorted(out or list(DEFAULT_KS))


def _select_uncertainty_column(df: pd.DataFrame, requested: str) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested uncertainty column {requested!r} not found in {list(df.columns)}")
        return requested
    for col in DEFAULT_UNCERTAINTY_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(
        "No uncertainty column found. Expected one of "
        f"{list(DEFAULT_UNCERTAINTY_COLUMNS)}. The final C-CRP scores.csv is not sufficient "
        "for the motivation study unless uncertainty fields are included."
    )


def load_event_uncertainty(
    path: str | Path,
    *,
    uncertainty_col: str = "",
    event_agg: str = "mean",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = _read_table(path)
    if df.empty:
        raise ValueError(f"Empty uncertainty input: {path}")
    event_column = _event_col(df)
    selected_col = _select_uncertainty_column(df, uncertainty_col)

    work = df[[event_column, selected_col]].copy()
    work["event_id"] = work[event_column].map(_text)
    work["uncertainty_value"] = work[selected_col].map(_finite_float)
    invalid = int(work["uncertainty_value"].isna().sum())
    work = work[work["event_id"].astype(bool) & work["uncertainty_value"].notna()].copy()
    if work.empty:
        raise ValueError(f"No finite event uncertainty rows in {path}")

    group = work.groupby("event_id", sort=False)["uncertainty_value"]
    if event_agg == "mean":
        uncertainty = group.mean()
    elif event_agg == "max":
        uncertainty = group.max()
    elif event_agg == "median":
        uncertainty = group.median()
    elif event_agg == "p90":
        uncertainty = group.quantile(0.9)
    else:
        raise ValueError(f"Unsupported event_agg={event_agg}")

    counts = group.size().rename("candidate_rows")
    out = pd.concat([uncertainty.rename("event_uncertainty"), counts], axis=1).reset_index()
    summary = {
        "uncertainty_input_rows": int(len(df)),
        "finite_uncertainty_rows": int(len(work)),
        "invalid_uncertainty_rows": invalid,
        "event_count": int(len(out)),
        "uncertainty_col": selected_col,
        "event_agg": event_agg,
    }
    return out, summary


def assign_uncertainty_bins(event_df: pd.DataFrame, *, n_bins: int) -> pd.DataFrame:
    if event_df.empty:
        raise ValueError("Cannot bin an empty event uncertainty frame.")
    bins = max(1, min(int(n_bins), len(event_df)))
    ordered = event_df.sort_values(["event_uncertainty", "event_id"], kind="mergesort").reset_index(drop=True)
    ordered["uncertainty_bin_index"] = (np.arange(len(ordered)) * bins // len(ordered)).astype(int)
    stats = (
        ordered.groupby("uncertainty_bin_index", sort=True)["event_uncertainty"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "bin_uncertainty_min", "max": "bin_uncertainty_max"})
    )
    ordered = ordered.merge(stats, on="uncertainty_bin_index", how="left")
    ordered["uncertainty_bin"] = ordered.apply(
        lambda row: (
            f"Q{int(row['uncertainty_bin_index']) + 1:02d} "
            f"[{row['bin_uncertainty_min']:.6g},{row['bin_uncertainty_max']:.6g}]"
        ),
        axis=1,
    )
    return ordered


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


def load_method_eval(
    path: str | Path,
    *,
    method: str,
    ks: list[int],
    expected_candidates_per_event: int = 0,
) -> pd.DataFrame:
    df = _read_table(path)
    if df.empty:
        raise ValueError(f"Empty method eval file: {path}")
    event_column = _event_col(df)
    if "positive_rank" not in df.columns:
        raise ValueError(f"Method eval file lacks positive_rank: {path}")

    event_ids = df[event_column].map(_text)
    duplicate_events = sorted({event_id for event_id in event_ids[event_ids.duplicated()].tolist() if event_id})
    if duplicate_events:
        preview = ", ".join(duplicate_events[:5])
        raise ValueError(f"{method} eval file has duplicate event_id rows: {preview}")

    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(df.to_dict(orient="records")):
        event_id = _text(record.get(event_column)) or str(idx)
        rank_float = _finite_float(record["positive_rank"])
        if not math.isfinite(rank_float) or rank_float < 1 or abs(rank_float - round(rank_float)) > 1e-9:
            raise ValueError(f"{method} has invalid positive_rank={record.get('positive_rank')!r} for event {event_id}")
        rank = int(round(rank_float))
        num_candidates = None
        if "num_candidates" in df.columns:
            num_candidates_float = _finite_float(record.get("num_candidates"))
            if (
                not math.isfinite(num_candidates_float)
                or num_candidates_float < 1
                or abs(num_candidates_float - round(num_candidates_float)) > 1e-9
            ):
                raise ValueError(f"{method} has invalid num_candidates={record.get('num_candidates')!r} for event {event_id}")
            num_candidates = int(round(num_candidates_float))
            if expected_candidates_per_event > 0 and num_candidates != expected_candidates_per_event:
                raise ValueError(
                    f"{method} event {event_id} has num_candidates={num_candidates}, "
                    f"expected {expected_candidates_per_event}"
                )
            if rank > num_candidates:
                raise ValueError(f"{method} event {event_id} has positive_rank={rank} beyond num_candidates={num_candidates}")
        row: dict[str, Any] = {
            "event_id": event_id,
            "method": method,
            "positive_rank": rank,
            "MRR": _metric_from_rank(rank, "MRR"),
        }
        for k in ks:
            row[f"HR@{k}"] = _metric_from_rank(rank, f"HR@{k}")
            row[f"NDCG@{k}"] = _metric_from_rank(rank, f"NDCG@{k}")
        rows.append(row)
    return pd.DataFrame(rows)


def _parse_method_eval_specs(specs: list[str], ccrp_eval_path: str) -> list[tuple[str, str]]:
    parsed = [("ccrp_v3", ccrp_eval_path)]
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected METHOD=PATH for --method_eval, got {spec!r}")
        method, path = spec.split("=", 1)
        method = method.strip()
        path = path.strip()
        if not method or not path:
            raise ValueError(f"Expected METHOD=PATH for --method_eval, got {spec!r}")
        parsed.append((method, path))
    methods = [method for method, _ in parsed]
    if len(methods) != len(set(methods)):
        raise ValueError(f"Duplicate method names in observation inputs: {methods}")
    return parsed


def build_observation_tables(
    *,
    domain: str,
    uncertainty_scores_path: str,
    ccrp_eval_path: str,
    method_eval_specs: list[str],
    uncertainty_col: str,
    event_agg: str,
    n_bins: int,
    ks: list[int],
    expected_events: int,
    expected_candidates_per_event: int,
    min_join_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    event_uncertainty, uncertainty_summary = load_event_uncertainty(
        uncertainty_scores_path,
        uncertainty_col=uncertainty_col,
        event_agg=event_agg,
    )
    if expected_events > 0 and len(event_uncertainty) != expected_events:
        raise ValueError(
            f"Expected {expected_events} uncertainty events for {domain}, got {len(event_uncertainty)}. "
            "Resolve the signal path before using the motivation study in the paper."
        )
    binned = assign_uncertainty_bins(event_uncertainty, n_bins=n_bins)

    event_rows: list[pd.DataFrame] = []
    join_report: list[dict[str, Any]] = []
    input_paths: dict[str, str] = {"uncertainty_scores": uncertainty_scores_path, "ccrp_eval": ccrp_eval_path}
    metrics = ["MRR"] + [f"HR@{k}" for k in ks] + [f"NDCG@{k}" for k in ks]
    for method, path in _parse_method_eval_specs(method_eval_specs, ccrp_eval_path):
        method_eval = load_method_eval(
            path,
            method=method,
            ks=ks,
            expected_candidates_per_event=expected_candidates_per_event,
        )
        joined = binned.merge(method_eval, on="event_id", how="inner")
        uncertainty_event_ids = set(binned["event_id"].tolist())
        method_event_ids = set(method_eval["event_id"].tolist())
        missing_eval_events = len(uncertainty_event_ids - method_event_ids)
        extra_eval_events = len(method_event_ids - uncertainty_event_ids)
        join_rate = float(len(joined) / len(binned)) if len(binned) else 0.0
        join_report.append(
            {
                "method": method,
                "eval_path": path,
                "method_eval_rows": int(len(method_eval)),
                "joined_events": int(len(joined)),
                "ccrp_uncertainty_events": int(len(binned)),
                "missing_eval_events": int(missing_eval_events),
                "extra_eval_events": int(extra_eval_events),
                "exact_event_match": bool(missing_eval_events == 0 and extra_eval_events == 0),
                "join_rate": join_rate,
            }
        )
        if extra_eval_events:
            raise ValueError(
                f"{method} has {extra_eval_events} eval events not present in the C-CRP uncertainty input. "
                "Resolve the same-candidate event alignment before using the motivation study in the paper."
            )
        if join_rate < min_join_rate:
            raise ValueError(f"{method} join_rate={join_rate:.6f} below min_join_rate={min_join_rate:.6f}")
        joined["domain"] = domain
        event_rows.append(joined)
        input_paths[f"{method}_eval"] = path

    event_bins = pd.concat(event_rows, ignore_index=True)
    summary_rows: list[dict[str, Any]] = []
    for (method, bin_index, bin_label), group in event_bins.groupby(
        ["method", "uncertainty_bin_index", "uncertainty_bin"],
        sort=True,
    ):
        row: dict[str, Any] = {
            "domain": domain,
            "method": method,
            "uncertainty_bin_index": int(bin_index),
            "uncertainty_bin": bin_label,
            "n_events": int(len(group)),
            "event_uncertainty_mean": float(group["event_uncertainty"].mean()),
            "event_uncertainty_min": float(group["event_uncertainty"].min()),
            "event_uncertainty_max": float(group["event_uncertainty"].max()),
            "avg_positive_rank": float(group["positive_rank"].mean()),
        }
        for metric in metrics:
            row[metric] = float(group[metric].mean())
        summary_rows.append(row)

    for method, group in event_bins.groupby("method", sort=True):
        row = {
            "domain": domain,
            "method": method,
            "uncertainty_bin_index": -1,
            "uncertainty_bin": "ALL",
            "n_events": int(len(group)),
            "event_uncertainty_mean": float(group["event_uncertainty"].mean()),
            "event_uncertainty_min": float(group["event_uncertainty"].min()),
            "event_uncertainty_max": float(group["event_uncertainty"].max()),
            "avg_positive_rank": float(group["positive_rank"].mean()),
        }
        for metric in metrics:
            row[metric] = float(group[metric].mean())
        summary_rows.append(row)

    provenance = {
        "artifact_class": "paper_critical_observation_motivation",
        "status_label": "paper_critical_observation_ready",
        "paper_claim_scope": "motivation_only_not_main_table_sota",
        "claim_limits": [
            "Representative uncertainty-behavior evidence only.",
            "Does not replace full official-baseline comparison, ablation, or paired statistical gates.",
            "Requires real event-level uncertainty fields; score-only C-CRP outputs are rejected.",
        ],
        "domain": domain,
        "git_commit": _git_commit(),
        "command": " ".join(sys.argv),
        "input_paths": input_paths,
        "input_sha256": {name: sha256_file(path) for name, path in input_paths.items()},
        "uncertainty_summary": uncertainty_summary,
        "join_report": join_report,
        "expected_events": expected_events,
        "expected_candidates_per_event": expected_candidates_per_event,
        "min_join_rate": min_join_rate,
        "ks": ks,
        "required_metrics": metrics,
    }
    return event_bins, pd.DataFrame(summary_rows), provenance


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=_json_default)


def _plot_summary(summary: pd.DataFrame, output_dir: Path, *, primary_metric: str) -> list[str]:
    plot_df = summary[summary["uncertainty_bin"] != "ALL"].copy()
    if primary_metric not in plot_df.columns:
        raise ValueError(f"primary_metric={primary_metric!r} not found in summary columns.")
    import matplotlib.pyplot as plt  # type: ignore

    paths: list[str] = []
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method, group in plot_df.groupby("method", sort=True):
        group = group.sort_values("uncertainty_bin_index")
        ax.plot(
            group["uncertainty_bin_index"] + 1,
            group[primary_metric],
            marker="o",
            linewidth=1.8,
            label=method,
        )
    ax.set_xlabel("C-CRP uncertainty bin (low to high)")
    ax.set_ylabel(primary_metric)
    ax.set_title("Uncertainty-motivation diagnostic")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output_dir / f"fig_uncertainty_motivation.{suffix}"
        fig.savefig(path, dpi=200)
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ks = _parse_ks(args.ks)
    event_bins, summary, provenance = build_observation_tables(
        domain=args.domain,
        uncertainty_scores_path=args.uncertainty_scores_path,
        ccrp_eval_path=args.ccrp_eval_path,
        method_eval_specs=args.method_eval,
        uncertainty_col=args.uncertainty_col,
        event_agg=args.event_agg,
        n_bins=args.n_bins,
        ks=ks,
        expected_events=args.expected_events,
        expected_candidates_per_event=args.expected_candidates_per_event,
        min_join_rate=args.min_join_rate,
    )
    _write_csv(output_dir / "observation_event_bins.csv", event_bins)
    _write_csv(output_dir / "observation_summary.csv", summary)
    if not args.skip_plot:
        provenance["figure_paths"] = _plot_summary(summary, output_dir, primary_metric=args.primary_metric)
    _write_json(output_dir / "observation_summary.json", {"rows": summary.to_dict(orient="records")})
    _write_json(output_dir / "observation_provenance.json", provenance)
    print(json.dumps({"ok": True, "output_dir": str(output_dir), "rows": len(event_bins)}, indent=2))


if __name__ == "__main__":
    main()
