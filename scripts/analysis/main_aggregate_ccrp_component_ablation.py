from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOMAINS = ("sports", "toys", "home", "tools")
DEFAULT_ABLATIONS = (
    "full",
    "without_boundary_uncertainty",
    "without_calibration_gap",
    "without_evidence_support",
    "without_counterevidence",
    "without_risk_penalty",
)
FULL_METRICS = ("MRR", "HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate package-audited C-CRP leave-one-component-out results "
            "across the four new full-scale same-candidate domains."
        )
    )
    parser.add_argument("--package_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--metric", default="NDCG@10")
    parser.add_argument("--expected_events", type=int, default=10000)
    parser.add_argument("--expected_candidates_per_event", type=int, default=101)
    parser.add_argument("--tie_epsilon", type=float, default=1e-12)
    parser.add_argument("--skip_plot", action="store_true")
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _event_count(row: dict[str, str]) -> int:
    for key in ("event_count", "n_events", "sample_count"):
        if str(row.get(key, "")).strip():
            return _as_int(row.get(key))
    return 0


def _classification(deltas: list[float], *, eps: float) -> str:
    nonworse = sum(delta >= -eps for delta in deltas)
    worse = sum(delta < -eps for delta in deltas)
    if nonworse >= 3:
        return "removal_nonworse_in_3plus_domains_harmful_or_redundant"
    if worse >= 3:
        return "directionally_supportive_not_significant"
    return "mixed_diagnostic"


def _plot_component_summary(rows: list[dict[str, Any]], output_dir: Path, *, metric: str) -> list[str]:
    if not rows:
        return []
    import matplotlib  # type: ignore

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore

    labels = [str(row["ablation"]).replace("without_", "wo_") for row in rows]
    means = [float(row["mean_delta_removal_minus_full"]) for row in rows]
    colors = ["#c44e52" if value >= 0 else "#4c78a8" for value in means]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.bar(range(len(labels)), means, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(f"Mean delta {metric} (removal minus full)")
    ax.set_title("C-CRP component ablation across full-scale domains")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths: list[str] = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"fig_component_ablation_four_domain.{suffix}"
        fig.savefig(path, dpi=200)
        paths.append(str(path))
    plt.close(fig)
    return paths


def aggregate_component_ablation(
    *,
    package_root: str | Path,
    output_dir: str | Path,
    domains: tuple[str, ...] = DEFAULT_DOMAINS,
    metric: str = "NDCG@10",
    expected_events: int = 10000,
    expected_candidates_per_event: int = 101,
    tie_epsilon: float = 1e-12,
    skip_plot: bool = False,
) -> dict[str, Any]:
    root = Path(package_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    warnings: list[str] = []
    input_paths: dict[str, str] = {}
    by_domain: dict[str, dict[str, dict[str, str]]] = {}

    if metric not in FULL_METRICS:
        failures.append(f"unsupported_metric:{metric}")
    for domain in domains:
        package = root / f"ccrp_ablation_{domain}"
        summary_path = package / "component_ablation_summary.csv"
        audit_path = package / "phase2_5_component_ablation_package_audit.json"
        provenance_path = package / "component_ablation_provenance.json"
        for label, path in {
            f"{domain}:component_summary": summary_path,
            f"{domain}:package_audit": audit_path,
            f"{domain}:component_provenance": provenance_path,
        }.items():
            input_paths[label] = str(path)
            if not path.exists() or path.stat().st_size <= 0:
                failures.append(f"missing_input:{label}:{path}")
        if failures:
            continue
        audit = _read_json(audit_path)
        if audit.get("ok") is not True or audit.get("paper_claim_ready") is not True:
            failures.append(f"package_audit_not_ready:{domain}")
        rows = _read_csv(summary_path)
        domain_rows = {str(row.get("ablation", "")).strip(): row for row in rows}
        by_domain[domain] = domain_rows
        for ablation in DEFAULT_ABLATIONS:
            if ablation not in domain_rows:
                failures.append(f"missing_ablation:{domain}:{ablation}")
        for ablation, row in domain_rows.items():
            count = _event_count(row)
            if count != expected_events:
                failures.append(f"bad_event_count:{domain}:{ablation}:{count}!={expected_events}")
            if _as_int(row.get("expected_candidates_per_event", expected_candidates_per_event)) != expected_candidates_per_event:
                failures.append(f"bad_expected_candidates:{domain}:{ablation}")
            if _as_float(row.get("score_coverage_rate")) != 1.0:
                failures.append(f"score_coverage_not_one:{domain}:{ablation}:{row.get('score_coverage_rate')}")
            if not _as_bool(row.get("audit_ok")):
                failures.append(f"audit_not_ok:{domain}:{ablation}")
            if not _as_bool(row.get("degeneracy_audit_ok")):
                failures.append(f"degeneracy_not_ok:{domain}:{ablation}")
            for m in FULL_METRICS:
                value = _as_float(row.get(m))
                if not math.isfinite(value):
                    failures.append(f"nonfinite_metric:{domain}:{ablation}:{m}:{row.get(m)}")

    delta_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    if not failures:
        for domain in domains:
            full = by_domain[domain]["full"]
            for ablation in DEFAULT_ABLATIONS:
                row = by_domain[domain][ablation]
                for m in FULL_METRICS:
                    full_value = _as_float(full[m])
                    removal_value = _as_float(row[m])
                    delta = removal_value - full_value
                    delta_rows.append(
                        {
                            "domain": domain,
                            "ablation": ablation,
                            "metric": m,
                            "delta_convention": "removal_minus_full",
                            "full_value": full_value,
                            "removal_value": removal_value,
                            "delta_removal_minus_full": delta,
                            "event_count": _event_count(row),
                            "score_coverage_rate": _as_float(row.get("score_coverage_rate")),
                            "audit_ok": _as_bool(row.get("audit_ok")),
                            "degeneracy_audit_ok": _as_bool(row.get("degeneracy_audit_ok")),
                            "tie_epsilon": tie_epsilon,
                        }
                    )
        for ablation in DEFAULT_ABLATIONS:
            if ablation == "full":
                continue
            for m in FULL_METRICS:
                values = [
                    float(row["delta_removal_minus_full"])
                    for row in delta_rows
                    if row["ablation"] == ablation and row["metric"] == m
                ]
                nonworse = sum(value >= -tie_epsilon for value in values)
                better = sum(value > tie_epsilon for value in values)
                worse = sum(value < -tie_epsilon for value in values)
                component_rows.append(
                    {
                        "ablation": ablation,
                        "metric": m,
                        "delta_convention": "removal_minus_full",
                        "domain_count": len(values),
                        "mean_delta_removal_minus_full": statistics.fmean(values),
                        "median_delta_removal_minus_full": statistics.median(values),
                        "min_delta_removal_minus_full": min(values),
                        "max_delta_removal_minus_full": max(values),
                        "nonworse_domain_count": nonworse,
                        "better_domain_count": better,
                        "worse_domain_count": worse,
                        "tie_epsilon": tie_epsilon,
                        "classification": _classification(values, eps=tie_epsilon),
                    }
                )

    delta_path = out / "component_ablation_four_domain_delta_rows.csv"
    component_path = out / "component_ablation_four_domain_component_summary.csv"
    _write_csv(delta_path, delta_rows)
    _write_csv(component_path, component_rows)
    figure_paths = [] if skip_plot else _plot_component_summary(
        [row for row in component_rows if row.get("metric") == metric], out, metric=metric
    )
    input_sha256 = {name: sha256_file(path) for name, path in input_paths.items() if Path(path).exists()}
    provenance = {
        "artifact_class": "paper_critical_component_ablation_cross_domain_diagnostic",
        "status_label": "paper_critical_component_ablation_cross_domain_ready" if not failures else "blocked",
        "ok": not failures,
        "paper_claim_ready": not failures,
        "failures": failures,
        "warnings": warnings,
        "domains": list(domains),
        "expected_domains": list(DEFAULT_DOMAINS),
        "expected_ablations": list(DEFAULT_ABLATIONS),
        "required_metrics": list(FULL_METRICS),
        "primary_metric": metric,
        "delta_convention": "removal_minus_full",
        "tie_epsilon": tie_epsilon,
        "expected_events": expected_events,
        "expected_candidates_per_event": expected_candidates_per_event,
        "git_commit": _git_commit(),
        "input_paths": input_paths,
        "input_sha256": input_sha256,
        "outputs": {
            "delta_rows_csv": str(delta_path),
            "component_summary_csv": str(component_path),
            "figure_paths": figure_paths,
        },
        "table_eligibility": "supplementary_diagnostic_only",
        "claim_limits": [
            "This is not main-table SOTA evidence.",
            "Delta is removal_minus_full; positive deltas mean removal is nonworse or better.",
            "Neutral or positive removals must be reported as weak, redundant, or harmful component evidence.",
            "No claim is made that every C-CRP component is necessary.",
        ],
    }
    _write_json(out / "component_ablation_four_domain_provenance.json", provenance)
    md_lines = [
        "# Four-Domain C-CRP Component Ablation Diagnostic",
        "",
        f"- OK: `{provenance['ok']}`",
        f"- Table eligibility: `{provenance['table_eligibility']}`",
        f"- Delta convention: `{provenance['delta_convention']}`",
        f"- Tie epsilon: `{tie_epsilon}`",
        "",
        "Positive deltas mean the component removal was nonworse or better than full C-CRP.",
        "",
        "| ablation | metric | mean delta | nonworse domains | classification |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in component_rows:
        if row["metric"] == metric:
            md_lines.append(
                "| {ablation} | {metric} | {mean_delta_removal_minus_full:.12g} | {nonworse_domain_count} | {classification} |".format(
                    **row
                )
            )
    if failures:
        md_lines.extend(["", "## Failures", *[f"- `{failure}`" for failure in failures]])
    (out / "component_ablation_four_domain_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    args = parse_args()
    domains = tuple(args.domain or DEFAULT_DOMAINS)
    provenance = aggregate_component_ablation(
        package_root=args.package_root,
        output_dir=args.output_dir,
        domains=domains,
        metric=args.metric,
        expected_events=args.expected_events,
        expected_candidates_per_event=args.expected_candidates_per_event,
        tie_epsilon=args.tie_epsilon,
        skip_plot=args.skip_plot,
    )
    print(
        json.dumps(
            {"ok": provenance["ok"], "paper_claim_ready": provenance["paper_claim_ready"], "failures": provenance["failures"]},
            indent=2,
        )
    )
    if not provenance["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
