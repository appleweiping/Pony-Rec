from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOMAINS = ("sports", "toys", "home", "tools")
FULL_METRICS = ("MRR", "HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate package-audited uncertainty observation/motivation results "
            "across full-scale same-candidate domains."
        )
    )
    parser.add_argument("--package_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--gate_method", default="ccrp_v3")
    parser.add_argument("--primary_metric", default="NDCG@10")
    parser.add_argument("--support_metric", action="append", default=[])
    parser.add_argument("--min_domains_for_stratification", type=int, default=3)
    parser.add_argument("--expected_events", type=int, default=10000)
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
        parsed = float(value)
    except Exception:
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def _plot_domain_deltas(rows: list[dict[str, Any]], output_dir: Path, *, primary_metric: str, gate_method: str) -> list[str]:
    plot_rows = [row for row in rows if row["method"] == gate_method and row["metric"] == primary_metric]
    if not plot_rows:
        return []
    import matplotlib  # type: ignore

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore

    domains = [str(row["domain"]) for row in plot_rows]
    deltas = [float(row["delta_high_minus_low"]) for row in plot_rows]
    colors = ["#4c78a8" if delta < 0 else "#c44e52" for delta in deltas]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.bar(domains, deltas, color=colors)
    ax.set_ylabel(f"High-low {primary_metric}")
    ax.set_title("C-CRP uncertainty-bin reliability diagnostic")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths: list[str] = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"fig_uncertainty_observation_four_domain.{suffix}"
        fig.savefig(path, dpi=200)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _plot_bin_trend(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    primary_metric: str,
    gate_method: str,
) -> list[str]:
    plot_rows = [
        row
        for row in rows
        if row["method"] == gate_method
        and str(row.get("uncertainty_bin")) != "ALL"
        and primary_metric in row
    ]
    if not plot_rows:
        return []
    import matplotlib  # type: ignore

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore

    by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in plot_rows:
        by_domain.setdefault(str(row["domain"]), []).append(row)
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    mean_by_bin: dict[int, list[float]] = {}
    for domain in sorted(by_domain):
        domain_rows = sorted(by_domain[domain], key=lambda row: _as_int(row.get("uncertainty_bin_index")))
        xs = [_as_int(row.get("uncertainty_bin_index")) + 1 for row in domain_rows]
        ys = [_as_float(row.get(primary_metric)) for row in domain_rows]
        ax.plot(xs, ys, marker="o", linewidth=1.5, alpha=0.65, label=domain)
        for x, y in zip(xs, ys):
            if math.isfinite(y):
                mean_by_bin.setdefault(x, []).append(y)
    mean_xs = sorted(mean_by_bin)
    mean_ys = [sum(mean_by_bin[x]) / len(mean_by_bin[x]) for x in mean_xs]
    if mean_xs:
        ax.plot(mean_xs, mean_ys, color="#111111", marker="s", linewidth=2.4, label="mean")
    ax.set_xlabel("C-CRP uncertainty quintile")
    ax.set_ylabel(primary_metric)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_title("Reliability across C-CRP uncertainty bins")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    paths: list[str] = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"fig_uncertainty_observation_four_domain_trend.{suffix}"
        fig.savefig(path, dpi=200)
        paths.append(str(path))
    plt.close(fig)
    return paths


def aggregate_uncertainty_observation(
    *,
    package_root: str | Path,
    output_dir: str | Path,
    domains: tuple[str, ...] = DEFAULT_DOMAINS,
    gate_method: str = "ccrp_v3",
    primary_metric: str = "NDCG@10",
    support_metrics: tuple[str, ...] = ("MRR", "HR@10"),
    min_domains_for_stratification: int = 3,
    expected_events: int = 10000,
    skip_plot: bool = False,
) -> dict[str, Any]:
    root = Path(package_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    warnings: list[str] = []
    exact_four_domain_set = tuple(domains) == DEFAULT_DOMAINS
    if not exact_four_domain_set:
        warnings.append(
            "non_default_domain_set_diagnostic_only:"
            f"{','.join(domains)}!= {','.join(DEFAULT_DOMAINS)}"
        )
    input_paths: dict[str, str] = {}
    all_summary_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []

    metrics = tuple(dict.fromkeys((primary_metric, *support_metrics, *FULL_METRICS)))
    for metric in (primary_metric, *support_metrics):
        if metric not in FULL_METRICS:
            failures.append(f"unsupported_metric:{metric}")

    for domain in domains:
        package = root / f"observation_{domain}"
        summary_path = package / "observation_summary.csv"
        audit_path = package / "phase2_5_observation_motivation_package_audit.json"
        provenance_path = package / "observation_provenance.json"
        for label, path in {
            f"{domain}:observation_summary": summary_path,
            f"{domain}:package_audit": audit_path,
            f"{domain}:observation_provenance": provenance_path,
        }.items():
            input_paths[label] = str(path)
            if not path.exists() or path.stat().st_size <= 0:
                failures.append(f"missing_input:{label}:{path}")
        if failures:
            continue
        audit = _read_json(audit_path)
        if audit.get("ok") is not True or audit.get("paper_claim_ready") is not True:
            failures.append(f"package_audit_not_ready:{domain}")
        provenance = _read_json(provenance_path)
        uncertainty_summary = provenance.get("uncertainty_summary", {})
        if _as_int(uncertainty_summary.get("event_count")) != expected_events:
            failures.append(f"bad_uncertainty_event_count:{domain}:{uncertainty_summary.get('event_count')}")
        rows = _read_csv(summary_path)
        all_rows = [row for row in rows if str(row.get("uncertainty_bin")) == "ALL"]
        if not any(str(row.get("method")) == gate_method for row in all_rows):
            failures.append(f"missing_gate_method_all_row:{domain}:{gate_method}")
        for row in all_rows:
            if _as_int(row.get("n_events")) != expected_events:
                failures.append(f"bad_all_event_count:{domain}:{row.get('method')}:{row.get('n_events')}")
        for row in rows:
            row_out = dict(row)
            row_out["domain"] = domain
            all_summary_rows.append(row_out)
        methods = sorted({str(row.get("method")) for row in rows if str(row.get("uncertainty_bin")) != "ALL"})
        for method in methods:
            method_rows = [
                row for row in rows if str(row.get("method")) == method and str(row.get("uncertainty_bin")) != "ALL"
            ]
            if not method_rows:
                continue
            method_rows.sort(key=lambda row: _as_int(row.get("uncertainty_bin_index")))
            low = method_rows[0]
            high = method_rows[-1]
            for metric in metrics:
                if metric not in low or metric not in high:
                    continue
                low_value = _as_float(low.get(metric))
                high_value = _as_float(high.get(metric))
                if not math.isfinite(low_value) or not math.isfinite(high_value):
                    failures.append(f"nonfinite_metric:{domain}:{method}:{metric}")
                    continue
                delta_rows.append(
                    {
                        "domain": domain,
                        "method": method,
                        "metric": metric,
                        "low_bin": low.get("uncertainty_bin"),
                        "high_bin": high.get("uncertainty_bin"),
                        "low_value": low_value,
                        "high_value": high_value,
                        "delta_high_minus_low": high_value - low_value,
                        "degrades_with_uncertainty": bool(high_value < low_value),
                    }
                )

    primary_degraded_domains = {
        row["domain"]
        for row in delta_rows
        if row["method"] == gate_method and row["metric"] == primary_metric and row["degrades_with_uncertainty"]
    }
    support_gate: dict[str, Any] = {}
    for metric in support_metrics:
        degraded = {
            row["domain"]
            for row in delta_rows
            if row["method"] == gate_method and row["metric"] == metric and row["degrades_with_uncertainty"]
        }
        support_gate[metric] = {
            "degraded_domain_count": len(degraded),
            "degraded_domains": sorted(degraded),
            "pass": len(degraded) >= min_domains_for_stratification,
        }
    primary_pass = len(primary_degraded_domains) >= min_domains_for_stratification
    support_pass = any(row.get("pass") is True for row in support_gate.values())
    claim_gate_pass = primary_pass and support_pass and exact_four_domain_set and not failures
    claim_status = "uncertainty_stratifies_reliability" if claim_gate_pass else "mixed_diagnostic_pattern"

    summary_path = out / "observation_four_domain_summary_rows.csv"
    delta_path = out / "observation_four_domain_domain_deltas.csv"
    _write_csv(summary_path, all_summary_rows)
    _write_csv(delta_path, delta_rows)
    figure_paths = []
    if not skip_plot:
        figure_paths.extend(
            _plot_domain_deltas(
                delta_rows,
                out,
                primary_metric=primary_metric,
                gate_method=gate_method,
            )
        )
        figure_paths.extend(
            _plot_bin_trend(
                all_summary_rows,
                out,
                primary_metric=primary_metric,
                gate_method=gate_method,
            )
        )
    input_sha256 = {name: sha256_file(path) for name, path in input_paths.items() if Path(path).exists()}
    provenance = {
        "artifact_class": "paper_critical_observation_motivation_cross_domain",
        "status_label": (
            "paper_critical_observation_cross_domain_ready"
            if not failures and exact_four_domain_set
            else "diagnostic_subset_only" if not failures else "blocked"
        ),
        "ok": not failures,
        "paper_claim_ready": not failures and exact_four_domain_set,
        "failures": failures,
        "warnings": warnings,
        "domains": list(domains),
        "required_domains": list(DEFAULT_DOMAINS),
        "exact_four_domain_set": exact_four_domain_set,
        "gate_method": gate_method,
        "primary_metric": primary_metric,
        "support_metrics": list(support_metrics),
        "min_domains_for_stratification": min_domains_for_stratification,
        "primary_gate": {
            "degraded_domain_count": len(primary_degraded_domains),
            "degraded_domains": sorted(primary_degraded_domains),
            "pass": primary_pass,
        },
        "support_gate": support_gate,
        "claim_gate_pass": claim_gate_pass,
        "claim_status": claim_status,
        "table_eligibility": "motivation_only_not_main_table_sota",
        "git_commit": _git_commit(),
        "input_paths": input_paths,
        "input_sha256": input_sha256,
        "outputs": {
            "summary_rows_csv": str(summary_path),
            "domain_deltas_csv": str(delta_path),
            "figure_paths": figure_paths,
        },
        "claim_limits": [
            "Bins are defined by C-CRP event uncertainty.",
            "This is descriptive motivation evidence, not a causal or statistically significant effect claim.",
            "Representative baselines are not exhaustive baseline evidence.",
            "No hyperparameter or method choice may be changed after inspecting this test-bin analysis.",
            "If claim_gate_pass is false, paper wording must be downgraded to mixed diagnostic pattern.",
        ],
    }
    _write_json(out / "observation_four_domain_provenance.json", provenance)
    md_lines = [
        "# Four-Domain Uncertainty Observation Diagnostic",
        "",
        f"- OK: `{provenance['ok']}`",
        f"- Claim status: `{claim_status}`",
        f"- Claim gate pass: `{claim_gate_pass}`",
        f"- Table eligibility: `{provenance['table_eligibility']}`",
        f"- Gate method: `{gate_method}`",
        f"- Primary metric: `{primary_metric}`",
        "",
        "| metric | degraded domains | pass |",
        "| --- | ---: | --- |",
        f"| {primary_metric} | {len(primary_degraded_domains)} | {primary_pass} |",
    ]
    for metric, row in support_gate.items():
        md_lines.append(f"| {metric} | {row['degraded_domain_count']} | {row['pass']} |")
    if failures:
        md_lines.extend(["", "## Failures", *[f"- `{failure}`" for failure in failures]])
    (out / "observation_four_domain_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    args = parse_args()
    support_metrics = tuple(args.support_metric or ("MRR", "HR@10"))
    provenance = aggregate_uncertainty_observation(
        package_root=args.package_root,
        output_dir=args.output_dir,
        domains=tuple(args.domain or DEFAULT_DOMAINS),
        gate_method=args.gate_method,
        primary_metric=args.primary_metric,
        support_metrics=support_metrics,
        min_domains_for_stratification=args.min_domains_for_stratification,
        expected_events=args.expected_events,
        skip_plot=args.skip_plot,
    )
    print(
        json.dumps(
            {
                "ok": provenance["ok"],
                "paper_claim_ready": provenance["paper_claim_ready"],
                "claim_status": provenance["claim_status"],
                "claim_gate_pass": provenance["claim_gate_pass"],
                "failures": provenance["failures"],
            },
            indent=2,
        )
    )
    if not provenance["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
