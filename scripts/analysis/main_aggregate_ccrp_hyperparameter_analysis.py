from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOMAINS = ("sports", "toys", "home", "tools")
DEFAULT_CONTROLS = ("eta", "weight_grid_label")
FULL_METRICS = ("MRR", "HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate package-audited C-CRP hyperparameter sensitivity/stability "
            "curves across the four new full-scale same-candidate domains."
        )
    )
    parser.add_argument("--package_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--metric", default="NDCG@10")
    parser.add_argument("--control", action="append", default=[])
    parser.add_argument("--expected_events", type=int, default=10000)
    parser.add_argument("--expected_candidates_per_event", type=int, default=101)
    parser.add_argument("--relative_drop_tolerance", type=float, default=0.05)
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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else float("nan")


def _classification(rows: list[dict[str, Any]], *, tolerance: float) -> str:
    if not rows:
        return "missing_stability_rows"
    stable = sum(1 for row in rows if row.get("stable_within_tolerance") is True)
    max_drop = max(_as_float(row.get("relative_drop_from_test_best")) for row in rows)
    if stable == len(rows) and max_drop <= tolerance:
        if all(row.get("best_value_match") is True for row in rows):
            return "same_valid_test_best_all_domains"
        return "stable_with_domain_specific_best_values"
    return "mixed_or_unstable_hyperparameter_sensitivity"


def _file_manifest(paths: list[Path]) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        manifest[path.name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return manifest


def _plot_stability(rows: list[dict[str, Any]], output_dir: Path, *, metric: str) -> list[str]:
    if not rows:
        return []
    import matplotlib  # type: ignore

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore

    labels = [f"{row['domain']}:{row['control']}" for row in rows]
    drops = [_as_float(row.get("relative_drop_from_test_best")) for row in rows]
    colors = ["#4c78a8" if _as_bool(row.get("stable_within_tolerance")) else "#c44e52" for row in rows]
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.axhline(0.05, color="#333333", linewidth=0.9, linestyle="--", label="5% tolerance")
    ax.bar(range(len(labels)), drops, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel(f"Relative drop from test-best {metric}")
    ax.set_title("C-CRP hyperparameter stability across domains")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    paths: list[str] = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"fig_hyperparameter_four_domain_stability.{suffix}"
        fig.savefig(path, dpi=200)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _control_sort_key(value: str) -> tuple[int, float, str]:
    try:
        return (0, float(value), value)
    except Exception:
        return (1, 0.0, value)


def _plot_mean_curves(rows: list[dict[str, Any]], output_dir: Path, *, metric: str, controls: tuple[str, ...]) -> list[str]:
    if not rows:
        return []
    import matplotlib  # type: ignore

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore

    paths: list[str] = []
    for control in controls:
        control_rows = [row for row in rows if row.get("control") == control and row.get("metric_name") == metric]
        if not control_rows:
            continue
        values = sorted({str(row.get("control_value", "")) for row in control_rows}, key=_control_sort_key)
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        for split in ("valid", "test"):
            means = []
            for value in values:
                metrics = [
                    _as_float(row.get("metric_value"))
                    for row in control_rows
                    if row.get("split") == split and str(row.get("control_value", "")) == value
                ]
                means.append(_mean(metrics))
            if control in {"eta", "confidence_weight"}:
                xs = [float(value) for value in values]
                ax.plot(xs, means, marker="o", linewidth=1.8, label=split)
                ax.set_xlabel(control)
            else:
                xs = list(range(len(values)))
                ax.plot(xs, means, marker="o", linewidth=1.8, label=split)
                ax.set_xticks(xs)
                ax.set_xticklabels(values, rotation=25, ha="right")
                ax.set_xlabel("C-CRP weight triple")
        ax.set_ylabel(f"Mean {metric} across domains")
        ax.set_title(f"C-CRP four-domain sensitivity: {control}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        stem = "fig_hyperparameter_four_domain_eta" if control == "eta" else "fig_hyperparameter_four_domain_weight"
        for suffix in ("png", "pdf"):
            path = output_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=200)
            paths.append(str(path))
        plt.close(fig)
    return paths


def aggregate_hyperparameter_analysis(
    *,
    package_root: str | Path,
    output_dir: str | Path,
    domains: tuple[str, ...] = DEFAULT_DOMAINS,
    controls: tuple[str, ...] = DEFAULT_CONTROLS,
    metric: str = "NDCG@10",
    expected_events: int = 10000,
    expected_candidates_per_event: int = 101,
    relative_drop_tolerance: float = 0.05,
    skip_plot: bool = False,
) -> dict[str, Any]:
    root = Path(package_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    warnings: list[str] = []
    exact_four_domain_set = tuple(domains) == DEFAULT_DOMAINS
    if not exact_four_domain_set:
        warnings.append(f"non_default_domain_set_diagnostic_only:{','.join(domains)}")
    if metric not in FULL_METRICS:
        failures.append(f"unsupported_metric:{metric}")
    unsupported_controls = sorted(set(controls) - set(DEFAULT_CONTROLS))
    if unsupported_controls:
        failures.append(f"unsupported_controls:{','.join(unsupported_controls)}")

    input_paths: dict[str, str] = {}
    all_curve_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []

    expected_key_count = expected_events * expected_candidates_per_event if expected_events > 0 else 0
    for domain in domains:
        package = root / f"ccrp_hyperparameter_{domain}"
        summary_path = package / "ccrp_hyperparameter_curve_summary.csv"
        provenance_path = package / "ccrp_hyperparameter_curve_provenance.json"
        audit_path = package / "phase2_5_hyperparameter_package_audit.json"
        for label, path in {
            f"{domain}:hyperparameter_summary": summary_path,
            f"{domain}:hyperparameter_provenance": provenance_path,
            f"{domain}:package_audit": audit_path,
        }.items():
            input_paths[label] = str(path)
            if not path.exists() or path.stat().st_size <= 0:
                failures.append(f"missing_input:{label}:{path}")
        if failures:
            continue

        audit = _read_json(audit_path)
        if audit.get("ok") is not True or audit.get("paper_claim_ready") is not True:
            failures.append(f"package_audit_not_ready:{domain}")
        module_audit = audit.get("module_audit", {})
        if isinstance(module_audit, dict):
            if module_audit.get("status_label") != "paper_critical_hyperparameter_curve_ready":
                failures.append(f"package_audit_bad_status:{domain}:{module_audit.get('status_label')}")
        provenance = _read_json(provenance_path)
        if provenance.get("artifact_class") != "paper_critical_hyperparameter_analysis":
            failures.append(f"bad_artifact_class:{domain}:{provenance.get('artifact_class')}")
        if provenance.get("status_label") != "paper_critical_hyperparameter_curve_ready":
            failures.append(f"bad_status_label:{domain}:{provenance.get('status_label')}")
        if provenance.get("paper_claim_scope") != "valid_and_test_stability_curve_candidate":
            failures.append(f"bad_paper_claim_scope:{domain}:{provenance.get('paper_claim_scope')}")
        if provenance.get("reporting_mode") != "valid_and_test":
            failures.append(f"not_valid_and_test:{domain}")
        if provenance.get("test_not_used_for_selection") is not True:
            failures.append(f"test_not_used_false:{domain}")
        if str(provenance.get("metric", "")) != metric:
            failures.append(f"metric_mismatch:{domain}:{provenance.get('metric')}!={metric}")
        for control in controls:
            if control not in [str(item) for item in provenance.get("controls", [])]:
                failures.append(f"missing_control_in_provenance:{domain}:{control}")
        source = provenance.get("sweep_source_provenance", {})
        if not isinstance(source, dict) or source.get("test_not_used_for_selection") is not True:
            failures.append(f"bad_sweep_source_provenance:{domain}")
        row_counts = source.get("row_counts", {}) if isinstance(source, dict) else {}
        if expected_key_count > 0 and isinstance(row_counts, dict):
            for key in ("valid_signal_rows", "test_signal_rows", "valid_candidate_rows", "test_candidate_rows"):
                if _as_int(row_counts.get(key)) != expected_key_count:
                    failures.append(f"bad_source_row_count:{domain}:{key}:{row_counts.get(key)}")
            for key in ("valid_ranking_events", "test_ranking_events"):
                if _as_int(row_counts.get(key)) != expected_events:
                    failures.append(f"bad_source_event_count:{domain}:{key}:{row_counts.get(key)}")

        rows = _read_csv(summary_path)
        for row in rows:
            row_out: dict[str, Any] = dict(row)
            row_out["domain"] = domain
            control = str(row.get("control", "")).strip()
            split = str(row.get("split", "")).strip()
            if control in controls:
                if split not in {"valid", "test"}:
                    failures.append(f"bad_split:{domain}:{control}:{split}")
                if str(row.get("row_kind", "")).strip() != "main_control":
                    failures.append(f"bad_row_kind:{domain}:{control}:{row.get('row_kind')}")
                if str(row.get("score_mode", "")).strip() != "full":
                    failures.append(f"bad_score_mode:{domain}:{control}:{row.get('score_mode')}")
                if str(row.get("ablation", "")).strip() != "full":
                    failures.append(f"bad_ablation:{domain}:{control}:{row.get('ablation')}")
                if not _as_bool(row.get("audit_ok")):
                    failures.append(f"audit_not_ok:{domain}:{control}:{row.get('control_value')}")
                if not _as_bool(row.get("degeneracy_audit_ok")):
                    failures.append(f"degeneracy_not_ok:{domain}:{control}:{row.get('control_value')}")
                if abs(_as_float(row.get("score_coverage_rate")) - 1.0) > 1e-12:
                    failures.append(f"coverage_not_one:{domain}:{control}:{row.get('control_value')}")
                if expected_key_count > 0 and _as_int(row.get("candidate_key_count")) != expected_key_count:
                    failures.append(f"candidate_key_count_mismatch:{domain}:{control}:{row.get('candidate_key_count')}")
                value = _as_float(row.get("metric_value"))
                if not math.isfinite(value) or not (0.0 <= value <= 1.0):
                    failures.append(f"bad_metric_value:{domain}:{control}:{row.get('control_value')}:{row.get('metric_value')}")
            all_curve_rows.append(row_out)

        report_by_control = {
            str(row.get("control", "")): row
            for row in provenance.get("stability_report", [])
            if isinstance(row, dict)
        }
        for control in controls:
            report = report_by_control.get(control)
            if not report:
                failures.append(f"missing_stability_report:{domain}:{control}")
                continue
            if report.get("metric") != metric:
                failures.append(f"stability_metric_mismatch:{domain}:{control}:{report.get('metric')}")
            if report.get("stable_within_tolerance") is not True:
                failures.append(f"stability_not_ready:{domain}:{control}:{report.get('reason')}")
            drop = _as_float(report.get("relative_drop_from_test_best"))
            if not math.isfinite(drop):
                failures.append(f"aggregate_stability_drop_nonfinite:{domain}:{control}:{report.get('relative_drop_from_test_best')}")
            elif drop > relative_drop_tolerance + 1e-12:
                failures.append(
                    f"aggregate_stability_drop_exceeds_tolerance:{domain}:{control}:{drop}>{relative_drop_tolerance}"
                )
            row_out = dict(report)
            row_out["domain"] = domain
            stability_rows.append(row_out)

    control_rows: list[dict[str, Any]] = []
    if not failures:
        for control in controls:
            rows = [row for row in stability_rows if row.get("control") == control]
            drops = [_as_float(row.get("relative_drop_from_test_best")) for row in rows]
            ranks = [_as_int(row.get("test_rank_of_valid_best")) for row in rows]
            control_rows.append(
                {
                    "control": control,
                    "metric": metric,
                    "domain_count": len(rows),
                    "stable_domain_count": sum(1 for row in rows if row.get("stable_within_tolerance") is True),
                    "best_value_match_domain_count": sum(1 for row in rows if row.get("best_value_match") is True),
                    "mean_relative_drop_from_test_best": _mean(drops),
                    "max_relative_drop_from_test_best": max(drops) if drops else float("nan"),
                    "worst_test_rank_of_valid_best": max(ranks) if ranks else 0,
                    "relative_drop_tolerance": relative_drop_tolerance,
                    "classification": _classification(rows, tolerance=relative_drop_tolerance),
                }
            )

    curve_path = out / "ccrp_hyperparameter_four_domain_curve_rows.csv"
    stability_path = out / "ccrp_hyperparameter_four_domain_stability_rows.csv"
    control_path = out / "ccrp_hyperparameter_four_domain_control_summary.csv"
    summary_md_path = out / "ccrp_hyperparameter_four_domain_summary.md"
    run_config_path = out / "run_config.json"
    log_snippets_path = out / "log_snippets.md"
    _write_csv(curve_path, all_curve_rows)
    _write_csv(stability_path, stability_rows)
    _write_csv(control_path, control_rows)
    figure_paths: list[str] = []
    if not skip_plot:
        figure_paths.extend(_plot_stability(stability_rows, out, metric=metric))
        figure_paths.extend(_plot_mean_curves(all_curve_rows, out, metric=metric, controls=controls))
    md_lines = [
        "# Four-Domain C-CRP Hyperparameter Sensitivity",
        "",
        f"- OK: `{not failures}`",
        f"- Paper claim ready: `{not failures and exact_four_domain_set}`",
        "- Table eligibility: `supplementary_hyperparameter_stability_only`",
        f"- Metric: `{metric}`",
        f"- Controls: `{', '.join(controls)}`",
        "",
        "| control | stable domains | max relative drop | worst test rank | classification |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in control_rows:
        md_lines.append(
            "| {control} | {stable_domain_count}/{domain_count} | {max_relative_drop_from_test_best:.12g} | {worst_test_rank_of_valid_best} | {classification} |".format(
                **row
            )
        )
    if failures:
        md_lines.extend(["", "## Failures", *[f"- `{failure}`" for failure in failures]])
    summary_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    command = " ".join(sys.argv)
    git_commit = _git_commit()
    run_config = {
        "artifact_class": "paper_critical_hyperparameter_cross_domain_run_config",
        "domain_count": len(domains),
        "domains": list(domains),
        "controls": list(controls),
        "metric": metric,
        "expected_events": expected_events,
        "expected_candidates_per_event": expected_candidates_per_event,
        "relative_drop_tolerance": relative_drop_tolerance,
        "package_root": str(root),
        "output_dir": str(out),
        "git_commit": git_commit,
        "command": command,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_config_path, run_config)
    log_snippets_path.write_text(
        "# Four-Domain C-CRP Hyperparameter Aggregate Log Snippet\n\n"
        f"- Command: `{command}`\n"
        f"- OK: `{not failures}`\n"
        f"- Input domains: `{', '.join(domains)}`\n"
        f"- Curve rows: `{len(all_curve_rows)}`\n"
        f"- Stability rows: `{len(stability_rows)}`\n"
        f"- Control summary rows: `{len(control_rows)}`\n"
        f"- Failures: `{len(failures)}`\n",
        encoding="utf-8",
    )
    input_sha256 = {name: sha256_file(path) for name, path in input_paths.items() if Path(path).exists()}
    all_controls_stable = bool(control_rows) and not failures and all(
        row["stable_domain_count"] == row["domain_count"] == len(domains)
        and _as_float(row.get("max_relative_drop_from_test_best")) <= relative_drop_tolerance + 1e-12
        for row in control_rows
    )
    output_files = [curve_path, stability_path, control_path, summary_md_path, run_config_path, log_snippets_path]
    output_files.extend(Path(path) for path in figure_paths)
    provenance = {
        "artifact_class": "paper_critical_hyperparameter_cross_domain_sensitivity",
        "status_label": "paper_critical_hyperparameter_cross_domain_ready" if not failures else "blocked",
        "ok": not failures,
        "paper_claim_ready": not failures and exact_four_domain_set and all_controls_stable,
        "failures": failures,
        "warnings": warnings,
        "domains": list(domains),
        "expected_domains": list(DEFAULT_DOMAINS),
        "exact_four_domain_set": exact_four_domain_set,
        "controls": list(controls),
        "metric": metric,
        "expected_events": expected_events,
        "expected_candidates_per_event": expected_candidates_per_event,
        "relative_drop_tolerance": relative_drop_tolerance,
        "all_controls_stable": all_controls_stable,
        "git_commit": git_commit,
        "command": command,
        "input_paths": input_paths,
        "input_sha256": input_sha256,
        "output_manifest": _file_manifest(output_files),
        "outputs": {
            "curve_rows_csv": str(curve_path),
            "stability_rows_csv": str(stability_path),
            "control_summary_csv": str(control_path),
            "summary_md": str(summary_md_path),
            "run_config_json": str(run_config_path),
            "log_snippets_md": str(log_snippets_path),
            "figure_paths": figure_paths,
        },
        "table_eligibility": "supplementary_hyperparameter_stability_only",
        "claim_limits": [
            "This aggregate is sensitivity/stability evidence only.",
            "Validation best values are reported; test rows are reporting-only and cannot select or change the method.",
            "This is not main-table SOTA evidence and does not replace paired official-baseline comparisons.",
            "Domain-specific best values should be described as stability within tolerance, not as a universal optimum.",
        ],
    }
    _write_json(out / "ccrp_hyperparameter_four_domain_provenance.json", provenance)
    return provenance


def main() -> None:
    args = parse_args()
    provenance = aggregate_hyperparameter_analysis(
        package_root=args.package_root,
        output_dir=args.output_dir,
        domains=tuple(args.domain or DEFAULT_DOMAINS),
        controls=tuple(args.control or DEFAULT_CONTROLS),
        metric=args.metric,
        expected_events=args.expected_events,
        expected_candidates_per_event=args.expected_candidates_per_event,
        relative_drop_tolerance=args.relative_drop_tolerance,
        skip_plot=args.skip_plot,
    )
    print(
        json.dumps(
            {
                "ok": provenance["ok"],
                "paper_claim_ready": provenance["paper_claim_ready"],
                "all_controls_stable": provenance["all_controls_stable"],
                "failures": provenance["failures"],
            },
            indent=2,
        )
    )
    if not provenance["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
