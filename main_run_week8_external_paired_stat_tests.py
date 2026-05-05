from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.eval.statistical_tests import (
    build_event_metric_frame,
    build_main_table_with_ci,
    compare_method_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired tests among completed external same-candidate baselines "
            "only. This is useful for protocols that must not be mixed with "
            "Week7.7 direct/structured-risk/SRPD event files."
        )
    )
    parser.add_argument(
        "--external_summary_glob",
        required=True,
        help="Glob over completed same-candidate external baseline summaries.",
    )
    parser.add_argument("--domains", default="books,electronics,movies")
    parser.add_argument(
        "--baselines",
        default="sasrec,gru4rec,bert4rec,lightgcn,llmemb_style_qwen3_sasrec,rlmrec_style_qwen3_graphcl,irllrec_style_qwen3_intent,setrec_style_qwen3_identifier,llmesr_style_qwen3_sasrec",
        help="Comma-separated methods used as paired-test anchors when present.",
    )
    parser.add_argument("--output_dir", default="outputs/summary/week8_external_paired_stat_tests")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--bootstrap_iters", type=int, default=2000)
    parser.add_argument("--permutation_iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _load_records(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    return pd.read_json(path, lines=True)


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _external_method_paths(summary_glob: str) -> dict[str, dict[str, Path]]:
    paths: dict[str, dict[str, Path]] = {}
    for path_text in sorted(glob.glob(summary_glob)):
        summary_path = Path(path_text).expanduser()
        if not summary_path.exists():
            continue
        for row in _read_csv_rows(summary_path):
            if row.get("status_label") != "same_schema_external_baseline":
                continue
            if row.get("artifact_class") != "completed_result":
                continue
            domain = row.get("domain", "").strip()
            method = row.get("baseline_name", "").strip()
            prediction_path = row.get("prediction_path", "").strip()
            if not domain or not method or not prediction_path:
                continue
            paths.setdefault(domain, {})[method] = Path(prediction_path).expanduser()
    return paths


def _resolve_path(path: Path, *, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root / path


def _build_method_frames(
    method_paths: dict[str, Path],
    *,
    repo_root: Path,
    k: int,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    frames: dict[str, pd.DataFrame] = {}
    manifest_rows: list[dict[str, Any]] = []
    for method, raw_path in sorted(method_paths.items()):
        path = _resolve_path(raw_path, repo_root=repo_root)
        exists = path.exists()
        row = {
            "method": method,
            "path": str(path),
            "exists": exists,
            "rows": "",
            "status": "ready" if exists else "missing",
        }
        if exists:
            raw_df = _load_records(path)
            frames[method] = build_event_metric_frame(raw_df, method=method, k=k)
            row["rows"] = len(raw_df)
        manifest_rows.append(row)
    return frames, manifest_rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["domain", "method", "path", "exists", "rows", "status"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    domains = _parse_csv_list(args.domains)
    baselines = tuple(_parse_csv_list(args.baselines))
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    external_paths = _external_method_paths(args.external_summary_glob)
    all_significance: list[pd.DataFrame] = []
    all_main_tables: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []

    _write_json(
        output_dir / "run_config.json",
        {
            "external_summary_glob": args.external_summary_glob,
            "domains": domains,
            "baselines": baselines,
            "k": args.k,
            "bootstrap_iters": args.bootstrap_iters,
            "permutation_iters": args.permutation_iters,
            "seed": args.seed,
            "alpha": args.alpha,
            "note": "External-only paired tests; no Week7.7 direct/structured/SRPD paths are loaded.",
        },
    )

    for domain in domains:
        method_paths = external_paths.get(domain, {})
        method_frames, domain_manifest = _build_method_frames(method_paths, repo_root=repo_root, k=args.k)
        for row in domain_manifest:
            row["domain"] = domain
        manifest_rows.extend(domain_manifest)

        present_baselines = tuple(name for name in baselines if name in method_frames)
        if not present_baselines or len(method_frames) < 2:
            print(f"[{domain}] Skipping stats; present_methods={sorted(method_frames)}")
            continue

        significance_df = compare_method_frames(
            method_frames,
            baselines=present_baselines,
            k=args.k,
            n_bootstrap=args.bootstrap_iters,
            n_permutations=args.permutation_iters,
            random_state=args.seed,
            alpha=args.alpha,
        )
        direct_name = present_baselines[0]
        main_table_df = build_main_table_with_ci(method_frames, significance_df, direct_name=direct_name, k=args.k)
        significance_df.insert(0, "domain", domain)
        main_table_df.insert(0, "domain", domain)

        significance_path = output_dir / f"{domain}_significance_tests.csv"
        main_table_path = output_dir / f"{domain}_main_table_with_ci.csv"
        significance_df.to_csv(significance_path, index=False)
        main_table_df.to_csv(main_table_path, index=False)
        all_significance.append(significance_df)
        all_main_tables.append(main_table_df)
        print(f"[{domain}] Saved {significance_path}")
        print(f"[{domain}] Saved {main_table_path}")

    _write_manifest(output_dir / "input_manifest.csv", manifest_rows)
    if all_significance:
        pd.concat(all_significance, ignore_index=True).to_csv(
            output_dir / "all_domains_significance_tests.csv",
            index=False,
        )
    if all_main_tables:
        pd.concat(all_main_tables, ignore_index=True).to_csv(
            output_dir / "all_domains_main_table_with_ci.csv",
            index=False,
        )
    print(f"Saved manifest: {output_dir / 'input_manifest.csv'}")


if __name__ == "__main__":
    main()
