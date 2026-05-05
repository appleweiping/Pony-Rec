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


DOMAIN_SUFFIX = {
    "beauty": "full973",
    "books": "small500",
    "electronics": "small500",
    "movies": "small500",
}

DEFAULT_SRPD_BEST = {
    "beauty": "srpd_v2",
    "books": "srpd_v2",
    "electronics": "srpd_v5",
    "movies": "srpd_v4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired Week8 significance tests for direct, structured-risk, "
            "SRPD-best, classical external baselines, and the LLM2Rec-style "
            "Qwen3 same-backbone baseline."
        )
    )
    parser.add_argument(
        "--week77_root",
        default="~/projects/uncertainty-llm4rec/export/week7_7_four_domain_final",
        help="Week7.7 four-domain export root containing direct/structured/SRPD prediction folders.",
    )
    parser.add_argument(
        "--external_summary_glob",
        default="outputs/*/tables/same_candidate_external_baseline_summary.csv",
        help="Glob over same-candidate external baseline summaries.",
    )
    parser.add_argument("--domains", default="beauty,books,electronics,movies")
    parser.add_argument(
        "--output_dir",
        default="outputs/summary/week8_llm2rec_style_qwen3_stat_tests",
    )
    parser.add_argument(
        "--baselines",
        default="direct,structured_risk,llm2rec_style_qwen3_sasrec",
        help="Comma-separated methods used as paired baselines when present.",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--bootstrap_iters", type=int, default=2000)
    parser.add_argument("--permutation_iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_records(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_json(path, lines=True)


def _week77_method_paths(week77_root: Path, domain: str) -> dict[str, Path]:
    suffix = DOMAIN_SUFFIX[domain]
    structured_exp = week77_root / f"{domain}_qwen3_local_structured_risk_{suffix}"
    srpd_variant = DEFAULT_SRPD_BEST[domain]
    srpd_exp = week77_root / f"{domain}_qwen3_rank_{srpd_variant}_{suffix}"
    return {
        "direct": structured_exp / "predictions" / "rank_predictions.jsonl",
        "structured_risk": structured_exp / "reranked" / "rank_reranked.jsonl",
        f"srpd_best_{srpd_variant}": srpd_exp / "predictions" / "rank_predictions.jsonl",
    }


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
    week77_root = Path(args.week77_root).expanduser()
    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    baselines = tuple(item.strip() for item in args.baselines.split(",") if item.strip())
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    external_paths = _external_method_paths(args.external_summary_glob)
    all_significance: list[pd.DataFrame] = []
    all_main_tables: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []

    config = {
        "week77_root": str(week77_root),
        "external_summary_glob": args.external_summary_glob,
        "domains": domains,
        "baselines": baselines,
        "k": args.k,
        "bootstrap_iters": args.bootstrap_iters,
        "permutation_iters": args.permutation_iters,
        "seed": args.seed,
        "alpha": args.alpha,
    }
    _write_json(output_dir / "run_config.json", config)

    for domain in domains:
        method_paths = _week77_method_paths(week77_root, domain)
        method_paths.update(external_paths.get(domain, {}))
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
        main_table_df = build_main_table_with_ci(method_frames, significance_df, k=args.k)
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
