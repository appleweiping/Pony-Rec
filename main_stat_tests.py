from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eval.statistical_tests import (
    build_event_metric_frame,
    build_main_table_with_ci,
    compare_method_frames,
)


def parse_method_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected METHOD=PATH.")
    method, path = value.split("=", 1)
    method = method.strip()
    if not method:
        raise argparse.ArgumentTypeError("Method name cannot be empty.")
    return method, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_paths",
        nargs="+",
        type=parse_method_path,
        required=True,
        help="Paired ranking records as METHOD=PATH. Use method names such as direct or structured_risk.",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/summary")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--bootstrap_iters", type=int, default=2000)
    parser.add_argument("--permutation_iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--baselines",
        type=str,
        default="direct,structured_risk",
        help="Comma-separated methods used as paired baselines.",
    )
    return parser.parse_args()


def load_records(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_json(path, lines=True)


def main() -> None:
    args = parse_args()
    method_frames = {}
    for method, path in args.input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing method records for {method}: {path}")
        raw_df = load_records(path)
        method_frames[method] = build_event_metric_frame(raw_df, method=method, k=args.k)

    baselines = tuple(item.strip() for item in args.baselines.split(",") if item.strip())
    significance_df = compare_method_frames(
        method_frames,
        baselines=baselines,
        k=args.k,
        n_bootstrap=args.bootstrap_iters,
        n_permutations=args.permutation_iters,
        random_state=args.seed,
        alpha=args.alpha,
    )
    main_table_df = build_main_table_with_ci(method_frames, significance_df, k=args.k)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    significance_path = output_dir / "significance_tests.csv"
    main_table_path = output_dir / "main_table_with_ci.csv"
    significance_df.to_csv(significance_path, index=False)
    main_table_df.to_csv(main_table_path, index=False)
    print(f"Saved significance tests to: {significance_path}")
    print(f"Saved main table with CI to: {main_table_path}")


if __name__ == "__main__":
    main()
