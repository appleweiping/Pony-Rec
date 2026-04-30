from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.uncertainty.baseline_reliability_proxy import (
    DEFAULT_BASELINE_RELIABILITY_PROXIES,
    build_proxy_audit,
)
from src.utils.exp_io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_reliability/week7_9_manifest.yaml",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/summary/baseline_reliability_proxy_audit.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if config_path.exists():
        config = load_yaml(config_path)
        rows = config.get("baselines", [])
    else:
        rows = DEFAULT_BASELINE_RELIABILITY_PROXIES
    if not isinstance(rows, list):
        raise ValueError("Baseline reliability config must contain a `baselines` list.")

    audit_df = build_proxy_audit(rows)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(out_path, index=False)

    proxy_only_count = int((audit_df["status_label"] == "proxy_only").sum()) if not audit_df.empty else 0
    completed_count = int((audit_df["status_label"] == "completed_result").sum()) if not audit_df.empty else 0
    print(f"Saved baseline reliability proxy audit to: {out_path}")
    print(f"completed_result={completed_count}, proxy_only={proxy_only_count}")


if __name__ == "__main__":
    main()
