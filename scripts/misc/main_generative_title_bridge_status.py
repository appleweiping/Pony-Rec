from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BRIDGE_STATUS_COLUMNS = [
    "generation_method",
    "catalog_mapping_method",
    "catalog_hit_rate",
    "recall_at_10_after_mapping",
    "ndcg_at_10_after_mapping",
    "out_of_catalog_rate",
    "hallucination_rate",
    "unsupported_confident_generation_rate",
    "accept_rate",
    "revision_rate",
    "fallback_rate",
    "status_label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/summary/generative_title_bridge_status.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=BRIDGE_STATUS_COLUMNS).to_csv(out_path, index=False)
    print(f"Saved generative title bridge status template to: {out_path}")


if __name__ == "__main__":
    main()
