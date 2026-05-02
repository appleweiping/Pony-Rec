from __future__ import annotations

import argparse

from src.data.protocol import (
    build_candidates_from_processed,
    load_yaml_config,
    protocol_config_from_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic target-plus-negative candidate sets.")
    parser.add_argument("--config", required=True, help="YAML dataset/candidate config.")
    parser.add_argument("--negative_count", type=int, default=None)
    parser.add_argument("--strategy", default=None, choices=["uniform", "popularity_stratified"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_cfg = load_yaml_config(args.config)
    cfg = protocol_config_from_dict(raw_cfg)
    counts = build_candidates_from_processed(
        cfg.processed_dir,
        seed=cfg.seed,
        negative_count=args.negative_count if args.negative_count is not None else cfg.negative_count,
        strategy=args.strategy or cfg.negative_strategy,
    )
    print(f"[build_candidates] saved={cfg.processed_dir}")
    print(f"[build_candidates] counts={counts}")


if __name__ == "__main__":
    main()
