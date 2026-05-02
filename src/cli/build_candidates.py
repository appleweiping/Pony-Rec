from __future__ import annotations

import argparse

from src.data.protocol import (
    build_candidates_from_processed,
    load_yaml_config,
    protocol_config_from_dict,
)
from src.utils.manifest import build_manifest, write_manifest


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
    negative_count = args.negative_count if args.negative_count is not None else cfg.negative_count
    strategy = args.strategy or cfg.negative_strategy
    counts = build_candidates_from_processed(
        cfg.processed_dir,
        seed=cfg.seed,
        negative_count=negative_count,
        strategy=strategy,
    )
    write_manifest(
        cfg.processed_dir / "candidates_manifest.json",
        build_manifest(
            config=raw_cfg,
            dataset=cfg.dataset,
            domain=cfg.domain,
            raw_data_paths=[str(cfg.raw_interactions_path), str(cfg.raw_items_path)],
            processed_data_paths=[str(cfg.processed_dir)],
            method="build_candidates",
            backend="none",
            model="none",
            prompt_template="none",
            seed=cfg.seed,
            candidate_size=negative_count + 1,
            mock_data_used="smoke" in cfg.dataset.lower(),
        ),
    )
    print(f"[build_candidates] saved={cfg.processed_dir}")
    print(f"[build_candidates] counts={counts}")


if __name__ == "__main__":
    main()
