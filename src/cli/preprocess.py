from __future__ import annotations

import argparse

from src.data.protocol import load_yaml_config, preprocess_from_config, protocol_config_from_dict
from src.utils.manifest import build_manifest, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild processed recommendation data from raw files.")
    parser.add_argument("--config", required=True, help="YAML dataset config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = protocol_config_from_dict(load_yaml_config(args.config))
    stats = preprocess_from_config(cfg)
    write_manifest(
        cfg.processed_dir / "manifest.json",
        build_manifest(
            config=load_yaml_config(args.config),
            dataset=cfg.dataset,
            domain=cfg.domain,
            raw_data_paths=[str(cfg.raw_interactions_path), str(cfg.raw_items_path)],
            processed_data_paths=[str(cfg.processed_dir)],
            method="preprocess",
            backend="none",
            model="none",
            prompt_template="none",
            seed=cfg.seed,
            candidate_size=None,
            mock_data_used="smoke" in cfg.dataset.lower(),
        ),
    )
    print(f"[preprocess] saved={cfg.processed_dir}")
    print(
        "[preprocess] users={filtered_users} items={filtered_items} interactions={filtered_interactions}".format(
            **stats
        )
    )


if __name__ == "__main__":
    main()
