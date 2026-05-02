from __future__ import annotations

import argparse

from src.data.protocol import load_yaml_config, preprocess_from_config, protocol_config_from_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild processed recommendation data from raw files.")
    parser.add_argument("--config", required=True, help="YAML dataset config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = protocol_config_from_dict(load_yaml_config(args.config))
    stats = preprocess_from_config(cfg)
    print(f"[preprocess] saved={cfg.processed_dir}")
    print(
        "[preprocess] users={filtered_users} items={filtered_items} interactions={filtered_interactions}".format(
            **stats
        )
    )


if __name__ == "__main__":
    main()
