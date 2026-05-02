from __future__ import annotations

import argparse

from src.baselines.recbole_adapter import export_recbole_atomic
from src.data.protocol import load_yaml_config, protocol_config_from_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export processed data to RecBole atomic format.")
    parser.add_argument("--dataset_config", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = protocol_config_from_dict(load_yaml_config(args.dataset_config))
    result = export_recbole_atomic(processed_dir=cfg.processed_dir, output_dir=args.output_dir, dataset_name=cfg.dataset)
    print(result)


if __name__ == "__main__":
    main()
