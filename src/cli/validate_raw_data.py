from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.protocol import load_yaml_config
from src.data.raw_validation import validate_raw_data_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw recommendation data readiness.")
    parser.add_argument("--dataset_config", required=True)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--min_post_filter_interactions", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.dataset_config)
    report = validate_raw_data_config(cfg, min_post_filter_interactions=args.min_post_filter_interactions)
    payload = report.to_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_path:
        path = Path(args.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text)
    if not report.ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
