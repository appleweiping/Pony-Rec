from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.cli.infer import _run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a local model or LoRA adapter through the common inference path.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    import asyncio

    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    infer_args = argparse.Namespace(
        config=args.config,
        split=args.split,
        input_path=None,
        output_path=None,
        max_samples=args.max_samples,
        resume=False,
    )
    asyncio.run(_run(cfg, infer_args))


if __name__ == "__main__":
    main()
