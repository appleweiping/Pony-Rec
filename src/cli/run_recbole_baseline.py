from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.baselines.recbole_adapter import copy_baseline_config_to_run, run_recbole_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a configured RecBole baseline.")
    parser.add_argument("--baseline_config", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/recbole")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.baseline_config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    output_dir = Path(args.output_dir) / str(cfg.get("model", "recbole")).lower()
    copy_baseline_config_to_run(args.baseline_config, output_dir)
    result = run_recbole_baseline(baseline_config=cfg, dataset_dir=args.dataset_dir, output_dir=output_dir)
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[run_recbole_baseline] saved={output_dir / 'result.json'}")


if __name__ == "__main__":
    main()
