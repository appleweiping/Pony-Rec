from __future__ import annotations

import argparse

from src.analysis.echo_chamber import echo_chamber_report
from src.data.protocol import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure popularity-confidence and exposure concentration effects.")
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = echo_chamber_report(read_jsonl(args.predictions_path))
    write_json(report, args.output_path)
    print(f"[echo_chamber_analysis] saved={args.output_path}")


if __name__ == "__main__":
    main()
