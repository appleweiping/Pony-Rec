from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate metric JSON files into a CSV table.")
    parser.add_argument("--metrics_glob", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def _flatten(prefix: str, data: dict, out: dict) -> None:
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            _flatten(name, value, out)
        else:
            out[name] = value


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(Path().glob(args.metrics_glob)):
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {"path": str(path)}
        _flatten("", data, row)
        rows.append(row)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        print(f"[aggregate] no files matched {args.metrics_glob}")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[aggregate] saved={output} rows={len(rows)}")


if __name__ == "__main__":
    main()
