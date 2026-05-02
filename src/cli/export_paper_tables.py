from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper-ready markdown tables from aggregated CSV metrics.")
    parser.add_argument("--aggregate_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--allow_smoke", action="store_true")
    return parser.parse_args()


def _markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    with Path(args.aggregate_csv).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = ["metadata.dataset", "metadata.domain", "metadata.seed", "metadata.method", "metadata.config_hash"]
    if rows and not args.allow_smoke:
        for idx, row in enumerate(rows):
            missing = [col for col in required if not row.get(col)]
            if missing:
                raise ValueError(f"Refusing paper export: row {idx} missing required metadata {missing}.")
            if row.get("metadata.backend_type") == "mock":
                raise ValueError("Refusing paper export: mock backend outputs require --allow_smoke.")
            if row.get("metadata.run_type") == "smoke":
                raise ValueError("Refusing paper export: smoke outputs require --allow_smoke.")
            if str(row.get("metadata.is_paper_result")).lower() not in {"true", "1"}:
                raise ValueError("Refusing paper export: is_paper_result must be true unless --allow_smoke is set.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_specs = {
        "main_results.md": ["path", "ranking.HR@1", "ranking.NDCG@10", "ranking.MRR@10"],
        "calibration.md": ["path", "calibration.ece", "calibration.mce", "calibration.brier"],
        "exposure.md": ["path", "exposure.head_exposure_share", "exposure.tail_exposure_share", "exposure.long_tail_coverage"],
    }
    for filename, columns in table_specs.items():
        existing = [col for col in columns if not rows or col in rows[0]]
        if not existing:
            existing = list(rows[0].keys())[:6] if rows else []
        (output_dir / filename).write_text(_markdown_table(rows, existing), encoding="utf-8")
    print(f"[export_paper_tables] saved={output_dir}")


if __name__ == "__main__":
    main()
