"""Audit c99 reprocess + DeepSeek pilot inputs for data vs invalid-output correlation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.diagnostics.c99_data_quality_audit import run_full_audit, write_cleaning_exemplars


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", type=Path, default=Path("data/processed"))
    p.add_argument(
        "--reprocess_dir",
        type=Path,
        default=Path("outputs/reprocessed_processed_source_100u_c99_seed42"),
    )
    p.add_argument(
        "--pilot_root",
        type=Path,
        default=Path("outputs/pilots/deepseek_v4_flash_processed_100u_c99_seed42"),
    )
    p.add_argument(
        "--domains",
        nargs="*",
        default=[
            "amazon_beauty",
            "amazon_books",
            "amazon_electronics",
            "amazon_movies",
        ],
    )
    p.add_argument("--splits", nargs="*", default=["valid", "test"])
    p.add_argument("--prompt_id", default="listwise_ranking_v1")
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/diagnostics/c99_data_quality_audit"),
    )
    p.add_argument(
        "--exemplar_dir",
        type=Path,
        default=Path("outputs/diagnostics/c99_prompt_cleaning_candidates"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run_full_audit(
        domains=list(args.domains),
        splits=list(args.splits),
        data_root=args.data_root,
        reprocess_dir=args.reprocess_dir,
        pilot_root=args.pilot_root,
        prompt_id=args.prompt_id,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "audit_report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[audit] wrote {out_json}")

    for unstable in ("amazon_books", "amazon_movies"):
        if unstable in args.domains:
            for sp in args.splits:
                write_cleaning_exemplars(
                    domain=unstable,
                    split=sp,
                    reprocess_dir=args.reprocess_dir,
                    data_root=args.data_root,
                    out_path=args.exemplar_dir / f"{unstable}_{sp}_exemplars.json",
                    n=5,
                )
    print(f"[audit] exemplars under {args.exemplar_dir}")


if __name__ == "__main__":
    main()
