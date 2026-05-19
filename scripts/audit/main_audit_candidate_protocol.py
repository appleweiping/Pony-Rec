from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eval.candidate_protocol_audit import audit_candidate_protocol, load_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="unknown")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Optional processed domain directory. Ranking valid/test files are preferred when present.",
    )
    parser.add_argument("--valid_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--items_path", type=str, default=None)
    parser.add_argument(
        "--negative_sampling_strategy",
        type=str,
        default="sampled_candidates_unspecified",
    )
    parser.add_argument(
        "--hard_negative_popularity_groups",
        type=str,
        default="head",
        help="Comma-separated popularity bins treated as hard negatives for audit reporting.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/summary/candidate_protocol_audit.csv",
    )
    return parser.parse_args()


def _resolve_split_path(data_dir: Path, split: str) -> Path:
    candidates = [
        data_dir / f"ranking_{split}.jsonl",
        data_dir / f"{split}.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No {split} split file found under {data_dir}")


def _read_optional(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    return load_table(path)


def main() -> None:
    args = parse_args()
    valid_path = Path(args.valid_path) if args.valid_path else None
    test_path = Path(args.test_path) if args.test_path else None
    train_path = Path(args.train_path) if args.train_path else None
    items_path = Path(args.items_path) if args.items_path else None

    if args.data_dir:
        data_dir = Path(args.data_dir)
        valid_path = valid_path or _resolve_split_path(data_dir, "valid")
        test_path = test_path or _resolve_split_path(data_dir, "test")
        train_candidate = data_dir / "train.jsonl"
        items_candidate = data_dir / "items.csv"
        if train_path is None and train_candidate.exists():
            train_path = train_candidate
        if items_path is None and items_candidate.exists():
            items_path = items_candidate

    if valid_path is None and test_path is None:
        raise ValueError("Provide --data_dir or at least one of --valid_path/--test_path.")

    split_frames: dict[str, pd.DataFrame] = {}
    if valid_path is not None:
        split_frames["valid"] = load_table(valid_path)
    if test_path is not None:
        split_frames["test"] = load_table(test_path)

    train_df = load_table(train_path) if train_path is not None and train_path.exists() else _read_optional(args.train_path)
    item_catalog_size = None
    if items_path is not None and items_path.exists():
        item_catalog_size = int(len(pd.read_csv(items_path)))

    hard_groups = {
        item.strip().lower()
        for item in args.hard_negative_popularity_groups.split(",")
        if item.strip()
    }
    audit_df = audit_candidate_protocol(
        split_frames,
        domain=args.domain,
        negative_sampling_strategy=args.negative_sampling_strategy,
        hard_negative_groups=hard_groups,
        train_df=train_df,
        item_catalog_size=item_catalog_size,
    )

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(out_path, index=False)
    print(f"Saved candidate protocol audit to: {out_path}")


if __name__ == "__main__":
    main()
