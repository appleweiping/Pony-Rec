from __future__ import annotations

import argparse

from src.data.protocol import read_jsonl, write_jsonl
from src.methods.uncertainty_reranking import RerankConfig, rerank_candidate_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply uncertainty-aware reranking, abstention, and truncation.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--lambda_penalty", type=float, default=0.5)
    parser.add_argument("--popularity_penalty", type=float, default=0.0)
    parser.add_argument("--exploration_bonus", type=float, default=0.0)
    parser.add_argument("--abstention_threshold", type=float, default=None)
    parser.add_argument("--truncation_threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RerankConfig(
        lambda_penalty=args.lambda_penalty,
        popularity_penalty=args.popularity_penalty,
        exploration_bonus=args.exploration_bonus,
        abstention_threshold=args.abstention_threshold,
        truncation_threshold=args.truncation_threshold,
    )
    rows = [rerank_candidate_record(row, cfg) for row in read_jsonl(args.input_path)]
    for row in rows:
        row["correctness"] = bool(row.get("predicted_item_id") == row.get("target_item_id")) if not row.get("abstained") else False
    write_jsonl(rows, args.output_path)
    print(f"[rerank] saved={args.output_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
