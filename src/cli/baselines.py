from __future__ import annotations

import argparse
from pathlib import Path

from src.baselines.standard import bm25_text_rank, popularity_rank, random_rank
from src.data.protocol import read_jsonl, write_jsonl
from src.utils.research_artifacts import utc_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic sanity baselines on candidate files.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--method", choices=["random", "popularity", "bm25"], default="popularity")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for sample in read_jsonl(args.input_path):
        candidates = [str(x) for x in sample.get("candidate_item_ids", [])]
        if args.method == "random":
            ranking = random_rank(candidates, args.seed, str(sample.get("user_id", "")))
        elif args.method == "bm25":
            history_text = " ".join(str(x) for x in sample.get("history_item_ids", []))
            ranking = bm25_text_rank(candidates, [str(x) for x in sample.get("candidate_texts", [])], history_text)
        else:
            ranking = popularity_rank(candidates, [int(x) for x in sample.get("candidate_popularity_counts", [])])
        predicted = ranking[0] if ranking else ""
        target = str(sample.get("target_item_id", ""))
        rows.append(
            {
                "dataset": Path(args.input_path).parent.name,
                "domain": Path(args.input_path).parent.name,
                "split": sample.get("dataset_split", "test"),
                "seed": args.seed,
                "method": args.method,
                "backend": "none",
                "model": args.method,
                "prompt_template_id": "none",
                "timestamp": utc_timestamp(),
                "user_id": sample.get("user_id"),
                "history_length": sample.get("history_length"),
                "history_length_bucket": sample.get("history_length_bucket"),
                "candidate_item_ids": candidates,
                "candidate_popularity_counts": sample.get("candidate_popularity_counts", []),
                "candidate_popularity_buckets": sample.get("candidate_popularity_buckets", []),
                "predicted_ranking": ranking,
                "predicted_item_id": predicted,
                "target_item_id": target,
                "correctness": predicted == target,
                "raw_confidence": 0.5,
                "calibrated_confidence": 0.5,
                "uncertainty_score": 0.5,
                "uncertainty_estimator_name": "baseline_constant",
                "is_valid": True,
            }
        )
    write_jsonl(rows, args.output_path)
    print(f"[baselines] method={args.method} saved={args.output_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
