#!/usr/bin/env python3
"""Compute a Popularity sanity baseline on the same-candidate sampled-ranking task.

This is a *trivial* non-personalized reference for the main same-candidate
NDCG@10 ranking table (Sports / Toys / Home / Tools, 10k users x 101 candidates).
For every test user it ranks the 101 candidates by descending item popularity,
where popularity is the interaction count of the item in the TRAINING
interactions file ONLY (no test/valid leakage). Ties are broken stably by
ascending candidate id, matching the table's
``stable_candidate_order_after_score_descending`` policy (deterministic, seed
free, candidate-order independent).

Metrics, computed over all test users, reproduce exactly the formulas used by
``src/eval/ranking_task_metrics.py`` for the official-baseline imports:

    rank          1-indexed position of the positive item after sorting
    HR@10  = 1  if rank <= 10 else 0                       (mean over users)
    NDCG@10= 1 / log2(rank + 1) if rank <= 10 else 0       (mean over users)
    MRR    = 1 / rank                                       (mean over users)

CPU only. No GPU, no torch, no model. Pure stdlib.

Usage (run on the server, CPU):

    python3 scripts/analysis/compute_popularity_baseline_same_candidate.py \
        --task-root outputs/baselines/external_tasks \
        --domains sports:sports_large10000_100neg_test_same_candidate \
                  toys:toys_large10000_100neg_test_same_candidate \
                  home:home_large10000_100neg_test_same_candidate \
                  tools:tools_large10000_100neg_test_same_candidate \
        --out outputs/summary/popularity_sanity_baseline_same_candidate.json

Each ``--domains`` entry is ``<display_name>:<task_dir_name>``. The task
directory must contain ``ranking_test.jsonl`` and ``train_interactions.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter


def _norm(value) -> str:
    """Normalize an id to a stripped string (matches importer ``_text``)."""
    if value is None:
        return ""
    return str(value).strip()


def load_train_popularity(train_path: str) -> Counter:
    """Popularity = number of TRAIN interaction rows per item_id.

    The train file is CSV with header ``user_id,item_id,timestamp,sequence_index``.
    We count one unit of popularity per interaction row (every interaction the
    item received in training), which is the standard most-popular reference.
    """
    pop: Counter = Counter()
    with open(train_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "item_id" not in reader.fieldnames:
            raise ValueError(
                f"train file {train_path} is missing an 'item_id' column; "
                f"got header={reader.fieldnames}"
            )
        for row in reader:
            item_id = _norm(row.get("item_id"))
            if item_id:
                pop[item_id] += 1
    return pop


def rank_positive_by_popularity(
    candidate_ids: list[str],
    positive_id: str,
    popularity: Counter,
) -> int:
    """Return the 1-indexed rank of the positive item.

    Candidates are sorted by (descending popularity, ascending candidate id).
    A candidate absent from the train file has popularity 0. The tie-break by
    candidate id is stable and deterministic, matching the table's
    candidate-order tie-break policy (independent of input ordering and seeds).
    """
    # sort key: higher popularity first, then ascending id as a stable tie-break
    ranked = sorted(candidate_ids, key=lambda cid: (-popularity.get(cid, 0), cid))
    # positive must be one of the candidates (verified per row by the caller)
    return ranked.index(positive_id) + 1


def evaluate_domain(ranking_path: str, train_path: str, k: int = 10) -> dict:
    popularity = load_train_popularity(train_path)

    n_users = 0
    sum_hr = 0.0
    sum_ndcg = 0.0
    sum_mrr = 0.0
    missing_positive = 0

    with open(ranking_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            candidate_ids = [_norm(c) for c in sample.get("candidate_item_ids", [])]
            candidate_ids = [c for c in candidate_ids if c]
            positive_id = _norm(sample.get("positive_item_id"))

            if not candidate_ids or not positive_id:
                # malformed row: skip but record it implicitly via n_users gap
                continue
            if positive_id not in candidate_ids:
                # positive is supposed to be one of the candidates; if not, the
                # positive can never be retrieved -> rank beyond the list.
                missing_positive += 1
                n_users += 1
                # rank = len + 1 (never in top-k), contributes 0 to HR/NDCG and
                # 1/(len+1) to MRR, mirroring build_ranking_eval_frame fallback.
                rank = len(candidate_ids) + 1
            else:
                rank = rank_positive_by_popularity(candidate_ids, positive_id, popularity)
                n_users += 1

            if rank <= k:
                sum_hr += 1.0
                sum_ndcg += 1.0 / math.log2(rank + 1)
            sum_mrr += 1.0 / rank

    if n_users == 0:
        raise ValueError(f"no usable test rows found in {ranking_path}")

    return {
        "n_users": n_users,
        "HR@%d" % k: sum_hr / n_users,
        "NDCG@%d" % k: sum_ndcg / n_users,
        "MRR": sum_mrr / n_users,
        "missing_positive_in_candidates": missing_positive,
        "n_train_items_with_popularity": len(popularity),
    }


def parse_domain_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"--domains entry must be '<name>:<task_dir>', got {spec!r}"
        )
    name, task_dir = spec.split(":", 1)
    name = name.strip()
    task_dir = task_dir.strip()
    if not name or not task_dir:
        raise argparse.ArgumentTypeError(f"bad --domains entry {spec!r}")
    return name, task_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-root",
        default="outputs/baselines/external_tasks",
        help="Root directory holding the per-domain same-candidate task dirs.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        type=parse_domain_spec,
        required=True,
        help="One or more '<display_name>:<task_dir_name>' entries.",
    )
    parser.add_argument("--k", type=int, default=10, help="Top-k cutoff (default 10).")
    parser.add_argument(
        "--ranking-file",
        default="ranking_test.jsonl",
        help="Name of the per-user ranking jsonl inside each task dir.",
    )
    parser.add_argument(
        "--train-file",
        default="train_interactions.csv",
        help="Name of the train interactions csv inside each task dir.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the aggregated JSON result.",
    )
    args = parser.parse_args(argv)

    results: dict[str, dict] = {}
    for display_name, task_dir in args.domains:
        ranking_path = os.path.join(args.task_root, task_dir, args.ranking_file)
        train_path = os.path.join(args.task_root, task_dir, args.train_file)
        if not os.path.isfile(ranking_path):
            raise FileNotFoundError(ranking_path)
        if not os.path.isfile(train_path):
            raise FileNotFoundError(train_path)
        metrics = evaluate_domain(ranking_path, train_path, k=args.k)
        metrics["task_dir"] = task_dir
        results[display_name] = metrics
        print(
            f"[{display_name}] n_users={metrics['n_users']} "
            f"NDCG@{args.k}={metrics['NDCG@%d' % args.k]:.4f} "
            f"HR@{args.k}={metrics['HR@%d' % args.k]:.4f} "
            f"MRR={metrics['MRR']:.4f} "
            f"(missing_pos={metrics['missing_positive_in_candidates']}, "
            f"train_items={metrics['n_train_items_with_popularity']})",
            flush=True,
        )

    payload = {
        "task": "popularity_sanity_baseline_same_candidate",
        "k": args.k,
        "tie_break": "stable_candidate_order_after_score_descending (ascending candidate id)",
        "popularity_source": "train_interactions.csv interaction-count per item_id (train only, no test/val leakage)",
        "metric_formulas": {
            "HR@k": "1 if positive_rank <= k else 0 (mean over users)",
            "NDCG@k": "1/log2(rank+1) if positive_rank <= k else 0 (mean over users)",
            "MRR": "1/positive_rank (mean over users)",
        },
        "domains": results,
    }

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"wrote {args.out}", flush=True)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
