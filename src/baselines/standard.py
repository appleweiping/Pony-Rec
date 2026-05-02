from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any


def random_rank(candidates: list[str], seed: int, key: str) -> list[str]:
    rng = random.Random(f"{seed}:{key}")
    out = list(candidates)
    rng.shuffle(out)
    return out


def popularity_rank(candidates: list[str], popularity_counts: list[int]) -> list[str]:
    return [
        item_id
        for item_id, _ in sorted(
            zip(candidates, popularity_counts),
            key=lambda pair: (-int(pair[1]), pair[0]),
        )
    ]


def itemknn_rank(
    candidates: list[str],
    history_item_ids: list[str],
    cooccurrence: dict[str, Counter[str]],
) -> list[str]:
    def score(item_id: str) -> tuple[int, str]:
        value = sum(cooccurrence.get(hist, Counter()).get(item_id, 0) for hist in history_item_ids)
        return (-value, item_id)

    return sorted(candidates, key=score)


def build_cooccurrence(train_rows: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    cooc: dict[str, Counter[str]] = defaultdict(Counter)
    for row in train_rows:
        items = [str(x) for x in row.get("train_item_ids", [])]
        for item in items:
            cooc[item].update(other for other in items if other != item)
    return cooc


def bm25_text_rank(candidates: list[str], candidate_texts: list[str], history_text: str) -> list[str]:
    query = set(history_text.lower().split())
    scores = []
    for item_id, text in zip(candidates, candidate_texts):
        tokens = text.lower().split()
        overlap = sum(1 for token in tokens if token in query)
        scores.append((item_id, overlap))
    return [item_id for item_id, _ in sorted(scores, key=lambda pair: (-pair[1], pair[0]))]
