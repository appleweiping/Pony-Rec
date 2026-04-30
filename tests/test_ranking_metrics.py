from __future__ import annotations

import unittest

import pandas as pd

from src.eval.ranking_metrics import compute_ranking_metrics


class RankingMetricsTest(unittest.TestCase):
    def test_single_positive_ranking_metrics(self) -> None:
        df = pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "a", "label": 0, "rank": 1},
                {"user_id": "u1", "item_id": "b", "label": 1, "rank": 2},
                {"user_id": "u2", "item_id": "c", "label": 1, "rank": 1},
                {"user_id": "u2", "item_id": "d", "label": 0, "rank": 2},
            ]
        )
        metrics = compute_ranking_metrics(df, k=1)
        self.assertAlmostEqual(metrics["HR@1"], 0.5)
        self.assertAlmostEqual(metrics["MRR@1"], 0.5)


if __name__ == "__main__":
    unittest.main()
