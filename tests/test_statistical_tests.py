from __future__ import annotations

import unittest

import pandas as pd

from src.eval.statistical_tests import (
    build_event_metric_frame,
    compare_method_frames,
)


class StatisticalTestsTest(unittest.TestCase):
    def test_paired_comparison_labels_winner_only_when_ci_positive(self) -> None:
        direct = pd.DataFrame(
            [
                {"source_event_id": "e1", "positive_rank": 2, "topk_item_ids": ["a"], "candidate_item_ids": ["a", "b"]},
                {"source_event_id": "e2", "positive_rank": 2, "topk_item_ids": ["c"], "candidate_item_ids": ["c", "d"]},
                {"source_event_id": "e3", "positive_rank": 2, "topk_item_ids": ["e"], "candidate_item_ids": ["e", "f"]},
            ]
        )
        ccrp = pd.DataFrame(
            [
                {"source_event_id": "e1", "positive_rank": 1, "topk_item_ids": ["b"], "candidate_item_ids": ["a", "b"]},
                {"source_event_id": "e2", "positive_rank": 1, "topk_item_ids": ["d"], "candidate_item_ids": ["c", "d"]},
                {"source_event_id": "e3", "positive_rank": 1, "topk_item_ids": ["f"], "candidate_item_ids": ["e", "f"]},
            ]
        )
        frames = {
            "direct": build_event_metric_frame(direct, method="direct", k=1),
            "ccrp": build_event_metric_frame(ccrp, method="ccrp", k=1),
        }
        result = compare_method_frames(
            frames,
            baselines=("direct",),
            k=1,
            n_bootstrap=50,
            n_permutations=50,
            random_state=7,
        )
        ndcg = result[result["metric"] == "NDCG@1"].iloc[0]
        self.assertGreater(ndcg["delta"], 0.0)
        self.assertIn(ndcg["result_label"], {"winner", "observed_best"})


if __name__ == "__main__":
    unittest.main()
