from __future__ import annotations

import unittest

import pandas as pd

from src.eval.candidate_protocol_audit import audit_candidate_protocol


class CandidateProtocolAuditTest(unittest.TestCase):
    def test_one_positive_hr_recall_equivalence_flag(self) -> None:
        valid = pd.DataFrame(
            [
                {
                    "source_event_id": "e1",
                    "user_id": "u1",
                    "candidate_item_ids": ["a", "b", "c"],
                    "candidate_titles": ["A", "B", "C"],
                    "candidate_popularity_groups": ["head", "tail", "tail"],
                    "candidate_labels": [0, 1, 0],
                    "positive_item_id": "b",
                }
            ]
        )
        test = pd.DataFrame(
            [
                {
                    "source_event_id": "e2",
                    "user_id": "u2",
                    "candidate_item_ids": ["d", "e", "f"],
                    "candidate_titles": ["D", "E", "F"],
                    "candidate_popularity_groups": ["head", "mid", "tail"],
                    "candidate_labels": [1, 0, 0],
                    "positive_item_id": "d",
                }
            ]
        )
        audit = audit_candidate_protocol({"valid": valid, "test": test}, domain="toy")
        self.assertTrue(audit["one_positive_setting"].all())
        self.assertTrue(audit["hr_recall_equivalent_flag"].all())
        self.assertEqual(int(audit["user_overlap_valid_test"].iloc[0]), 0)


if __name__ == "__main__":
    unittest.main()
