from __future__ import annotations

import unittest

from src.llm.parser import parse_candidate_ranking_response, parse_pointwise_response
from src.uncertainty.verbalized_confidence import normalize_confidence_value


class ConfidenceParserTest(unittest.TestCase):
    def test_normalizes_percent_confidence(self) -> None:
        self.assertAlmostEqual(normalize_confidence_value("83%"), 0.83)
        self.assertAlmostEqual(normalize_confidence_value(83), 0.83)

    def test_pointwise_json_parser(self) -> None:
        parsed = parse_pointwise_response('{"recommend": "yes", "confidence": "high", "reason": "match"}')
        self.assertEqual(parsed["recommend"], "yes")
        self.assertAlmostEqual(parsed["confidence"], 0.8)

    def test_ranking_rejects_out_of_candidate_items(self) -> None:
        parsed = parse_candidate_ranking_response(
            '{"ranked_item_ids": ["A", "Z"], "confidence": 0.7}',
            allowed_item_ids=["A", "B"],
            topk=2,
        )
        self.assertFalse(parsed["parse_success"])
        self.assertTrue(parsed["contains_out_of_candidate_item"])


if __name__ == "__main__":
    unittest.main()
