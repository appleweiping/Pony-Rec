from __future__ import annotations

import unittest

from src.shadow.ccrp import compute_ccrp_record
from src.shadow.parser import parse_shadow_response
from src.shadow.schema import get_shadow_variant


class ShadowParserTest(unittest.TestCase):
    def test_shadow_v1_is_main_ccrp_method(self) -> None:
        spec = get_shadow_variant("shadow_v1")
        self.assertEqual(spec.method_name, "C-CRP")
        self.assertEqual(spec.paper_role, "main_method")

    def test_parse_shadow_json(self) -> None:
        parsed = parse_shadow_response(
            '{"relevance_probability": 0.7, "evidence_support": 0.8, "counterevidence_strength": 0.1, "reason": "fit"}',
            variant="shadow_v1",
        )
        self.assertTrue(parsed["parse_success"])
        self.assertAlmostEqual(parsed["relevance_probability"], 0.7)

    def test_ccrp_not_one_minus_probability(self) -> None:
        scored = compute_ccrp_record(
            {
                "relevance_probability": 0.8,
                "calibrated_relevance_probability": 0.7,
                "evidence_support": 0.2,
                "counterevidence_strength": 0.1,
            }
        )
        self.assertIn("ccrp_boundary_uncertainty", scored)
        self.assertNotAlmostEqual(scored["ccrp_uncertainty"], 0.3)


if __name__ == "__main__":
    unittest.main()
