from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.eval.calibration_metrics import compute_calibration_metrics, expected_calibration_error


class CalibrationMetricsTest(unittest.TestCase):
    def test_ece_is_zero_for_perfect_bins(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        ece, mce, bins = expected_calibration_error(y_true, y_prob, n_bins=2)
        self.assertAlmostEqual(ece, 0.0)
        self.assertAlmostEqual(mce, 0.0)
        self.assertEqual(sum(item.count for item in bins), 4)

    def test_compute_metrics_adds_correctness(self) -> None:
        df = pd.DataFrame(
            {
                "recommend": ["yes", "no"],
                "label": [1, 1],
                "confidence": [0.9, 0.2],
            }
        )
        metrics = compute_calibration_metrics(df)
        self.assertEqual(metrics["num_samples"], 2)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
