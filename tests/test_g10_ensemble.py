from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dads_crnn.evaluate_g10_ensemble import source_quantile_threshold


class G10EnsembleTests(unittest.TestCase):
    def test_source_quantile_threshold_uses_frozen_source_rank(self) -> None:
        rows = []
        for source, values in {
            "a": [0.1, 0.2, 0.3, 0.4],
            "b": [0.2, 0.3, 0.4, 0.5],
            "c": [0.3, 0.4, 0.5, 0.6],
            "d": [0.4, 0.5, 0.6, 0.7],
        }.items():
            rows.extend(
                {
                    "label": 0,
                    "background_source": source,
                    "calibrated_probability": value,
                }
                for value in values
            )
        result = source_quantile_threshold(pd.DataFrame(rows), 0.25, 0.60)
        self.assertEqual(result["background_sources"], 4)
        self.assertEqual(result["order_statistic_rank"], 3)
        self.assertGreaterEqual(result["threshold"], result["pooled_threshold"])
        self.assertAlmostEqual(result["source_threshold"], np.nextafter(0.5, np.inf))

    def test_source_quantile_threshold_rejects_missing_groups(self) -> None:
        rows = pd.DataFrame(
            {
                "label": [0, 0],
                "background_source": ["", "a"],
                "calibrated_probability": [0.1, 0.2],
            }
        )
        with self.assertRaises(ValueError):
            source_quantile_threshold(rows, 0.1, 0.5)


if __name__ == "__main__":
    unittest.main()
