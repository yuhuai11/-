from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dads_crnn.evaluate_low_fpr import (
    clopper_pearson,
    evaluate_low_fpr,
    threshold_at_target_fpr,
)


def prediction_rows(split: str, probabilities: list[float]) -> pd.DataFrame:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    return pd.DataFrame(
        {
            "path": [f"/tmp/{split}_{index}.wav" for index in range(8)],
            "sha256": [f"{split}_{index:02d}" for index in range(8)],
            "label": labels,
            "uav_source": ["", "", "", "", "u1", "u1", "u2", "u2"],
            "background_source": ["b1", "b1", "b2", "b2", "", "", "", ""],
            "condition": ["background_only"] * 4 + ["uav_only"] * 4,
            "ood_split": [split] * 8,
            "calibrated_probability": probabilities,
        }
    )


class LowFprEvaluationTests(unittest.TestCase):
    def test_threshold_respects_conservative_budget(self) -> None:
        result = threshold_at_target_fpr(np.arange(100, dtype=np.float64) / 100.0, 0.01)
        self.assertEqual(result["allowed_false_positives"], 1)
        self.assertLessEqual(result["actual_false_positives"], 1)
        self.assertLessEqual(result["empirical_fpr"], 0.01)

    def test_zero_success_interval_is_finite(self) -> None:
        interval = clopper_pearson(0, 100)
        self.assertEqual(interval["low"], 0.0)
        self.assertGreater(interval["high"], 0.0)
        self.assertLess(interval["high"], 0.1)

    def test_end_to_end_uses_tune_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tune_path = root / "tune.csv"
            holdout_path = root / "holdout.csv"
            prediction_rows(
                "tune", [0.05, 0.10, 0.15, 0.20, 0.70, 0.75, 0.80, 0.85]
            ).to_csv(tune_path, index=False)
            prediction_rows(
                "holdout", [0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90]
            ).to_csv(holdout_path, index=False)
            result = evaluate_low_fpr(
                tune_path,
                holdout_path,
                root / "out",
                experiment="test",
                target_fprs=[0.25],
                bootstrap_samples=20,
                bootstrap_seed=42,
            )
            point = result["operating_points"][0]
            self.assertLessEqual(point["tune"]["fpr"], 0.25)
            self.assertEqual(point["holdout"]["recall"], 1.0)
            self.assertTrue((root / "out" / "metrics.json").is_file())
            self.assertTrue((root / "out" / "operating_points.csv").is_file())


if __name__ == "__main__":
    unittest.main()
