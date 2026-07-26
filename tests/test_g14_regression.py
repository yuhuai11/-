from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dads_crnn.evaluate_g14_regression import _load_prediction_csv, _metric_delta


class G14RegressionTests(unittest.TestCase):
    def test_prediction_csv_must_align_by_label_and_hash(self) -> None:
        manifest = pd.DataFrame(
            {"label": [0, 1], "sha256": ["a" * 64, "b" * 64]}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            manifest.assign(raw_probability=[0.1, 0.9]).to_csv(path, index=False)
            observed = _load_prediction_csv(path, manifest, "raw_probability")
            np.testing.assert_allclose(observed, [0.1, 0.9])

            manifest.iloc[::-1].assign(raw_probability=[0.1, 0.9]).to_csv(
                path, index=False
            )
            with self.assertRaisesRegex(ValueError, "labels do not align"):
                _load_prediction_csv(path, manifest, "raw_probability")

    def test_metric_delta_is_candidate_minus_baseline(self) -> None:
        baseline = {"f1": 0.8, "auc": 0.9, "recall": 0.7, "specificity": 0.95}
        candidate = {"f1": 0.79, "auc": 0.92, "recall": 0.75, "specificity": 0.90}
        delta = _metric_delta(candidate, baseline)
        self.assertAlmostEqual(delta["f1"], -0.01)
        self.assertAlmostEqual(delta["auc"], 0.02)
        self.assertAlmostEqual(delta["recall"], 0.05)
        self.assertAlmostEqual(delta["specificity"], -0.05)


if __name__ == "__main__":
    unittest.main()
