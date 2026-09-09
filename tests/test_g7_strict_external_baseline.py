from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dads_crnn.evaluate_g7_strict_external_baseline import (
    _numeric_leaves,
    summarize_strict_seeds,
)
from dads_crnn.evaluate_g7_strict_multiscale import aggregate_nonoverlap


class G7StrictExternalBaselineTests(unittest.TestCase):
    def test_numeric_leaves_excludes_counts_and_thresholds(self) -> None:
        result = _numeric_leaves(
            {
                "ranking": {"roc_auc": 0.9},
                "operating": {"threshold": 0.5, "f1": 0.8, "tp": 10},
            }
        )
        self.assertEqual(result, {"ranking.roc_auc": 0.9, "operating.f1": 0.8})

    def test_three_seed_summary_uses_sample_standard_deviation(self) -> None:
        reports = {}
        for seed, value in zip((42, 43, 44), (0.7, 0.8, 0.9)):
            reports[f"g7_strict_seed{seed}"] = {
                "datasets": {"g13": {"ranking": {"roc_auc": value}}}
            }
        result = summarize_strict_seeds(reports)
        metric = result["metrics"]["g13.ranking.roc_auc"]
        self.assertAlmostEqual(metric["mean"], 0.8)
        self.assertAlmostEqual(metric["std"], 0.1)
        self.assertEqual(metric["values_by_seed"]["42"], 0.7)

    def test_nonoverlap_aggregation_preserves_recording_boundaries(self) -> None:
        rows = pd.DataFrame(
            {
                "label": [1, 1, 1, 0, 0],
                "recording_group": ["a", "a", "a", "b", "b"],
                "source_group": ["positive"] * 3 + ["negative"] * 2,
                "dataset_origin": ["uav"] * 3 + ["noise"] * 2,
                "subtype": ["drone"] * 3 + ["traffic"] * 2,
            }
        )
        output_rows, output_scores, audit = aggregate_nonoverlap(
            rows, np.asarray([0.8, 1.0, 0.2, 0.1, 0.3]), windows_per_decision=2
        )
        self.assertEqual(output_rows["label"].tolist(), [1, 0])
        np.testing.assert_allclose(output_scores, [0.9, 0.2])
        self.assertEqual(audit["dropped_incomplete_halfsecond_views"], 1)


if __name__ == "__main__":
    unittest.main()
