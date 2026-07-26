from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dads_crnn.evaluate_g9_guard import negative_metrics


class G9GuardEvaluationTests(unittest.TestCase):
    def test_negative_metrics_include_segment_recording_source_and_class_fpr(self) -> None:
        rows = pd.DataFrame(
            {
                "recording_group": ["r1", "r1", "r2", "r2"],
                "source_group": ["s1", "s1", "s2", "s2"],
                "hard_negative_class": ["airplane", "airplane", "engine", "engine"],
            }
        )
        result = negative_metrics(rows, np.array([0.1, 0.8, 0.2, 0.3]), 0.5)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["segment_false_positive_rate"], 0.25)
        self.assertEqual(result["recording_any_false_positive_rate"], 0.5)
        self.assertEqual(result["source_any_false_positive_rate"], 0.5)
        by_class = {row["hard_negative_class"]: row for row in result["per_class"]}
        self.assertEqual(by_class["airplane"]["false_positive_rate"], 0.5)
        self.assertEqual(by_class["engine"]["false_positive_rate"], 0.0)

    def test_negative_metrics_reject_non_finite_or_misaligned_probabilities(self) -> None:
        rows = pd.DataFrame(
            {
                "recording_group": ["r1"],
                "source_group": ["s1"],
                "hard_negative_class": ["engine"],
            }
        )
        with self.assertRaises(ValueError):
            negative_metrics(rows, np.array([]), 0.5)
        with self.assertRaises(ValueError):
            negative_metrics(rows, np.array([np.nan]), 0.5)


if __name__ == "__main__":
    unittest.main()
