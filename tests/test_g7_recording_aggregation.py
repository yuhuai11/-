from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dads_crnn.evaluate_g7_recording_aggregation import (
    aggregate_recordings,
    aggregate_scores,
    native_window_count,
)


class NativeWindowProtocolTests(unittest.TestCase):
    def test_window_count_keeps_all_complete_halves_without_padding_tail(self) -> None:
        self.assertEqual(native_window_count(44100, 44100, 16000, 8000), 2)
        self.assertEqual(native_window_count(66150, 44100, 16000, 8000), 3)
        self.assertEqual(native_window_count(22000, 44100, 16000, 8000), 0)


class RecordingAggregationTests(unittest.TestCase):
    def test_supported_fixed_aggregations(self) -> None:
        values = np.asarray([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(aggregate_scores(values, method="mean_probability"), 0.5)
        self.assertAlmostEqual(aggregate_scores(values, method="max_probability"), 0.9)
        self.assertAlmostEqual(
            aggregate_scores(
                values, method="topk_mean_probability", topk_fraction=0.5
            ),
            0.85,
        )
        logit_mean = aggregate_scores(values, method="mean_logit_then_sigmoid")
        self.assertAlmostEqual(logit_mean, 0.5)

    def test_constrained_autopool_respects_single_instance_weight_cap(self) -> None:
        values = np.asarray([1.0, 0.0])
        for cap in (0.55, 0.65, 0.75):
            pooled = aggregate_scores(
                values,
                method="constrained_autopool_probability",
                max_instance_weight=cap,
            )
            self.assertAlmostEqual(pooled, cap)

    def test_standard_two_window_cap_reduces_to_mean(self) -> None:
        values = np.asarray([0.1, 0.9])
        pooled = aggregate_scores(
            values,
            method="constrained_autopool_probability",
            max_instance_weight=0.5,
        )
        self.assertAlmostEqual(pooled, values.mean())

    def test_recording_windows_must_be_consecutive(self) -> None:
        rows = pd.DataFrame(
            {
                "recording_index": [0, 0],
                "recording_id": ["a", "a"],
                "segment_index": [0, 2],
                "label": [1, 1],
                "probability": [0.2, 0.8],
            }
        )
        with self.assertRaisesRegex(ValueError, "consecutive"):
            aggregate_recordings(rows, method="mean_probability")

    def test_recordings_are_not_mixed(self) -> None:
        rows = pd.DataFrame(
            {
                "recording_index": [0, 0, 1, 1],
                "recording_id": ["a", "a", "b", "b"],
                "segment_index": [0, 1, 0, 1],
                "label": [0, 0, 1, 1],
                "probability": [0.1, 0.3, 0.7, 0.9],
            }
        )
        result = aggregate_recordings(rows, method="mean_probability")
        self.assertEqual(result["recording_id"].tolist(), ["a", "b"])
        self.assertEqual(result["windows"].tolist(), [2, 2])
        np.testing.assert_allclose(result["probability"], [0.2, 0.8])


if __name__ == "__main__":
    unittest.main()
