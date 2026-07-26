from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dads_crnn.calibrate_g18_unknown import (
    aggregate_recording_logits,
    calibration_gate,
    maximum_softmax_scores,
    select_threshold,
    threshold_metrics,
)


class G18UnknownCalibrationTests(unittest.TestCase):
    def test_aggregation_supports_unknown_target(self) -> None:
        frame = pd.DataFrame(
            {
                "audio_sha256": ["a", "a", "b"],
                "model_id": ["KNOWN", "KNOWN", "NOVEL"],
                "is_known": [True, True, False],
                "target_index": [0, 0, -1],
            }
        )
        rows, logits = aggregate_recording_logits(
            frame, np.asarray([[2.0, 0.0], [4.0, 0.0], [0.0, 1.0]])
        )
        self.assertEqual(rows["target_index"].tolist(), [0, -1])
        np.testing.assert_allclose(logits, [[3.0, 0.0], [0.0, 1.0]])

    def test_maximum_softmax_score(self) -> None:
        scores, predictions = maximum_softmax_scores(
            np.asarray([[2.0, 0.0], [0.0, 3.0]])
        )
        np.testing.assert_array_equal(predictions, [0, 1])
        self.assertTrue(np.all(scores > 0.8))

    def test_threshold_selection_respects_known_acceptance(self) -> None:
        known = np.asarray([0.95, 0.90, 0.80, 0.70])
        unknown = np.asarray([0.75, 0.60, 0.55, 0.40])
        threshold, metrics = select_threshold(
            known, unknown, minimum_known_acceptance=0.75
        )
        self.assertEqual(threshold, 0.80)
        self.assertGreaterEqual(metrics["known_acceptance_rate"], 0.75)
        self.assertEqual(metrics["unknown_recall"], 1.0)

    def test_threshold_boundary_is_accepted(self) -> None:
        metrics = threshold_metrics(
            np.asarray([0.7]), np.asarray([0.7]), threshold=0.7
        )
        self.assertEqual(metrics["known_acceptance_rate"], 1.0)
        self.assertEqual(metrics["unknown_recall"], 0.0)

    def test_threshold_is_valid_for_hierarchical_decision(self) -> None:
        threshold, _ = select_threshold(
            np.asarray([0.9]), np.asarray([0.9]), minimum_known_acceptance=1.0
        )
        self.assertGreater(threshold, 0.0)
        self.assertLess(threshold, 1.0)

    def test_calibration_gate_can_stop_holdout(self) -> None:
        passed, checks = calibration_gate(
            {
                "balanced_accuracy": 0.80,
                "unknown_recall": 0.40,
                "known_end_to_end_accuracy": 0.90,
                "known_unknown_auroc": 0.85,
            },
            {
                "minimum_tune_balanced_accuracy": 0.70,
                "minimum_tune_unknown_recall": 0.50,
                "minimum_tune_known_end_to_end_accuracy": 0.80,
                "minimum_tune_known_unknown_auroc": 0.70,
            },
        )
        self.assertFalse(passed)
        self.assertFalse(checks["unknown_recall"]["passed"])


if __name__ == "__main__":
    unittest.main()
