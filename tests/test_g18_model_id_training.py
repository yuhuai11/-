from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dads_crnn.train_g18_model_id import (
    aggregate_recording_logits,
    balanced_epoch_indices,
    classification_metrics,
)


class G18ModelIdTrainingTests(unittest.TestCase):
    def test_epoch_plan_is_class_balanced_without_replacement(self) -> None:
        targets = np.asarray([0] * 8 + [1] * 5 + [2] * 6)
        indices = balanced_epoch_indices(targets, seed=42, epoch=1)
        selected = targets[indices]
        self.assertEqual(
            {int(key): int(value) for key, value in zip(*np.unique(selected, return_counts=True))},
            {0: 5, 1: 5, 2: 5},
        )
        self.assertEqual(len(indices), len(set(indices.tolist())))

    def test_recording_aggregation_means_logits_and_preserves_target(self) -> None:
        frame = pd.DataFrame(
            {
                "audio_sha256": ["a", "a", "b"],
                "target_index": [0, 0, 1],
            }
        )
        targets, logits = aggregate_recording_logits(
            frame, np.asarray([[2.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
        )
        np.testing.assert_array_equal(targets, [0, 1])
        np.testing.assert_allclose(logits, [[3.0, 0.0], [0.0, 3.0]])

    def test_metrics_report_worst_class(self) -> None:
        targets = np.asarray([0, 0, 1, 1])
        logits = np.asarray([[2, 0], [2, 0], [2, 0], [0, 2]], dtype=float)
        result = classification_metrics(targets, logits, classes=2)
        self.assertEqual(result["per_class_recall"], [1.0, 0.5])
        self.assertEqual(result["minimum_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
