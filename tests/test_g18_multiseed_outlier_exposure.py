from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dads_crnn.probe_g18_multiseed_outlier_exposure import (
    AGGREGATE_METRICS,
    SEEDS,
    _validate_recording_disjoint,
)


class G18MultiseedOutlierExposureTests(unittest.TestCase):
    def test_replication_seeds_are_frozen(self) -> None:
        self.assertEqual(SEEDS, (43, 44))

    def test_required_aggregate_metrics_are_unique(self) -> None:
        self.assertEqual(len(AGGREGATE_METRICS), len(set(AGGREGATE_METRICS)))
        self.assertIn("known_end_to_end_accuracy", AGGREGATE_METRICS)
        self.assertIn("unknown_recall", AGGREGATE_METRICS)

    def test_recording_overlap_is_rejected(self) -> None:
        frames = {
            "known_train": pd.DataFrame({"audio_sha256": ["a"]}),
            "known_tune": pd.DataFrame({"audio_sha256": ["b"]}),
            "unknown_oe_train": pd.DataFrame({"audio_sha256": ["c"]}),
            "unknown_calibration": pd.DataFrame({"audio_sha256": ["a"]}),
        }
        with self.assertRaisesRegex(ValueError, "recording overlap"):
            _validate_recording_disjoint(frames)

    def test_sample_standard_deviation_convention(self) -> None:
        values = np.asarray([0.8, 0.9, 1.0])
        self.assertAlmostEqual(float(values.std(ddof=1)), 0.1)


if __name__ == "__main__":
    unittest.main()
