from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dads_crnn.g22_harmonic_features import (
    aggregate_recording_features,
    extract_segment_harmonic_features,
    normalize_class_scores,
)


class G22HarmonicFeatureTests(unittest.TestCase):
    def test_features_are_finite_and_fixed_width(self) -> None:
        time = np.arange(16000) / 16000.0
        waveform = (
            np.sin(2 * np.pi * 150 * time)
            + 0.5 * np.sin(2 * np.pi * 300 * time)
        ).astype(np.float32)

        features = extract_segment_harmonic_features(waveform)

        self.assertEqual(features.shape, (61,))
        self.assertTrue(np.isfinite(features).all())
        self.assertAlmostEqual(float(features[40] * 500.0), 150.0, delta=4.0)

    def test_recording_aggregation_uses_mean_and_deviation(self) -> None:
        frame = pd.DataFrame(
            {
                "audio_sha256": ["a", "a", "b", "b"],
                "target_index": [0, 0, 1, 1],
            }
        )
        segments = np.arange(16, dtype=np.float32).reshape(4, 4)

        values, targets, identities = aggregate_recording_features(frame, segments)

        self.assertEqual(values.shape, (2, 8))
        self.assertEqual(targets.tolist(), [0, 1])
        self.assertEqual(identities, ["a", "b"])

    def test_score_normalization_removes_row_offset_and_scale(self) -> None:
        scores = np.asarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])

        normalized = normalize_class_scores(scores)

        np.testing.assert_allclose(normalized[0], normalized[1], atol=1.0e-7)
        np.testing.assert_allclose(normalized.mean(axis=1), 0.0, atol=1.0e-7)


if __name__ == "__main__":
    unittest.main()
