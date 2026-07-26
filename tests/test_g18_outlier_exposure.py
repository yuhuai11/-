from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dads_crnn.probe_g18_outlier_exposure import (
    fit_linear_ood_head,
    known_scores,
    split_unknown_recordings,
)


class G18OutlierExposureTests(unittest.TestCase):
    def test_split_is_model_balanced_and_recording_disjoint(self) -> None:
        rows = []
        for model in ("A", "B"):
            for index in range(10):
                for segment in range(2):
                    rows.append(
                        {
                            "model_id": model,
                            "audio_sha256": f"{model}-{index}",
                            "segment": segment,
                        }
                    )
        train, calibration = split_unknown_recordings(
            pd.DataFrame(rows), train_fraction=0.6, seed=42
        )
        self.assertEqual(train.audio_sha256.nunique(), 12)
        self.assertEqual(calibration.audio_sha256.nunique(), 8)
        self.assertFalse(
            set(train.audio_sha256) & set(calibration.audio_sha256)
        )

    def test_linear_oe_scores_separable_example(self) -> None:
        rng = np.random.default_rng(42)
        known = rng.normal(2.0, 0.2, size=(40, 6))
        unknown = rng.normal(-2.0, 0.2, size=(20, 6))
        pca, scaler, classifier = fit_linear_ood_head(
            known,
            unknown,
            components=3,
            regularization_c=0.1,
            seed=42,
        )
        self.assertGreater(
            known_scores(known, pca, scaler, classifier).mean(),
            known_scores(unknown, pca, scaler, classifier).mean(),
        )


if __name__ == "__main__":
    unittest.main()
