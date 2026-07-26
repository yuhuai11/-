from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dads_crnn.evaluate_g14_counterfactual import (
    synthesize_pair,
)
from dads_crnn.prepare_g14_counterfactual_pairs import _balanced_select


class G14CounterfactualPairTests(unittest.TestCase):
    def test_synthesis_controls_rms_snr_and_peak(self) -> None:
        time = np.arange(16000, dtype=np.float64) / 16000.0
        background = (0.2 * np.sin(2 * np.pi * 300 * time)).astype(np.float32)
        uav = (0.5 * np.sin(2 * np.pi * 900 * time)).astype(np.float32)
        for snr in (-15.0, -10.0, -5.0, 0.0):
            negative, positive, diagnostic = synthesize_pair(
                background, uav, snr, epsilon=1e-8, peak_limit=0.99
            )
            self.assertTrue(np.isfinite(negative).all())
            self.assertTrue(np.isfinite(positive).all())
            self.assertAlmostEqual(
                diagnostic["negative_rms"], diagnostic["positive_rms"], places=6
            )
            self.assertAlmostEqual(diagnostic["achieved_snr_db"], snr, places=5)
            self.assertLessEqual(diagnostic["joint_peak"], 0.990001)

    def test_shared_gain_makes_negative_identical_across_snr(self) -> None:
        time = np.arange(16000, dtype=np.float64) / 16000.0
        background = (0.95 * np.sin(2 * np.pi * 300 * time)).astype(np.float32)
        uav = (0.95 * np.sin(2 * np.pi * 900 * time)).astype(np.float32)
        snrs = (-15.0, -10.0, -5.0, 0.0)
        required = [
            synthesize_pair(
                background, uav, snr, epsilon=1e-8, peak_limit=0.99
            )[2]["required_common_gain"]
            for snr in snrs
        ]
        shared_gain = min(required)
        pairs = [
            synthesize_pair(
                background,
                uav,
                snr,
                epsilon=1e-8,
                peak_limit=0.99,
                common_gain_override=shared_gain,
            )
            for snr in snrs
        ]
        self.assertTrue(
            all(np.array_equal(pairs[0][0], pair[0]) for pair in pairs[1:])
        )
        self.assertTrue(all(pair[2]["joint_peak"] <= 0.990001 for pair in pairs))

    def test_shared_gain_rejects_gain_that_can_clip(self) -> None:
        background = np.full(16, 0.9, dtype=np.float32)
        uav = np.full(16, 0.9, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "no larger"):
            synthesize_pair(
                background,
                uav,
                0.0,
                epsilon=1e-8,
                peak_limit=0.5,
                common_gain_override=1.0,
            )

    def test_balanced_selection_is_deterministic_and_without_replacement(self) -> None:
        rows = pd.DataFrame(
            {
                "source_group": ["a"] * 5 + ["b"] * 5,
                "device": ["x"] * 10,
                "scene_label": ["park"] * 10,
                "segment_sha256": [f"{index:064x}" for index in range(10)],
            }
        )
        first = _balanced_select(
            rows, 6, strata=["source_group", "device", "scene_label"], seed=42
        )
        second = _balanced_select(
            rows, 6, strata=["source_group", "device", "scene_label"], seed=42
        )
        self.assertEqual(
            first["segment_sha256"].tolist(), second["segment_sha256"].tolist()
        )
        self.assertFalse(first["segment_sha256"].duplicated().any())
        self.assertEqual(first.groupby("source_group").size().to_dict(), {"a": 3, "b": 3})

    def test_balanced_selection_rejects_oversubscription(self) -> None:
        rows = pd.DataFrame(
            {
                "source_group": ["a"],
                "segment_sha256": ["a" * 64],
            }
        )
        with self.assertRaisesRegex(ValueError, "Requested"):
            _balanced_select(rows, 2, strata=["source_group"], seed=42)


if __name__ == "__main__":
    unittest.main()
