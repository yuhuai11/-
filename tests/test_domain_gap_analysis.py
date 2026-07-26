import unittest

import pandas as pd

from dads_crnn.analyze_domain_gap import validate_paired_identity


class DomainGapAnalysisTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "path": ["a.wav", "b.wav"],
                "sha256": ["a", "b"],
                "label": [0, 1],
                "source_group": ["background", "uav"],
                "uav_source": ["", "uav.wav"],
                "background_source": ["background.wav", ""],
                "condition": ["background_only", "uav_only"],
                "ood_split": ["holdout", "holdout"],
                "calibrated_probability": [0.2, 0.8],
                "selected_prediction": [0, 1],
            }
        )

    def test_paired_identity_accepts_identical_frames(self) -> None:
        left = self._frame()
        self.assertEqual(
            validate_paired_identity(left, left.copy(), split="holdout"),
            validate_paired_identity(left, left.copy(), split="holdout"),
        )

    def test_paired_identity_rejects_reordering(self) -> None:
        left = self._frame()
        right = left.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "identity/order"):
            validate_paired_identity(left, right, split="holdout")

    def test_invalid_probability_is_rejected(self) -> None:
        left = self._frame()
        right = left.copy()
        right.loc[0, "calibrated_probability"] = 1.1
        with self.assertRaisesRegex(ValueError, "Invalid prediction"):
            validate_paired_identity(left, right, split="holdout")


if __name__ == "__main__":
    unittest.main()
