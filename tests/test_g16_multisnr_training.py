from __future__ import annotations

import unittest

from dads_crnn.train_g16_multisnr import selection_status


class G16MultiSnrTrainingTests(unittest.TestCase):
    def _raw(self, dads: float, g14: float) -> dict:
        return {
            "dads_validation": {"f1": dads},
            "g14_tune": {"f1": g14},
        }

    def _pair(self, lift: float, ordering: float, fpr: float, tpr: float, mono: float):
        return {
            "by_snr": {
                str(float(snr)): {
                    "mean_lift": lift,
                    "ordering_accuracy": ordering,
                    "negative_fpr": fpr,
                    "positive_tpr": tpr,
                }
                for snr in (-15, -10, -5, 0)
            },
            "monotonicity": mono,
        }

    def _gates(self) -> dict:
        return {
            "maximum_dads_f1_drop": 0.003,
            "maximum_raw_g14_f1_drop": 0.001,
            "minimum_each_snr_mean_lift_gain": 0.005,
            "minimum_each_snr_ordering_gain": 0.0,
            "maximum_each_snr_background_fpr_increase": 0.0,
            "minimum_monotonicity_gain": 0.02,
            "maximum_low_snr_tpr_drop": 0.005,
            "low_snr_db": [-15.0, -10.0],
        }

    def test_all_preregistered_gates_can_pass(self) -> None:
        status = selection_status(
            baseline_raw=self._raw(0.99, 0.98),
            candidate_raw=self._raw(0.99, 0.98),
            baseline_pair=self._pair(0.10, 0.90, 0.10, 0.70, 0.80),
            candidate_pair=self._pair(0.106, 0.91, 0.09, 0.696, 0.83),
            gates=self._gates(),
        )
        self.assertTrue(status["eligible"])

    def test_low_snr_tpr_tradeoff_blocks_checkpoint(self) -> None:
        candidate = self._pair(0.106, 0.91, 0.09, 0.69, 0.83)
        status = selection_status(
            baseline_raw=self._raw(0.99, 0.98),
            candidate_raw=self._raw(0.99, 0.98),
            baseline_pair=self._pair(0.10, 0.90, 0.10, 0.70, 0.80),
            candidate_pair=candidate,
            gates=self._gates(),
        )
        self.assertFalse(status["checks"]["low_snr_tpr_safe"])
        self.assertFalse(status["eligible"])


if __name__ == "__main__":
    unittest.main()
