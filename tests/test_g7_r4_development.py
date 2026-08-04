import unittest

from dads_crnn.evaluate_g7_r4_development import _selection


def report(dads_f1, kielce_recall, tau_fpr, low15, low10):
    return {
        "domains": {
            "dads_halfsec": {"threshold_0_5": {"f1": dads_f1}},
            "kielce_17_uav": {"recall_at_0_5": kielce_recall},
            "tau_urban_2022": {"fpr_at_0_5": tau_fpr},
        },
        "low_snr": {"by_snr": {
            "-15.0": {"positive_tpr_at_0_5": low15},
            "-10.0": {"positive_tpr_at_0_5": low10},
        }},
    }


class G7R4DevelopmentTests(unittest.TestCase):
    def test_all_gates_must_pass(self):
        baseline = report(0.99, 0.90, 0.02, 0.30, 0.40)
        candidate = report(0.988, 0.906, 0.024, 0.30, 0.42)
        result = _selection(baseline, candidate)
        self.assertTrue(result["eligible"])

    def test_low_snr_regression_blocks_promotion(self):
        baseline = report(0.99, 0.90, 0.02, 0.30, 0.40)
        candidate = report(0.99, 0.91, 0.02, 0.29, 0.45)
        result = _selection(baseline, candidate)
        self.assertFalse(result["eligible"])
        self.assertFalse(result["checks"]["low_snr_recall_noninferior"])


if __name__ == "__main__":
    unittest.main()
