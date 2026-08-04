import unittest

from dads_crnn.summarize_g7_r4c import METRICS


class G7R4CSummaryTests(unittest.TestCase):
    def test_required_confirmation_metrics_are_registered(self):
        self.assertIn("dads_f1", METRICS)
        self.assertIn("cross_standardized_pauc", METRICS)
        self.assertIn("minus15_tpr_at_0_5", METRICS)
        self.assertIn("minus10_tpr_at_0_5", METRICS)


if __name__ == "__main__":
    unittest.main()
