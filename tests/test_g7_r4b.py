import tempfile
import unittest
from pathlib import Path

from dads_crnn.audit_g7_r4b import audit


class G7R4BAuditTests(unittest.TestCase):
    def test_checked_in_configs_are_single_variable_ablation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            result = audit(
                Path("configs/g7_r4a_source_balanced_fc1.yaml"),
                Path("configs/g7_r4b_lowsnr_curriculum.yaml"),
                output,
            )
            self.assertTrue(result["passed"])
            self.assertAlmostEqual(
                result["expected_low_snr_fraction_of_all_positive_views"], 0.099
            )


if __name__ == "__main__":
    unittest.main()
