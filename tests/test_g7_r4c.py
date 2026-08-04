import tempfile
import unittest
from pathlib import Path

from dads_crnn.audit_g7_r4c import audit


class G7R4CAuditTests(unittest.TestCase):
    def test_checked_in_configs_are_single_dose_ablation(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit(
                Path("configs/g7_r4b_lowsnr_curriculum.yaml"),
                Path("configs/g7_r4c_lowsnr_conservative.yaml"),
                Path(directory) / "audit.json",
            )
            self.assertTrue(result["passed"])
            self.assertAlmostEqual(
                result["expected_low_snr_fraction_of_all_positive_views"], 0.0495
            )


if __name__ == "__main__":
    unittest.main()
