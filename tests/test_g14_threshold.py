from __future__ import annotations

import unittest
from pathlib import Path

from dads_crnn.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class G14ThresholdTests(unittest.TestCase):
    def test_selection_is_shared_tune_only_and_strict(self) -> None:
        config = load_config(ROOT / "configs/g14_c_tune_only_threshold.yaml")
        self.assertEqual(
            config["selection"]["rule"],
            "lowest_shared_threshold_passing_every_seed",
        )
        self.assertEqual(config["seeds"], [42, 43, 44])
        self.assertEqual(config["selection"]["dads_val_maximum_f1_drop"], 0.005)
        self.assertEqual(
            config["selection"]["dads_val_maximum_specificity_drop"], 0.005
        )
        selection_text = str(config["selection"]).lower()
        self.assertNotIn("holdout", selection_text)
        self.assertNotIn("guard", selection_text)
        self.assertNotIn("test", selection_text)

    def test_fit_and_evaluation_outputs_are_separate(self) -> None:
        script = (ROOT / "scripts/run_g14_c_threshold.sh").read_text(encoding="utf-8")
        self.assertIn("preflight|fit|evaluate", script)
        self.assertIn('--mode "$mode"', script)


if __name__ == "__main__":
    unittest.main()
