from __future__ import annotations

import unittest

from dads_crnn.config import load_config


class G7R8ProtocolTest(unittest.TestCase):
    def test_candidate_preserves_model_and_training_controls(self) -> None:
        baseline = load_config("configs/g7_r7_freq_mixstyle.yaml")
        candidate = load_config("configs/g7_r8_urban_negatives.yaml")
        self.assertEqual(candidate["model"], baseline["model"])
        for key in baseline["train"]:
            if key == "pos_weight":
                continue
            self.assertEqual(candidate["train"][key], baseline["train"][key])
        self.assertNotIn("sampling", candidate["train"])
        self.assertAlmostEqual(candidate["train"]["pos_weight"], 155685 / 136368)


if __name__ == "__main__":
    unittest.main()
