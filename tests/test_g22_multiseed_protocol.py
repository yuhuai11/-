from __future__ import annotations

import unittest
from pathlib import Path

from dads_crnn.config import load_config
from dads_crnn.evaluate_g22_harmonic_multiseed import PROTOCOL


class G22MultiseedProtocolTests(unittest.TestCase):
    def test_registered_config_keeps_fixed_weight_and_three_seeds(self) -> None:
        config = load_config(Path("configs/g22_harmonic_fusion_multiseed.yaml"))

        self.assertEqual(config["protocol"], PROTOCOL)
        self.assertEqual(config["fusion"]["harmonic_weight"], 0.25)
        self.assertEqual(set(config["inputs"]["checkpoints"]), {"42", "43", "44"})
        self.assertTrue(config["gates"]["mean_accuracy_noninferior"])
        self.assertTrue(config["gates"]["mean_macro_f1_strictly_improved"])


if __name__ == "__main__":
    unittest.main()
