from __future__ import annotations

import unittest
from pathlib import Path

from dads_crnn.config import load_config
from dads_crnn.probe_g22_harmonic_fusion import PROTOCOL, _verify_config


class G22ProtocolTests(unittest.TestCase):
    def test_registered_probe_has_fixed_primary_weight(self) -> None:
        config = load_config(Path("configs/g22_harmonic_fusion_probe.yaml"))

        _verify_config(config)

        self.assertEqual(config["protocol"], PROTOCOL)
        self.assertEqual(config["fusion"]["primary_harmonic_weight"], 0.25)
        self.assertEqual(config["data"]["split_unit"], "audio_sha256")

    def test_unknown_inputs_are_rejected(self) -> None:
        config = load_config(Path("configs/g22_harmonic_fusion_probe.yaml"))
        config["inputs"]["unknown_tune"] = {"path": "forbidden.csv"}

        with self.assertRaisesRegex(ValueError, "Unknown or Holdout"):
            _verify_config(config)

    def test_primary_weight_cannot_be_selected_after_results(self) -> None:
        config = load_config(Path("configs/g22_harmonic_fusion_probe.yaml"))
        config["fusion"]["primary_harmonic_weight"] = 0.5

        with self.assertRaisesRegex(ValueError, "pre-registered"):
            _verify_config(config)


if __name__ == "__main__":
    unittest.main()
