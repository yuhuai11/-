from __future__ import annotations

import unittest
from pathlib import Path

from dads_crnn.config import load_config
from dads_crnn.train_g21_fc1_transfer import PROTOCOL, _verify_config


class G21ProtocolTests(unittest.TestCase):
    def test_registered_config_is_known_only_and_narrow(self) -> None:
        config = load_config(Path("configs/g21_fc1_transfer.yaml"))

        _verify_config(config)

        self.assertEqual(config["protocol"], PROTOCOL)
        self.assertEqual(
            config["model"]["trainable_detector_prefixes"], ["backbone.fc1."]
        )
        self.assertLess(
            config["train"]["detector_learning_rate"],
            config["train"]["head_learning_rate"],
        )

    def test_unknown_input_is_rejected(self) -> None:
        config = load_config(Path("configs/g21_fc1_transfer.yaml"))
        config["inputs"]["unknown_tune"] = {"path": "forbidden.csv"}

        with self.assertRaisesRegex(ValueError, "Unknown or Holdout"):
            _verify_config(config)

    def test_broad_unfreezing_is_rejected(self) -> None:
        config = load_config(Path("configs/g21_fc1_transfer.yaml"))
        config["model"]["trainable_detector_prefixes"] = ["backbone."]

        with self.assertRaisesRegex(ValueError, "fc1"):
            _verify_config(config)


if __name__ == "__main__":
    unittest.main()
