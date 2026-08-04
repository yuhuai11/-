from __future__ import annotations

import unittest
from pathlib import Path

from dads_crnn.config import load_config
from dads_crnn.train_g20_closed_set import (
    PROTOCOL,
    _scheduler_lambda,
    _verify_config,
)


class G20ProtocolTests(unittest.TestCase):
    def test_registered_config_is_known_only(self) -> None:
        config = load_config(Path("configs/g20_closed_set_multiscale.yaml"))

        _verify_config(config)

        self.assertEqual(config["protocol"], PROTOCOL)
        self.assertEqual(set(config["inputs"]) & {"unknown_tune", "known_holdout"}, set())
        self.assertEqual(config["model"]["attention_heads"], 4)

    def test_config_rejects_unseen_model_input(self) -> None:
        config = load_config(Path("configs/g20_closed_set_multiscale.yaml"))
        config["inputs"]["unknown_tune"] = {"path": "forbidden.csv"}

        with self.assertRaisesRegex(ValueError, "Unknown or Holdout"):
            _verify_config(config)

    def test_warmup_cosine_schedule(self) -> None:
        values = [
            _scheduler_lambda(index, warmup=2, epochs=6) for index in range(6)
        ]

        self.assertAlmostEqual(values[0], 0.5)
        self.assertAlmostEqual(values[1], 1.0)
        self.assertGreater(values[2], values[3])
        self.assertAlmostEqual(values[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
