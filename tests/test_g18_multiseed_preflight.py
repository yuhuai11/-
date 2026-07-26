from __future__ import annotations

import unittest

from dads_crnn.preflight_g18_multiseed import comparable_config
from dads_crnn.train_g18_model_id import MULTISEED_PROTOCOL, SUPPORTED_PROTOCOLS


class G18MultiseedPreflightTests(unittest.TestCase):
    def test_multiseed_protocol_is_supported(self) -> None:
        self.assertIn(MULTISEED_PROTOCOL, SUPPORTED_PROTOCOLS)

    def test_comparison_ignores_only_seed(self) -> None:
        first = {"train": {"seed": 43, "epochs": 25}, "value": 1}
        second = {"train": {"seed": 44, "epochs": 25}, "value": 1}
        self.assertEqual(comparable_config(first), comparable_config(second))
        second["train"]["epochs"] = 24
        self.assertNotEqual(comparable_config(first), comparable_config(second))


if __name__ == "__main__":
    unittest.main()
