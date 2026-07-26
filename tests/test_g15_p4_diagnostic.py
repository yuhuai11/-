from __future__ import annotations

import unittest
from pathlib import Path

from dads_crnn.config import load_config


class G15P4DiagnosticTests(unittest.TestCase):
    def test_config_is_tune_only_and_nonpromotional(self) -> None:
        config = load_config(Path("configs/g15_p4_counterfactual_tune.yaml"))
        self.assertEqual(set(config["pair_manifests"]), {"tune"})
        self.assertNotIn("dev_holdout", str(config["pair_manifests"]))
        self.assertEqual(
            [model["role"] for model in config["models"]],
            ["baseline", "terminated_development_diagnostic"],
        )
        self.assertEqual(
            config["models"][1]["checkpoint"],
            "artifacts/g15_constrained_adaptation/p3b_seed42/seed_42/last.pt",
        )
        self.assertEqual(
            config["mixing"]["expected_target_snr_db"],
            [-15.0, -10.0, -5.0, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
