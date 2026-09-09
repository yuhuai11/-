from __future__ import annotations

import unittest

from dads_crnn.config import load_config


CONFIGS = {
    "control": "configs/g7_r9_domain_quota_control.yaml",
    "tau05": "configs/g7_r9_domain_quota_tau05.yaml",
    "tau10": "configs/g7_r9_domain_quota_tau10.yaml",
}


class G7R9ProtocolTests(unittest.TestCase):
    def test_arms_only_change_dose_and_output_identity(self) -> None:
        configs = {name: load_config(path) for name, path in CONFIGS.items()}
        normalized = []
        for config in configs.values():
            config.pop("protocol")
            config.pop("output_dir")
            config["train"]["sampling"]["negative_domain_fractions"] = "DOSE"
            normalized.append(config)
        self.assertEqual(normalized[0], normalized[1])
        self.assertEqual(normalized[0], normalized[2])

    def test_doses_and_fixed_training_budget(self) -> None:
        expected_tau = {"control": 0.0, "tau05": 0.05, "tau10": 0.10}
        for name, path in CONFIGS.items():
            config = load_config(path)
            sampling = config["train"]["sampling"]
            self.assertEqual(sampling["type"], "class_domain_quota_batch")
            self.assertEqual(sampling["samples_per_epoch"], 292096)
            self.assertEqual(sampling["positive_per_batch"], 60)
            self.assertAlmostEqual(
                sampling["negative_domain_fractions"]["negative:tau_urban_train"],
                expected_tau[name],
            )
            self.assertAlmostEqual(config["train"]["pos_weight"], 155685 / 136368)


if __name__ == "__main__":
    unittest.main()
