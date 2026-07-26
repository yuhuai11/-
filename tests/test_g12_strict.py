from __future__ import annotations

import unittest

import pandas as pd

from dads_crnn.evaluate_g12_strict import _validate_config, leave_one_source_out_audit


class G12StrictTests(unittest.TestCase):
    def test_frozen_config_requires_g7_balanced_fallback(self) -> None:
        config = {
            "algorithm": "g12_strict_source_conformal_v1",
            "seeds": [42, 43, 44],
            "calibration": {
                "target_fpr": 0.01,
                "source_coverage": 0.95,
                "minimum_tune_tpr_retention": 0.80,
                "group_column": "background_source",
                "threshold_selection": "maximum_estimable_source_order_statistic",
                "balanced_mode_policy": "retain_g7",
            },
        }
        _validate_config(config)
        config["calibration"]["balanced_mode_policy"] = "g12"
        with self.assertRaises(ValueError):
            _validate_config(config)

    def test_leave_one_source_out_hashes_source_names(self) -> None:
        rows = []
        for source, scores in {
            "private-a": [0.1, 0.2, 0.3],
            "private-b": [0.2, 0.3, 0.4],
            "private-c": [0.3, 0.4, 0.5],
        }.items():
            rows.extend(
                {
                    "label": 0,
                    "background_source": source,
                    "calibrated_probability": score,
                }
                for score in scores
            )
        result = leave_one_source_out_audit(pd.DataFrame(rows), 0.01)
        self.assertEqual(result["sources"], 3)
        self.assertTrue(result["source_identifiers_hashed"])
        self.assertNotIn("private-a", str(result))


if __name__ == "__main__":
    unittest.main()
