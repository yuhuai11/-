from __future__ import annotations

import unittest

from pathlib import Path

from dads_crnn.evaluate_g11_final import UNLOCK_PHRASE, _validate_config, evaluate_final


class G11FinalTests(unittest.TestCase):
    def test_unlock_phrase_is_explicit_and_stable(self) -> None:
        self.assertEqual(UNLOCK_PHRASE, "RUN_G11_FINAL_ONCE")
        with self.assertRaises(PermissionError):
            evaluate_final(Path("does-not-exist.yaml"), "")

    def test_rejects_non_frozen_mode_mapping(self) -> None:
        config = {
            "algorithm": "g11_final_dual_mode_v1",
            "evaluation": {
                "modes": {
                    "strict": {"target_fpr": 0.01, "candidate_threshold_source": "pooled"},
                    "balanced": {"target_fpr": 0.05, "candidate_threshold_source": "pooled"},
                }
            },
            "final_data": {"datasets": {"unseen": {}, "real_world": {}}},
        }
        with self.assertRaises(ValueError):
            _validate_config(config)


if __name__ == "__main__":
    unittest.main()
