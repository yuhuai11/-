from __future__ import annotations

import unittest
from pathlib import Path

from dads_crnn.evaluate_g13_confirmation import (
    ALGORITHM,
    UNLOCK_PHRASE,
    _validate_config,
    evaluate_once,
)


class G13EvaluationTests(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "algorithm": ALGORITHM,
            "models": {"candidate_checkpoints": ["a", "b", "c"]},
            "contract": {
                "candidate_seeds": [42, 43, 44],
                "target_fpr": 0.01,
                "candidate_ensemble": "arithmetic_mean_raw_logit",
                "balanced_mode_policy": "retain_g7",
            },
        }

    def test_unlock_phrase_is_explicit_and_stable(self) -> None:
        self.assertEqual(UNLOCK_PHRASE, "RUN_G13_EXTERNAL_CONFIRMATION_ONCE")
        with self.assertRaises(PermissionError):
            evaluate_once(Path("does-not-exist.yaml"), "")

    def test_frozen_contract_accepts_only_expected_candidate(self) -> None:
        _validate_config(self._config())
        config = self._config()
        config["contract"]["candidate_seeds"] = [42]
        with self.assertRaises(ValueError):
            _validate_config(config)

    def test_balanced_mode_cannot_be_promoted(self) -> None:
        config = self._config()
        config["contract"]["balanced_mode_policy"] = "promote_candidate"
        with self.assertRaises(ValueError):
            _validate_config(config)


if __name__ == "__main__":
    unittest.main()
