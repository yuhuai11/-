from __future__ import annotations

import unittest

import pandas as pd

from dads_crnn.prepare_g15_development_registry import (
    normalize_dads_validation,
    validate_isolation,
)


class G15DevelopmentRegistryTests(unittest.TestCase):
    def test_dads_validation_normalization_is_not_training_eligible(self) -> None:
        frame = pd.DataFrame(
            {
                "split": ["val", "val"],
                "label": [0, 1],
                "cache_path": ["a.npy", "b.npy"],
                "raw_audio_sha256": ["a" * 64, "b" * 64],
                "recording_group": ["a", "b"],
            }
        )
        result = normalize_dads_validation(frame)
        self.assertEqual(set(result["split"]), {"validation"})
        self.assertFalse(result["teacher_distillation_eligible"].any())
        self.assertFalse(result["background_mix_eligible"].any())

    def test_rejects_training_hash_overlap(self) -> None:
        development = {
            "dads_validation": pd.DataFrame({"audio_sha256": ["a" * 64]})
        }
        with self.assertRaisesRegex(ValueError, "isolation failed"):
            validate_isolation(
                development,
                training_hashes={"a" * 64},
                consumed_hashes=frozenset(),
            )


if __name__ == "__main__":
    unittest.main()
