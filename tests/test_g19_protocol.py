from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dads_crnn.probe_g19_class_conditional import (
    development_gate_checks,
    split_unknown_development_recordings,
)
from dads_crnn.train_g19_representation import (
    _capture_rng_state,
    _contains_forbidden_split,
    _restore_rng_state,
)


class G19ProtocolTests(unittest.TestCase):
    def test_unknown_split_is_recording_level_stratified_and_deterministic(
        self,
    ) -> None:
        rows = []
        for model_id in ("UNKNOWN_A", "UNKNOWN_B"):
            for recording in range(5):
                for segment in range(2):
                    rows.append(
                        {
                            "model_id": model_id,
                            "audio_sha256": f"{model_id}-{recording}",
                            "segment_index": segment,
                        }
                    )
        frame = pd.DataFrame(rows)

        first_calibration, first_audit = split_unknown_development_recordings(
            frame,
            calibration_fraction=0.4,
            seed=42,
        )
        second_calibration, second_audit = split_unknown_development_recordings(
            frame,
            calibration_fraction=0.4,
            seed=42,
        )

        self.assertEqual(
            set(first_calibration["audio_sha256"]),
            set(second_calibration["audio_sha256"]),
        )
        self.assertEqual(
            set(first_audit["audio_sha256"]),
            set(second_audit["audio_sha256"]),
        )
        self.assertFalse(
            set(first_calibration["audio_sha256"])
            & set(first_audit["audio_sha256"])
        )
        self.assertEqual(
            first_calibration.groupby("model_id")["audio_sha256"]
            .nunique()
            .to_dict(),
            {"UNKNOWN_A": 2, "UNKNOWN_B": 2},
        )
        self.assertEqual(
            first_audit.groupby("model_id")["audio_sha256"].nunique().to_dict(),
            {"UNKNOWN_A": 3, "UNKNOWN_B": 3},
        )

    def test_development_config_rejects_holdout_keys_recursively(self) -> None:
        self.assertTrue(
            _contains_forbidden_split(
                {"nested": {"unknown_holdout": {"path": "anything.csv"}}}
            )
        )
        self.assertTrue(_contains_forbidden_split({"final": "anything.csv"}))
        self.assertTrue(
            _contains_forbidden_split(
                {"known_tune": {"path": "registry/known_holdout.csv"}}
            )
        )
        self.assertTrue(
            _contains_forbidden_split(
                {"unknown_tune": {"path": "registry/X6D_segments.csv"}}
            )
        )
        self.assertFalse(
            _contains_forbidden_split(
                {
                    "known_train": {"path": "known_train.csv"},
                    "known_tune": {"path": "known_tune.csv"},
                    "unknown_tune": {"path": "unknown_tune.csv"},
                }
            )
        )

    def test_rng_state_round_trip_is_weights_only_safe_and_exact(self) -> None:
        random.seed(19)
        np.random.seed(19)
        torch.manual_seed(19)
        state = _capture_rng_state()
        expected = (random.random(), float(np.random.random()), torch.rand(3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rng.pt"
            torch.save({"rng_state": state}, path)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        _restore_rng_state(loaded["rng_state"])
        observed = (random.random(), float(np.random.random()), torch.rand(3))

        self.assertEqual(observed[0], expected[0])
        self.assertEqual(observed[1], expected[1])
        torch.testing.assert_close(observed[2], expected[2])

    def test_development_gate_cannot_pass_with_zero_unknown_recall(self) -> None:
        metrics = {
            "known_acceptance_rate": 1.0,
            "known_correct_and_accepted_rate": 1.0,
            "minimum_supported_predicted_class_known_acceptance": 1.0,
            "unknown_recall": 0.0,
            "balanced_open_set_accuracy": 0.5,
            "known_unknown_roc_auc": 0.95,
        }
        gates = {
            "minimum_tune_known_acceptance": 0.9,
            "minimum_known_correct_and_accepted_rate": 0.7,
            "minimum_supported_predicted_class_known_acceptance": 0.7,
            "minimum_same_unknown_recording_recall": 0.8,
            "minimum_same_unknown_model_recall": 0.7,
            "minimum_development_balanced_open_set_accuracy": 0.85,
            "minimum_tune_known_unknown_auroc": 0.7,
        }

        passed, checks = development_gate_checks(
            metrics,
            {"UNKNOWN_A": {"unknown_recall": 0.0}},
            gates,
        )

        self.assertFalse(passed)
        self.assertFalse(checks["same_unknown_recording_recall"]["passed"])
        self.assertTrue(checks["known_unknown_roc_auc"]["passed"])


if __name__ == "__main__":
    unittest.main()
