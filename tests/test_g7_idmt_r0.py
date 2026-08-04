from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from dads_crnn.evaluate_g7_idmt_r0 import (
    _segment_metrics,
    validate_development_manifest,
)


class G7IdmtR0Tests(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "contract": {
                "expected_segments": 2,
                "expected_recordings": 1,
                "expected_sessions": 1,
                "expected_events": 1,
                "evaluation_role": "development_test",
                "allowed_locations": ["Allowed"],
                "windows_per_recording": 2,
                "forbidden_tokens": ["Hohenwarte", "final_holdout"],
            }
        }

    def _rows(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "path": ["/tmp/a.wav", "/tmp/a.wav"],
                "label": [0, 0],
                "role": ["development_test", "development_test"],
                "location": ["Allowed", "Allowed"],
                "session_id": ["s", "s"],
                "event_group": ["e", "e"],
                "microphone": ["ME", "ME"],
                "traffic_content": ["vehicle", "vehicle"],
                "recording_id": ["r", "r"],
                "recording_sha256": ["a" * 64, "a" * 64],
                "segment_index": [0, 1],
                "start_sample_16k": [0, 16_000],
                "end_sample_16k": [16_000, 32_000],
                "model_pcm_sha256": ["b" * 64, "c" * 64],
            }
        )

    def test_segment_metrics_are_pure_negative_fpr(self) -> None:
        result = _segment_metrics(np.asarray([0.1, 0.8]), 0.5)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["true_negatives"], 1)
        self.assertEqual(result["fpr"], 0.5)
        self.assertEqual(result["specificity"], 0.5)

    def test_manifest_rejects_locked_location(self) -> None:
        rows = self._rows()
        rows.loc[:, "location"] = "Hohenwarte"
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            rows.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "location set changed"):
                validate_development_manifest(path, self._config())

    def test_manifest_accepts_two_windows_from_one_recording(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            self._rows().to_csv(path, index=False)
            observed = validate_development_manifest(path, self._config())
        self.assertEqual(len(observed), 2)


if __name__ == "__main__":
    unittest.main()
