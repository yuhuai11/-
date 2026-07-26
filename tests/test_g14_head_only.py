from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dads_crnn.config import load_config
from dads_crnn.metrics import file_level_metrics


ROOT = Path(__file__).resolve().parents[1]


class G14HeadOnlyTests(unittest.TestCase):
    def test_file_metrics_aggregate_g14_segments_by_archive_member(self) -> None:
        rows = pd.DataFrame(
            {
                "archive_path": ["uav.zip", "uav.zip", "background.zip"],
                "archive_member": ["uav.wav", "uav.wav", "street.wav"],
                "audio_sha256": ["uav", "uav", "background"],
                "label": [1, 1, 0],
            }
        )
        result = file_level_metrics(
            rows,
            np.asarray([0.8, 1.0, 0.1]),
            [0.5],
            aggregation="mean",
        )
        self.assertEqual(result[0]["files"], 2)
        self.assertEqual(result[0]["tp"], 1)
        self.assertEqual(result[0]["tn"], 1)

    def test_file_metrics_preserve_dads_identity_priority(self) -> None:
        rows = pd.DataFrame(
            {
                "parquet_file": ["part.parquet", "part.parquet"],
                "row_group": [0, 0],
                "row_in_group": [7, 7],
                "archive_path": ["different.zip", "different.zip"],
                "archive_member": ["a.wav", "b.wav"],
                "label": [1, 1],
            }
        )
        result = file_level_metrics(
            rows,
            np.asarray([0.8, 1.0]),
            [0.5],
            aggregation="max",
        )
        self.assertEqual(result[0]["files"], 1)

    def test_config_locks_g7_and_only_requests_head_training(self) -> None:
        config = load_config(ROOT / "configs/g14_panns_head_only.yaml")
        self.assertEqual(config["model"]["trainable_scope"], "binary_head_only")
        self.assertEqual(
            config["model"]["binary_checkpoint_sha256"],
            "d357f194105ec27a61838e0f4c2e7575932e2dd861e529ef970b1fb467954cdc",
        )
        self.assertNotIn("augmentation", config["train"])
        self.assertEqual(
            config["train"]["sampling"]["type"],
            "class_source_sqrt_balanced_batch",
        )
        self.assertEqual(
            config["data"]["split_names"],
            {"train": "train", "val": "tune", "test": "dev_holdout"},
        )


if __name__ == "__main__":
    unittest.main()
