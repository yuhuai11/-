from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dads_crnn.data_firewall import file_sha256
from dads_crnn.evaluate_g7_r1_guards import (
    _verified_internal_metrics,
    _verify_guard_cache,
)
from dads_crnn.metrics import binary_metrics


class G7R1GuardTests(unittest.TestCase):
    def test_internal_metrics_are_recomputed_from_hash_bound_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "seed_42"
            run_dir.mkdir(parents=True)
            checkpoint_path = run_dir / "best.pt"
            checkpoint_path.write_bytes(b"checkpoint-placeholder")

            manifest_path = root / "manifest.csv"
            pd.DataFrame(
                {
                    "split": ["train", "val", "val", "test", "test"],
                    "label": [1, 0, 1, 0, 1],
                }
            ).to_csv(manifest_path, index=False)

            values = {
                "val_labels": np.asarray([0, 1], dtype=np.float32),
                "val_probabilities": np.asarray([0.1, 0.9], dtype=np.float32),
                "test_labels": np.asarray([0, 1], dtype=np.float32),
                "test_probabilities": np.asarray([0.2, 0.8], dtype=np.float32),
            }
            hashes = {}
            for name, value in values.items():
                path = run_dir / f"{name}.npy"
                np.save(path, value, allow_pickle=False)
                hashes[name] = file_sha256(path)

            val_metrics = binary_metrics(
                values["val_labels"].astype(np.int64),
                values["val_probabilities"],
                0.5,
            )
            test_metrics = binary_metrics(
                values["test_labels"].astype(np.int64),
                values["test_probabilities"],
                0.5,
            )
            protocol = {"inputs": {"training_manifest": manifest_path.as_posix()}}
            metrics = {
                "prediction_sha256": hashes,
                "val_threshold_metrics": [val_metrics],
                "threshold_metrics": [test_metrics],
            }

            observed = _verified_internal_metrics(
                root, protocol, checkpoint_path, metrics
            )

            self.assertEqual(observed["val"]["tp"], 1)
            self.assertEqual(observed["test"]["tn"], 1)

    def test_internal_metrics_reject_stored_metric_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "seed_42"
            run_dir.mkdir(parents=True)
            checkpoint_path = run_dir / "best.pt"
            checkpoint_path.write_bytes(b"checkpoint-placeholder")
            manifest_path = root / "manifest.csv"
            pd.DataFrame(
                {"split": ["val", "val", "test", "test"], "label": [0, 1, 0, 1]}
            ).to_csv(manifest_path, index=False)

            hashes = {}
            for prefix in ("val", "test"):
                labels = np.asarray([0, 1], dtype=np.float32)
                probabilities = np.asarray([0.1, 0.9], dtype=np.float32)
                for suffix, value in (
                    ("labels", labels),
                    ("probabilities", probabilities),
                ):
                    name = f"{prefix}_{suffix}"
                    path = run_dir / f"{name}.npy"
                    np.save(path, value, allow_pickle=False)
                    hashes[name] = file_sha256(path)

            correct = binary_metrics(
                np.asarray([0, 1]), np.asarray([0.1, 0.9]), 0.5
            )
            drifted = dict(correct)
            drifted["f1"] = 0.0
            protocol = {"inputs": {"training_manifest": manifest_path.as_posix()}}
            metrics = {
                "prediction_sha256": hashes,
                "val_threshold_metrics": [drifted],
                "threshold_metrics": [correct],
            }

            with self.assertRaisesRegex(ValueError, "metric drift for f1"):
                _verified_internal_metrics(root, protocol, checkpoint_path, metrics)

    def test_guard_cache_hash_is_verified_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "guard.npy"
            np.save(
                cache_path,
                np.asarray([0.25, -0.25], dtype=np.float32),
                allow_pickle=False,
            )
            rows = pd.DataFrame(
                {
                    "cache_path": [cache_path.as_posix()],
                    "cache_sha256": [file_sha256(cache_path)],
                }
            )

            self.assertEqual(_verify_guard_cache(rows, root)["files"], 1)
            cache_path.write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "cache SHA256 mismatch"):
                _verify_guard_cache(rows, root)


if __name__ == "__main__":
    unittest.main()
