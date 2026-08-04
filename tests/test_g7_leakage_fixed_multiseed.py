from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dads_crnn.summarize_g7_leakage_fixed_multiseed import build_aggregate


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class G7LeakageFixedMultiseedTests(unittest.TestCase):
    def _write_seed(self, root: Path, seed: int, f1: float) -> None:
        seed_dir = root / f"seed_{seed}"
        seed_dir.mkdir(parents=True)
        (seed_dir / "best.pt").write_bytes(f"checkpoint-{seed}".encode())
        prediction_hashes = {}
        for name in (
            "val_probabilities",
            "val_labels",
            "test_probabilities",
            "test_labels",
        ):
            path = seed_dir / f"{name}.npy"
            np.save(path, np.asarray([seed], dtype=np.float32))
            prediction_hashes[name] = _sha(path)
        item = {
            "threshold": 0.5,
            "accuracy": f1,
            "balanced_accuracy": f1,
            "precision": f1,
            "recall": f1,
            "specificity": f1,
            "false_positive_rate": 1.0 - f1,
            "f1": f1,
            "auc": f1,
            "pr_auc": f1,
        }
        metrics = {
            "seed": seed,
            "model_type": "panns_cnn14_16k",
            "initialization": "audioset",
            "feature_type": "panns_log_mel",
            "parameter_count": 10,
            "mixed_precision": True,
            "amp_init_scale": 8192.0,
            "amp_nonfinite_gradient_skip_steps": 0,
            "official_pretraining_sha256": "weights",
            "best_epoch": seed,
            "elapsed_seconds": float(seed),
            "locked_datasets_read": [],
            "training_inputs": {
                "manifest_sha256": "manifest",
                "input_audits": [
                    {"protocol": "dads_native_half_second_content_component_v2"}
                ],
            },
            "checkpoint_sha256": _sha(seed_dir / "best.pt"),
            "prediction_sha256": prediction_hashes,
            "val_threshold_metrics": [item],
            "threshold_metrics": [item],
            "val_file_metrics": {"mean": [item], "max": [item]},
            "test_file_metrics": {"mean": [item], "max": [item]},
        }
        (seed_dir / "metrics.json").write_text(json.dumps(metrics))

    def test_aggregate_uses_sample_standard_deviation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_seed(root, 42, 0.8)
            self._write_seed(root, 43, 0.9)
            self._write_seed(root, 44, 1.0)
            result = build_aggregate(root, [42, 43, 44])
            self.assertAlmostEqual(result["primary_internal_test"]["f1"]["mean"], 0.9)
            self.assertAlmostEqual(
                result["primary_internal_test"]["f1"]["sample_std"], 0.1
            )
            self.assertEqual(result["selection_rule"], "report_all_seeds_no_seed_selection")
            self.assertFalse(result["fresh_final_holdout"])

    def test_rejects_mismatched_training_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_seed(root, 42, 0.8)
            self._write_seed(root, 43, 0.9)
            path = root / "seed_43" / "metrics.json"
            metrics = json.loads(path.read_text())
            metrics["training_inputs"]["manifest_sha256"] = "different"
            path.write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, "identical training input identity"):
                build_aggregate(root, [42, 43])


if __name__ == "__main__":
    unittest.main()
