import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dads_crnn.audit_experiment_delta import config_differences
from dads_crnn.calibrate_ood import (
    _prediction_frame,
    fit_temperature,
    probabilities_from_logits,
    probability_metrics,
    search_threshold,
    verify_manifest_audio_hashes,
)
from dads_crnn.config import load_config
from dads_crnn.gate_candidate import (
    _validate_prediction_content,
    assess_internal_subgroups,
    paired_multiway_bootstrap,
    supported_meaningful_improvement_metrics,
)
from dads_crnn.evaluate_external import subgroup_metrics
from dads_crnn.train import _build_model


class G6TemporalAttentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline_config = load_config("configs/crnn_dads_full_augmented_g2.yaml")
        self.candidate_config = load_config(
            "configs/crnn_dads_full_augmented_g6_temporal_attention.yaml"
        )

    def test_config_diff_and_parameter_delta_are_exact(self) -> None:
        self.assertEqual(
            set(config_differences(self.baseline_config, self.candidate_config)),
            {"model.temporal_pooling", "output_dir"},
        )
        baseline = _build_model(self.baseline_config)
        candidate = _build_model(self.candidate_config)
        baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
        candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())
        self.assertEqual(baseline_parameters, 1_012_193)
        self.assertEqual(candidate_parameters, 1_012_449)
        self.assertEqual(candidate_parameters - baseline_parameters, 256)
        self.assertNotIn("temporal_attention.weight", baseline.state_dict())
        self.assertEqual(
            set(candidate.state_dict()) - set(baseline.state_dict()),
            {"temporal_attention.weight"},
        )

    def test_zero_attention_nests_mean_without_advancing_rng(self) -> None:
        torch.manual_seed(1234)
        baseline = _build_model(self.baseline_config)
        baseline_rng = torch.get_rng_state().clone()
        torch.manual_seed(1234)
        candidate = _build_model(self.candidate_config)
        candidate_rng = torch.get_rng_state().clone()
        self.assertTrue(torch.equal(baseline_rng, candidate_rng))
        incompatible = candidate.load_state_dict(baseline.state_dict(), strict=False)
        self.assertEqual(incompatible.missing_keys, ["temporal_attention.weight"])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertIsNone(candidate.temporal_attention.bias)
        self.assertTrue(
            torch.equal(
                candidate.temporal_attention.weight,
                torch.zeros_like(candidate.temporal_attention.weight),
            )
        )
        baseline.eval()
        candidate.eval()
        features = torch.randn(3, 1, 64, 101)
        with torch.no_grad():
            baseline_logits = baseline(features)
            candidate_logits = candidate(features)
        self.assertTrue(
            torch.allclose(baseline_logits, candidate_logits, rtol=1e-6, atol=1e-7)
        )

    def test_zero_attention_is_trainable(self) -> None:
        model = _build_model(self.candidate_config)
        sequence = torch.randn(4, 12, 256)
        target = torch.linspace(-1.0, 1.0, 256)
        pooled = model.temporal_pool(sequence)
        loss = (pooled * target).sum()
        loss.backward()
        gradient = model.temporal_attention.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.norm()), 0.0)

    def test_invalid_temporal_pooling_is_rejected(self) -> None:
        config = load_config("configs/crnn_dads_full_augmented_g2.yaml")
        config["model"]["temporal_pooling"] = "maximum"
        with self.assertRaisesRegex(ValueError, "temporal_pooling"):
            _build_model(config)

    def test_internal_segment_kind_guard_rejects_hidden_drop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = pd.DataFrame(
                {
                    "split": ["test"] * 8,
                    "label": [0, 0, 0, 0, 1, 1, 1, 1],
                    "segment_kind": ["full"] * 8,
                }
            )
            manifest_path = root / "manifest.csv"
            manifest.to_csv(manifest_path, index=False)
            labels = manifest["label"].to_numpy(dtype=np.int64)
            baseline_probabilities = np.array(
                [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9]
            )
            candidate_probabilities = baseline_probabilities.copy()
            candidate_probabilities[-1] = 0.1
            paths = {}
            for role, probabilities in (
                ("baseline", baseline_probabilities),
                ("candidate", candidate_probabilities),
            ):
                label_path = root / f"{role}_labels.npy"
                probability_path = root / f"{role}_probabilities.npy"
                np.save(label_path, labels)
                np.save(probability_path, probabilities)
                paths[role] = {
                    "name": role,
                    "test_labels": label_path,
                    "test_probabilities": probability_path,
                }
            config = {
                "dads_manifest": manifest_path,
                "baseline": paths["baseline"],
                "candidate": paths["candidate"],
                "guardrails": {
                    "threshold": 0.5,
                    "internal_subgroup_checks": [
                        {
                            "split": "test",
                            "group_field": "segment_kind",
                            "group": "full",
                            "metric": "recall",
                            "max_drop": 0.10,
                        }
                    ],
                },
            }
            result = assess_internal_subgroups(config)
            self.assertFalse(result["passed"])
            self.assertEqual(result["checks"][0]["samples"], 4)

    def test_paired_multiway_bootstrap_requires_supported_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.array([0] * 20 + [1] * 20, dtype=np.int64)
            source_groups = [f"negative_{index}" for index in range(20)] + [
                f"positive_{index}" for index in range(20)
            ]
            baseline = pd.DataFrame(
                {
                    "label": labels,
                    "source_group": source_groups,
                    "uav_source": [""] * 20
                    + [f"uav_{index}" for index in range(20)],
                    "background_source": [f"background_{index}" for index in range(20)]
                    + [f"background_{index}" for index in range(20)],
                    "calibrated_probability": np.where(labels == 1, 0.1, 0.9),
                    "selected_prediction": np.where(labels == 1, 0, 1),
                }
            )
            candidate = baseline.copy()
            candidate["calibrated_probability"] = np.where(labels == 1, 0.9, 0.1)
            candidate["selected_prediction"] = labels
            baseline_path = root / "baseline.csv"
            candidate_path = root / "candidate.csv"
            baseline.to_csv(baseline_path, index=False)
            candidate.to_csv(candidate_path, index=False)
            settings = {
                "cluster_fields": ["uav_source", "background_source"],
                "method": "multiway_bayesian",
                "samples": 100,
                "seed": 42,
                "confidence": 0.95,
                "noninferiority_margins": {
                    metric: 0.02
                    for metric in (
                        "f1",
                        "balanced_accuracy",
                        "recall",
                        "specificity",
                        "auc",
                    )
                },
                "improvement_metrics": ["f1", "recall", "auc"],
                "improvement_lower_bound": 0.0,
            }
            config = {
                "baseline": {"predictions": {"holdout": baseline_path}},
                "candidate": {"predictions": {"holdout": candidate_path}},
                "guardrails": {
                    "external_promotion": {"paired_multiway_bootstrap": settings}
                },
            }
            self.assertTrue(paired_multiway_bootstrap(config)["passed"])
            config["candidate"]["predictions"]["holdout"] = baseline_path
            self.assertFalse(paired_multiway_bootstrap(config)["passed"])

    def test_point_and_bootstrap_improvement_must_be_same_metric(self) -> None:
        external = {
            "minimum_improvement_any": [
                {"metric": "f1", "passed": True},
                {"metric": "auc", "passed": False},
            ]
        }
        bootstrap = {
            "improvement_checks": [
                {"metric": "f1", "passed": False},
                {"metric": "auc", "passed": True},
            ]
        }
        self.assertEqual(
            supported_meaningful_improvement_metrics(external, bootstrap), []
        )
        bootstrap["improvement_checks"][0]["passed"] = True
        self.assertEqual(
            supported_meaningful_improvement_metrics(external, bootstrap), ["f1"]
        )

    def test_manifest_audio_hash_audit_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "sample.wav"
            audio.write_bytes(b"frozen-audio")
            expected = hashlib.sha256(audio.read_bytes()).hexdigest()
            manifest = root / "manifest.csv"
            pd.DataFrame({"path": [audio.as_posix()], "sha256": [expected]}).to_csv(
                manifest, index=False
            )
            result = verify_manifest_audio_hashes(manifest)
            self.assertTrue(result["verified"])
            self.assertEqual(result["unique_files"], 1)
            audio.write_bytes(b"drifted-audio")
            with self.assertRaisesRegex(ValueError, "Audio hash mismatch"):
                verify_manifest_audio_hashes(manifest)

    def test_calibration_csv_round_trip_passes_strict_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.array([0, 1, 0, 1, 0, 1, 1, 0], dtype=np.int64)
            logits = np.array(
                [-1.1234567, -0.2345678, 0.4567891, 0.8123456,
                 -0.7654321, 0.2456789, -0.1123456, 0.3345678],
                dtype=np.float32,
            )
            rows = pd.DataFrame(
                {
                    "label": labels,
                    "source_group": ["negative", "uav"] * 4,
                    "condition": ["background", "uav_only"] * 4,
                }
            )
            temperature = fit_temperature(labels, logits)
            raw = probabilities_from_logits(logits)
            calibrated = probabilities_from_logits(logits, temperature)
            threshold, feasible, _ = search_threshold(
                labels,
                calibrated,
                target_recall=0.5,
                target_specificity=0.5,
            )
            predictions = _prediction_frame(rows, logits, raw, calibrated, threshold)
            prediction_paths = {}
            prediction_hashes = {}
            for split in ("tune", "holdout"):
                path = root / f"{split}.csv"
                predictions.to_csv(path, index=False)
                prediction_paths[split] = path
                prediction_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()

            selected_metrics = probability_metrics(
                labels, calibrated, threshold, ece_bins=5
            )
            calibration = {
                "temperature": temperature,
                "selected_threshold": threshold,
                "target_recall": 0.5,
                "target_specificity": 0.5,
                "constraints_feasible_on_tune": feasible,
                "prediction_sha256": prediction_hashes,
                "tune_metrics": {"temperature_selected": selected_metrics},
                "holdout_metrics": {"temperature_selected": selected_metrics},
                "holdout_subgroups": subgroup_metrics(rows, calibrated, threshold),
            }
            config = {
                role: {
                    "require_prediction_sha256": True,
                    "predictions": prediction_paths,
                }
                for role in ("baseline", "candidate")
            }
            summary = _validate_prediction_content(
                config, {"baseline": calibration, "candidate": calibration}
            )
            self.assertTrue(summary["candidate"]["holdout"]["metrics_match"])
            round_tripped = pd.read_csv(prediction_paths["holdout"])
            self.assertTrue(
                np.allclose(
                    round_tripped["raw_probability"],
                    probabilities_from_logits(round_tripped["logit"].to_numpy()),
                    rtol=1e-10,
                    atol=1e-12,
                )
            )


if __name__ == "__main__":
    unittest.main()
