import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dads_crnn.evaluate_external import stratified_bootstrap_ci
from dads_crnn.external_data import external_groups
from dads_crnn.calibrate_ood import fit_temperature, negative_log_likelihood, search_threshold
from dads_crnn.prepare_val_ood import _source_disjoint_split
from dads_crnn.augmentation import WaveformAugmenter
from dads_crnn.sampling import ClassSourceBalancedSampler
from dads_crnn.audit_experiment_delta import ALLOWED_CHANGES, config_differences
from dads_crnn.config import load_config
from dads_crnn.gate_candidate import assess_external, assess_internal
from dads_crnn.train import _build_feature_extractor, _build_model


class ExternalEvaluationTests(unittest.TestCase):
    def test_unseen_group_parser(self) -> None:
        source, condition = external_groups(
            "unseen", Path("DJI_Mini_2_paperdist_00001_T08_00001.wav"), 1
        )
        self.assertEqual(source, "DJI_Mini_2")
        self.assertEqual(condition, "T08")

    def test_real_world_group_parser(self) -> None:
        self.assertEqual(
            external_groups("real_world", Path("drone+helicopter_25.wav"), 1),
            ("drone", "helicopter"),
        )
        self.assertEqual(
            external_groups("real_world", Path("traffic_01.wav"), 0),
            ("background", "traffic"),
        )

    def test_bootstrap_ci_contains_perfect_score(self) -> None:
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        probabilities = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
        confidence = stratified_bootstrap_ci(labels, probabilities, 0.5, samples=20, seed=42)
        self.assertEqual(confidence["f1"], {"low": 1.0, "high": 1.0})

    def test_ood_split_keeps_sources_disjoint(self) -> None:
        frame = pd.DataFrame(
            {
                "label": [0, 0, 0, 0, 1, 1, 1, 1],
                "source_group": ["b1", "b1", "b2", "b2", "u1", "u1", "u2", "u2"],
            }
        )
        result = _source_disjoint_split(frame, tune_fraction=0.5, seed=42)
        tune = set(result.loc[result["ood_split"] == "tune", "source_group"])
        holdout = set(result.loc[result["ood_split"] == "holdout", "source_group"])
        self.assertFalse(tune & holdout)
        self.assertEqual(set(result["ood_split"]), {"tune", "holdout"})

    def test_temperature_reduces_nll(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        logits = np.array([-12.0, -8.0, 2.0, -2.0, 8.0, 12.0])
        temperature = fit_temperature(labels, logits)
        self.assertLess(
            negative_log_likelihood(labels, logits, temperature),
            negative_log_likelihood(labels, logits, 1.0),
        )

    def test_threshold_search_enforces_constraints(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        probabilities = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        threshold, feasible, _ = search_threshold(
            labels, probabilities, target_recall=0.8, target_specificity=0.9
        )
        self.assertTrue(feasible)
        self.assertGreaterEqual(threshold, 0.3)
        self.assertLessEqual(threshold, 0.7)

    def test_background_mix_hits_target_snr(self) -> None:
        config = {
            "positive_mix_probability": 1.0,
            "mix_snr_db": [5.0],
            "mix_snr_weights": [1.0],
            "frequency_response_probability": 0.0,
            "frequency_response_anchors": 8,
            "frequency_response_std_db": 3.0,
            "frequency_response_limit_db": 6.0,
            "reverb_probability": 0.0,
            "reverb_delay_ms": [12.0, 80.0],
            "reverb_gain": [0.05, 0.25],
            "colored_noise_probability": 0.0,
            "colored_noise_snr_db": [10.0, 30.0],
            "time_shift_probability": 0.0,
            "max_time_shift_ms": 100.0,
        }
        time = np.arange(16000) / 16000
        signal = np.sin(2 * np.pi * 400 * time).astype(np.float32)
        background = np.sin(2 * np.pi * 900 * time).astype(np.float32)
        augmenter = WaveformAugmenter(config, sample_rate=16000, seed=42)
        output, metadata = augmenter.apply(signal, 1, lambda: background)
        self.assertTrue(metadata["background_mix"])
        self.assertAlmostEqual(metadata["achieved_snr_db"], 5.0, places=5)
        self.assertTrue(np.isfinite(output).all())
        self.assertLessEqual(float(np.max(np.abs(output))), 1.0)

    def test_class_source_sampler_balances_hierarchy(self) -> None:
        rows = pd.DataFrame(
            {
                "label": [0, 0, 0, 0, 1, 1],
                "source_path": ["long", "long", "long", "short", "u1", "u2"],
            }
        )
        sampler = ClassSourceBalancedSampler(
            rows, source_column="source_path", num_samples=100, seed=42
        )
        weights = sampler.weights.numpy()
        self.assertAlmostEqual(float(weights[rows.label == 0].sum()), 0.5)
        self.assertAlmostEqual(float(weights[rows.label == 1].sum()), 0.5)
        self.assertAlmostEqual(float(weights[:3].sum()), float(weights[3]))
        first = list(iter(sampler))
        repeated = list(
            iter(
                ClassSourceBalancedSampler(
                    rows, source_column="source_path", num_samples=100, seed=42
                )
            )
        )
        self.assertEqual(first, repeated)

    def test_g4_changes_only_approved_snr_distribution(self) -> None:
        baseline = load_config("configs/crnn_dads_full_augmented_g2.yaml")
        candidate = load_config("configs/crnn_dads_full_augmented_g4_low_snr.yaml")
        self.assertEqual(set(config_differences(baseline, candidate)), ALLOWED_CHANGES)
        self.assertNotIn("sampling", candidate["train"])

    def test_negative_label_is_never_background_mixed(self) -> None:
        config = load_config("configs/crnn_dads_full_augmented_g4_low_snr.yaml")["train"][
            "augmentation"
        ]
        augmenter = WaveformAugmenter(config, sample_rate=16000, seed=42)
        signal = np.ones(16000, dtype=np.float32)
        _, metadata = augmenter.apply(signal, 0, lambda: signal)
        self.assertFalse(metadata["background_mix"])
        self.assertIsNone(metadata["target_snr_db"])

    def test_minus10_background_mix_hits_target(self) -> None:
        config = load_config("configs/crnn_dads_full_augmented_g4_low_snr.yaml")["train"][
            "augmentation"
        ]
        augmenter = WaveformAugmenter(config, sample_rate=16000, seed=42)
        time = np.arange(16000) / 16000
        signal = np.sin(2 * np.pi * 400 * time).astype(np.float32)
        background = np.sin(2 * np.pi * 900 * time).astype(np.float32)
        _, achieved = augmenter._mix_background(signal, background, -10.0)
        self.assertAlmostEqual(achieved, -10.0, places=5)

    def test_augmentation_rejects_invalid_probability(self) -> None:
        config = dict(
            load_config("configs/crnn_dads_full_augmented_g4_low_snr.yaml")["train"][
                "augmentation"
            ]
        )
        config["positive_mix_probability"] = 1.1
        with self.assertRaises(ValueError):
            WaveformAugmenter(config, sample_rate=16000, seed=42)

    def test_candidate_gate_requires_internal_noninferiority(self) -> None:
        guardrails = load_config("configs/g4_low_snr_guardrails.yaml")["guardrails"]

        def metrics(f1: float) -> dict:
            row = {
                "threshold": 0.5,
                "f1": f1,
                "auc": 0.9998,
                "recall": 0.995,
                "specificity": 0.995,
            }
            return {"val_threshold_metrics": [row], "threshold_metrics": [row]}

        self.assertTrue(assess_internal(metrics(0.995), metrics(0.994), guardrails)["passed"])
        self.assertFalse(assess_internal(metrics(0.995), metrics(0.980), guardrails)["passed"])

    def test_external_gate_requires_pareto_and_minus10_gain(self) -> None:
        guardrails = load_config("configs/g4_low_snr_guardrails.yaml")["guardrails"]
        baseline_values = {
            "f1": 0.71,
            "balanced_accuracy": 0.71,
            "recall": 0.72,
            "specificity": 0.70,
            "auc": 0.76,
        }
        candidate_values = dict(baseline_values, recall=0.75, auc=0.78, f1=0.73)
        baseline = {
            "constraints_feasible_on_tune": False,
            "holdout_metrics": {"temperature_selected": baseline_values},
            "holdout_subgroups": [
                {"group_field": "condition", "group": "snr_-10_db", "recall": 0.63}
            ],
        }
        candidate = {
            "constraints_feasible_on_tune": False,
            "holdout_metrics": {"temperature_selected": candidate_values},
            "holdout_subgroups": [
                {"group_field": "condition", "group": "snr_-10_db", "recall": 0.67}
            ],
        }
        self.assertTrue(assess_external(baseline, candidate, guardrails)["passed"])
        self.assertFalse(assess_external(baseline, baseline, guardrails)["passed"])

    def test_g5_mfcc64_preserves_shape_and_model_capacity(self) -> None:
        baseline = load_config("configs/crnn_dads_full_augmented_g2.yaml")
        candidate = load_config("configs/crnn_dads_full_augmented_g5_mfcc64.yaml")
        waveform = np.zeros((2, 16000), dtype=np.float32)
        import torch

        baseline_feature = _build_feature_extractor(baseline)
        candidate_feature = _build_feature_extractor(candidate)
        baseline_output = baseline_feature(torch.from_numpy(waveform))
        candidate_output = candidate_feature(torch.from_numpy(waveform))
        self.assertEqual(tuple(baseline_output.shape), (2, 1, 64, 101))
        self.assertEqual(tuple(candidate_output.shape), tuple(baseline_output.shape))
        self.assertTrue(torch.equal(baseline_feature.window, candidate_feature.window))
        baseline_parameters = sum(p.numel() for p in _build_model(baseline).parameters())
        candidate_parameters = sum(p.numel() for p in _build_model(candidate).parameters())
        self.assertEqual(baseline_parameters, 1_012_193)
        self.assertEqual(candidate_parameters, baseline_parameters)

    def test_legacy_mfcc_default_remains_hamming(self) -> None:
        config = load_config("configs/resnet10_cbam_dads_full.yaml")
        feature = _build_feature_extractor(config)
        self.assertEqual(feature.window_type, "hamming")

    def test_g5_subgroup_guard_rejects_hidden_condition_drop(self) -> None:
        guardrails = load_config("configs/g5_mfcc64_guardrails.yaml")["guardrails"]
        values = {
            "f1": 0.71,
            "balanced_accuracy": 0.71,
            "recall": 0.72,
            "specificity": 0.70,
            "auc": 0.76,
        }
        conditions = [
            "uav_only",
            "snr_+10_db",
            "snr_+5_db",
            "snr_+0_db",
            "snr_-5_db",
            "snr_-10_db",
            "snr_-15_db",
        ]
        baseline_groups = [
            {"group_field": "condition", "group": group, "recall": 0.70}
            for group in conditions
        ] + [
            {
                "group_field": "condition",
                "group": "background_only",
                "specificity": 0.70,
            }
        ]
        candidate_groups = [dict(row) for row in baseline_groups]
        candidate_groups[0]["recall"] = 0.66
        baseline = {
            "constraints_feasible_on_tune": False,
            "holdout_metrics": {"temperature_selected": values},
            "holdout_subgroups": baseline_groups,
        }
        candidate = {
            "constraints_feasible_on_tune": False,
            "holdout_metrics": {
                "temperature_selected": dict(values, f1=0.74, recall=0.75, auc=0.78)
            },
            "holdout_subgroups": candidate_groups,
        }
        result = assess_external(baseline, candidate, guardrails)
        self.assertFalse(result["passed"])
        self.assertFalse(result["subgroup_checks_passed"])


if __name__ == "__main__":
    unittest.main()
