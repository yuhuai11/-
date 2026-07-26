from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dads_crnn.evaluate_low_fpr import evaluate_low_fpr
from dads_crnn.evaluate_source_robust_fpr import (
    evaluate_frozen_source_robust,
    fit_source_robust_calibration,
    reject_locked_input_path,
    source_block_threshold,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prediction_rows(
    split: str,
    negative_groups: dict[str, list[float]],
    positive_groups: dict[str, list[float]],
) -> pd.DataFrame:
    rows = []
    index = 0
    for source, probabilities in negative_groups.items():
        for probability in probabilities:
            key = f"{split}-negative-{index}"
            rows.append(
                {
                    "dataset": "val_ood",
                    "path": f"/tmp/val_ood/{key}.wav",
                    "sha256": _digest(key),
                    "label": 0,
                    "source_group": source,
                    "uav_source": "",
                    "background_source": source,
                    "condition": "background_only",
                    "ood_split": split,
                    "calibrated_probability": probability,
                }
            )
            index += 1
    for source, probabilities in positive_groups.items():
        for probability in probabilities:
            key = f"{split}-positive-{index}"
            rows.append(
                {
                    "dataset": "val_ood",
                    "path": f"/tmp/val_ood/{key}.wav",
                    "sha256": _digest(key),
                    "label": 1,
                    "source_group": source,
                    "uav_source": source,
                    "background_source": "",
                    "condition": "uav_only",
                    "ood_split": split,
                    "calibrated_probability": probability,
                }
            )
            index += 1
    return pd.DataFrame(rows)


def write_bound_inputs(root: Path, name: str, rows: pd.DataFrame) -> tuple[Path, Path]:
    prediction_path = root / f"{name}_predictions.csv"
    manifest_path = root / f"{name}_manifest.csv"
    rows.to_csv(prediction_path, index=False)
    rows.drop(columns=["calibrated_probability"]).to_csv(manifest_path, index=False)
    return prediction_path, manifest_path


class SourceBlockThresholdTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = prediction_rows(
            "tune",
            {"a": [0.1, 0.2, 0.3, 0.4], "b": [0.1, 0.2, 0.8, 0.9]},
            {"u1": [0.6, 0.7], "u2": [0.8, 0.9]},
        )

    def test_caps_every_tune_background_source(self) -> None:
        result = source_block_threshold(
            self.rows, 0.25, source_miscoverage_delta=0.34
        )
        self.assertEqual(result["order_statistic_rank"], 2)
        self.assertEqual(result["threshold"], float(np.nextafter(0.8, np.inf)))
        for source in result["per_source"]:
            self.assertLessEqual(
                source["global_false_positives"], source["allowed_false_positives"]
            )
        self.assertGreater(result["threshold"], result["pooled_threshold"])

    def test_ties_and_zero_budget_remain_conservative(self) -> None:
        rows = prediction_rows(
            "tune",
            {"a": [0.9, 0.9, 0.9], "b": [0.1, 0.2, 0.3]},
            {"u": [0.5, 0.8]},
        )
        result = source_block_threshold(rows, 0.25, source_miscoverage_delta=0.34)
        self.assertGreater(result["threshold"], 0.9)
        self.assertEqual(result["tune_pooled_fpr"], 0.0)

    def test_score_one_can_produce_threshold_above_one(self) -> None:
        rows = prediction_rows(
            "tune",
            {"a": [0.1, 1.0], "b": [0.2, 0.3]},
            {"u": [0.8, 0.9]},
        )
        result = source_block_threshold(rows, 0.01, source_miscoverage_delta=0.34)
        self.assertGreater(result["threshold"], 1.0)

    def test_positive_scores_and_row_order_do_not_change_threshold(self) -> None:
        expected = source_block_threshold(
            self.rows, 0.25, source_miscoverage_delta=0.34
        )["threshold"]
        changed = self.rows.copy()
        changed.loc[changed["label"] == 1, "calibrated_probability"] = [0.01, 0.02, 0.03, 0.04]
        changed = changed.sample(frac=1.0, random_state=7).reset_index(drop=True)
        observed = source_block_threshold(
            changed, 0.25, source_miscoverage_delta=0.34
        )["threshold"]
        self.assertEqual(expected, observed)

    def test_insufficient_source_groups_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "insufficient_source_groups"):
            source_block_threshold(self.rows, 0.25, source_miscoverage_delta=0.05)

    def test_blank_negative_source_is_rejected(self) -> None:
        rows = self.rows.copy()
        rows.loc[rows["label"] == 0, "background_source"] = " "
        with self.assertRaisesRegex(ValueError, "background_source"):
            source_block_threshold(rows, 0.25, source_miscoverage_delta=0.34)


class SourceRobustArtifactTests(unittest.TestCase):
    def test_fit_binds_manifest_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = prediction_rows(
                "tune",
                {"a": [0.1, 0.2, 0.3, 0.4], "b": [0.1, 0.2, 0.8, 0.9]},
                {"u1": [0.6, 0.7], "u2": [0.8, 0.9]},
            )
            predictions, manifest = write_bound_inputs(root, "tune", rows)
            output = root / "thresholds.json"
            result = fit_source_robust_calibration(
                predictions,
                manifest,
                output,
                experiment="test",
                target_fprs=[0.25],
                source_miscoverage_delta=0.34,
                holdout_fpr_caps=[0.5],
                minimum_holdout_tprs=[0.1],
                minimum_tpr_retention=0.5,
            )
            self.assertTrue(output.is_file())
            self.assertFalse(result["protocol"]["holdout_used_for_threshold_selection"])
            self.assertEqual(result["input_audit"]["manifest"]["rows"], len(rows))
            with self.assertRaises(FileExistsError):
                fit_source_robust_calibration(
                    predictions,
                    manifest,
                    output,
                    experiment="test",
                    target_fprs=[0.25],
                    source_miscoverage_delta=0.34,
                    holdout_fpr_caps=[0.5],
                    minimum_holdout_tprs=[0.1],
                    minimum_tpr_retention=0.5,
                )

    def test_nonintegral_label_is_rejected_before_cast(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = prediction_rows(
                "tune", {"a": [0.1, 0.2], "b": [0.2, 0.3]}, {"u": [0.8, 0.9]}
            )
            rows["label"] = rows["label"].astype(np.float64)
            rows.loc[0, "label"] = 0.5
            predictions, manifest = write_bound_inputs(root, "tune", rows)
            with self.assertRaisesRegex(ValueError, "exactly binary"):
                fit_source_robust_calibration(
                    predictions,
                    manifest,
                    root / "out.json",
                    experiment="test",
                    target_fprs=[0.25],
                    source_miscoverage_delta=0.34,
                    holdout_fpr_caps=[0.5],
                    minimum_holdout_tprs=[0.1],
                    minimum_tpr_retention=0.5,
                )

    def test_locked_name_variants_and_symlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "UNSEEN.csv",
                "real_world.csv",
                "real-world.csv",
                "realworld.csv",
                "real world.csv",
                "external_confirmation_v2.csv",
                "g13_external_confirmation.csv",
            ):
                with self.assertRaises(ValueError):
                    reject_locked_input_path(root / name)
            locked = root / "real_world_predictions.csv"
            locked.write_text("x\n", encoding="utf-8")
            alias = root / "safe.csv"
            alias.symlink_to(locked)
            with self.assertRaises(ValueError):
                reject_locked_input_path(alias)

    def test_frozen_end_to_end_preserves_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tune = prediction_rows(
                "tune",
                {"ta": [0.1, 0.2, 0.3, 0.4], "tb": [0.1, 0.2, 0.8, 0.9]},
                {"tu1": [0.6, 0.7], "tu2": [0.8, 0.9]},
            )
            holdout = prediction_rows(
                "holdout",
                {"ha": [0.1, 0.2, 0.3, 0.4], "hb": [0.2, 0.3, 0.4, 0.5]},
                {"hu1": [0.6, 0.7], "hu2": [0.8, 0.9]},
            )
            tune_predictions, tune_manifest = write_bound_inputs(root, "tune", tune)
            holdout_predictions, holdout_manifest = write_bound_inputs(root, "holdout", holdout)
            frozen = root / "frozen.json"
            fit_source_robust_calibration(
                tune_predictions,
                tune_manifest,
                frozen,
                experiment="test",
                target_fprs=[0.25],
                source_miscoverage_delta=0.34,
                holdout_fpr_caps=[0.5],
                minimum_holdout_tprs=[0.1],
                minimum_tpr_retention=0.5,
            )
            reference_dir = root / "reference"
            reference = evaluate_low_fpr(
                tune_predictions,
                holdout_predictions,
                reference_dir,
                experiment="reference",
                target_fprs=[0.25],
                bootstrap_samples=20,
                bootstrap_seed=42,
            )
            result = evaluate_frozen_source_robust(
                tune_predictions,
                tune_manifest,
                holdout_predictions,
                holdout_manifest,
                frozen,
                reference_dir / "metrics.json",
                reference_dir / "metrics.json",
                root / "evaluation",
                experiment="g8",
                bootstrap_samples=20,
                bootstrap_seed=42,
            )
            self.assertTrue(result["ranking_consistency"]["holdout"]["roc_auc"]["passed"])
            self.assertEqual(
                result["ranking_metrics"]["holdout"], reference["ranking_metrics"]["holdout"]
            )
            self.assertTrue((root / "evaluation" / "metrics.json").is_file())
            self.assertTrue((root / "evaluation" / "operating_points.csv").is_file())


if __name__ == "__main__":
    unittest.main()
