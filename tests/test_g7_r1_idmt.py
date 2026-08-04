from __future__ import annotations

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from dads_crnn.calibrate_ood import fit_temperature, probabilities_from_logits
from dads_crnn.data_firewall import file_sha256
from dads_crnn.evaluate_g7_idmt_r0 import validate_development_manifest
from dads_crnn.evaluate_g7_r1_idmt import (
    _aligned_baseline_predictions,
    _paired_session_bootstrap,
    _subset_fpr,
    _validated_guard_score_mappings,
)
from dads_crnn.evaluate_low_fpr import threshold_at_target_fpr
from dads_crnn.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class G7R1IdmtTests(unittest.TestCase):
    def test_formal_r0_inventory_reconstructs_exact_baseline(self) -> None:
        config = load_config(ROOT / "configs/g7_idmt_r0_gpu.yaml")
        rows = validate_development_manifest(
            ROOT
            / "artifacts/g7_improvement/stage_b/development_segments_dedup.csv",
            config,
        )
        probabilities, strict, sensitivity = _aligned_baseline_predictions(
            rows,
            ROOT
            / "artifacts/g7_improvement/stage_c_r0_gpu/evaluation/predictions.csv",
            temperature=2.258312527893372,
            strict_threshold=0.7236399840238329,
            sensitivity_threshold=0.5085329108689571,
        )
        self.assertEqual(probabilities.shape, (29346,))
        self.assertEqual(int(strict.sum()), 1960)
        self.assertEqual(int(sensitivity.sum()), 5126)
        self.assertAlmostEqual(float(strict.mean()), 0.06678934096640088)
        self.assertAlmostEqual(
            _subset_fpr(
                rows,
                strict,
                microphone="ME",
                traffic_content="vehicle_passing",
            )["fpr"],
            0.20804331013147717,
        )
        paired = _paired_session_bootstrap(
            rows, strict, strict, samples=20, seed=7
        )
        self.assertEqual(paired["candidate_minus_baseline_fpr_delta"], 0.0)
        self.assertEqual(paired["one_sided_95pct_upper"], 0.0)

    def test_guard_tune_mapping_is_recomputed_and_tampering_fails(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            regression = root / "regression"
            prediction_dir = regression / "predictions"
            prediction_dir.mkdir(parents=True)
            guard_path = regression / "guard_metrics.json"
            tune_manifest_path = root / "tune.csv"
            labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
            logits = np.asarray([-4.0, -2.0, 0.5, -0.5, 2.0, 4.0])
            sha = [f"{index:064x}" for index in range(len(labels))]
            manifest = pd.DataFrame({"sha256": sha, "label": labels})
            manifest.to_csv(tune_manifest_path, index=False)
            tune = manifest.copy()
            tune["r1_logit"] = logits
            tune_path = prediction_dir / "val_ood_tune.csv"
            tune.to_csv(tune_path, index=False)
            holdout_path = prediction_dir / "val_ood_holdout.csv"
            g9_path = prediction_dir / "g9_guard.csv"
            pd.DataFrame({"value": [1]}).to_csv(holdout_path, index=False)
            pd.DataFrame({"value": [1]}).to_csv(g9_path, index=False)

            temperature = fit_temperature(labels, logits)
            probabilities = probabilities_from_logits(logits, temperature)
            strict = threshold_at_target_fpr(
                probabilities[labels == 0], 0.01
            )
            sensitivity = threshold_at_target_fpr(
                probabilities[labels == 0], 0.05
            )
            score_protocol = {
                "primary_comparison": "per_model_same_tune_calibration",
                "temperature_fit": "tune_all_labels_nll",
                "threshold_selection": "negative_only_conservative_empirical_quantile",
                "target_fprs": [0.01, 0.05],
                "calibration_manifest_role": "val_ood_tune",
                "holdout_used": False,
                "idmt_used": False,
                "baseline_g7_mapping": {
                    "temperature": 2.258312527893372,
                    "strict_threshold": 0.7236399840238329,
                    "sensitivity_threshold": 0.5085329108689571,
                },
                "fixed_g7_mapping_role": "deployment_compatibility_diagnostic_only",
            }
            protocol = {
                "contract": {"score_protocol": score_protocol},
                "inputs": {"val_ood_tune_manifest": "tune.csv"},
            }
            outputs = {"guard_evaluation": guard_path}
            guard = {
                "score_protocol": score_protocol,
                "candidate_calibration": {
                    "temperature": temperature,
                    "strict_target_fpr_1": strict,
                    "sensitivity_target_fpr_5": sensitivity,
                    "holdout_used_for_temperature_or_threshold": False,
                    "idmt_used_for_temperature_or_threshold": False,
                },
                "predictions": {
                    "val_ood_tune": {
                        "path": tune_path.relative_to(root).as_posix(),
                        "sha256": file_sha256(tune_path),
                    },
                    "val_ood_holdout": {
                        "path": holdout_path.relative_to(root).as_posix(),
                        "sha256": file_sha256(holdout_path),
                    },
                    "g9_guard": {
                        "path": g9_path.relative_to(root).as_posix(),
                        "sha256": file_sha256(g9_path),
                    },
                },
            }
            observed = _validated_guard_score_mappings(
                root, protocol, outputs, guard
            )
            self.assertEqual(observed["candidate_r1"]["temperature"], temperature)
            self.assertEqual(
                observed["candidate_r1"]["strict_threshold"],
                strict["threshold"],
            )

            tampered = copy.deepcopy(guard)
            tampered["candidate_calibration"]["strict_target_fpr_1"][
                "threshold"
            ] += 0.01
            with self.assertRaisesRegex(ValueError, "cannot be reproduced"):
                _validated_guard_score_mappings(
                    root, protocol, outputs, tampered
                )


if __name__ == "__main__":
    unittest.main()
