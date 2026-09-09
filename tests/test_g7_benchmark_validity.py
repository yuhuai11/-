from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dads_crnn.evaluate_g7_r6_external_suite import _promotion_decision, _report
from dads_crnn.g7_benchmark_validity import (
    build_validity_report,
    label_purity,
    majority_lookup,
)


def _manifest_rows(origin: str, label: int, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_role": ["train", "model_validation"],
            "dataset_origin": [origin, origin],
            "label": [label, label],
            "audio_sha256": [f"{prefix}a", f"{prefix}b"],
            "segment_sha256": [f"{prefix}s1", f"{prefix}s2"],
            "recording_group": [f"{prefix}r1", f"{prefix}r2"],
            "source_group": [f"{prefix}g1", f"{prefix}g2"],
        }
    )


class G7BenchmarkValidityTests(unittest.TestCase):
    def test_majority_origin_lookup_exposes_perfect_corpus_shortcut(self) -> None:
        development = pd.concat(
            [_manifest_rows("uav_corpus", 1, "u"), _manifest_rows("noise_corpus", 0, "n")],
            ignore_index=True,
        )
        benchmark = pd.DataFrame(
            {
                "dataset_origin": ["uav_corpus", "noise_corpus"],
                "label": [1, 0],
            }
        )
        result = majority_lookup(development, benchmark, "dataset_origin")
        self.assertEqual(result["coverage"], 1.0)
        self.assertEqual(result["accuracy_on_covered_rows"], 1.0)

    def test_label_purity_reports_mixed_groups(self) -> None:
        frame = pd.DataFrame(
            {"source_group": ["a", "a", "b"], "label": [0, 1, 1]}
        )
        result = label_purity(frame, "source_group")
        self.assertEqual(result["pure_label_groups"], 1)
        self.assertEqual(result["mixed_label_groups"], 1)

    def test_validity_report_blocks_independent_claim(self) -> None:
        development = pd.concat(
            [
                _manifest_rows("kielce_17_uav", 1, "u"),
                _manifest_rows("tau_urban_2022", 0, "n"),
            ],
            ignore_index=True,
        )
        calibration = _manifest_rows("cal", 0, "c")
        kielce_tau = pd.concat(
            [
                _manifest_rows("kielce_17_uav", 1, "x"),
                _manifest_rows("tau_urban_2022", 0, "y"),
            ],
            ignore_index=True,
        )
        g13 = pd.DataFrame(
            {"label": [1, 0], "source_group": ["p", "q"], "sha256": ["z1", "z2"]}
        )
        idmt = pd.DataFrame(
            {"session_id": ["i"], "recording_sha256": ["i1"]}
        )
        esc = pd.DataFrame(
            {"source_group": ["e"], "sha256": ["e1"], "cache_sha256": ["e2"]}
        )
        result = build_validity_report(
            development, calibration, kielce_tau, g13, idmt, esc
        )
        self.assertFalse(result["independent_final_claim_allowed"])
        self.assertTrue(
            result["corpus_label_confounding"][
                "kielce_tau_corpus_label_shortcut_detected"
            ]
        )
        self.assertFalse(result["hard_exact_metadata_overlap_detected"])

    def test_report_adds_source_group_operating_points(self) -> None:
        rows = pd.DataFrame(
            {
                "label": [1, 1, 0, 0],
                "recording_group": ["r1", "r2", "r3", "r4"],
                "source_group": ["positive", "positive", "negative", "negative"],
                "dataset_origin": ["uav", "uav", "noise", "noise"],
                "subtype": ["uav", "uav", "noise", "noise"],
            }
        )
        result = _report(rows, np.array([0.9, 0.8, 0.1, 0.2]), {"fixed": 0.5})
        self.assertEqual(result["source_groups"], 2)
        self.assertEqual(
            result["source_group_mean_operating_points"]["fixed"]["accuracy"], 1.0
        )
        self.assertEqual(
            result["source_group_macro_recording_operating_points"]["fixed"][
                "macro_balanced_accuracy"
            ],
            1.0,
        )

    def test_promotion_requires_all_cross_domain_non_regression_gates(self) -> None:
        def mixed(f1: float, fpr: float, auc: float = 0.9):
            return {
                "segment_operating_points": {
                    "fixed_0_5": {"f1": f1, "false_positive_rate": fpr}
                },
                "segment_ranking": {"roc_auc": auc},
                "recording_mean_ranking": {"roc_auc": auc},
                "source_group_macro_recording_operating_points": {
                    "fixed_0_5": {
                        "macro_balanced_accuracy": (f1 + 1.0 - fpr) / 2.0,
                        "macro_false_positive_rate": fpr,
                    }
                },
            }

        def negative(fpr: float):
            return {
                "segment_operating_points": {
                    "fixed_0_5": {"false_positive_rate": fpr}
                },
                "source_group_macro_recording_operating_points": {
                    "fixed_0_5": {"macro_false_positive_rate": fpr}
                },
            }

        reports = {
            "g7_r2_control": {
                "datasets": {
                    "kielce_tau_holdout": mixed(0.9, 0.02),
                    "g13_ddl_aerosonic": mixed(0.7, 0.1, 0.9),
                    "idmt_traffic": negative(0.2),
                    "esc50_fold5_guard": negative(0.1),
                }
            },
            "g7_r6_candidate": {
                "datasets": {
                    "kielce_tau_holdout": mixed(0.95, 0.01),
                    "g13_ddl_aerosonic": mixed(0.6, 0.1, 0.8),
                    "idmt_traffic": negative(0.3),
                    "esc50_fold5_guard": negative(0.05),
                }
            },
        }
        result = _promotion_decision(reports)
        self.assertFalse(result["promoted_on_reusable_benchmark"])
        self.assertIn(
            "g13_group_balanced_accuracy_not_lower", result["failed_criteria"]
        )
        self.assertIn("idmt_group_fpr_not_higher", result["failed_criteria"])


if __name__ == "__main__":
    unittest.main()
