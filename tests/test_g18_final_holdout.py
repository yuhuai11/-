from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dads_crnn.evaluate_g18_final_holdout import (
    AGGREGATE_METRICS,
    SEEDS,
    score_oe_space,
    summarize_seed,
)


class G18FinalHoldoutTests(unittest.TestCase):
    def test_seed_order_and_metrics_are_frozen(self) -> None:
        self.assertEqual(SEEDS, (42, 43, 44))
        self.assertEqual(len(AGGREGATE_METRICS), len(set(AGGREGATE_METRICS)))

    def test_oe_space_scoring(self) -> None:
        space = {
            "pca_mean": np.zeros(2),
            "pca_components": np.eye(2),
            "scaler_mean": np.zeros(2),
            "scaler_scale": np.ones(2),
            "logistic_coef": np.asarray([[1.0, 0.0]]),
            "logistic_intercept": np.asarray([0.0]),
        }
        scores = score_oe_space(np.asarray([[-2.0, 0.0], [2.0, 0.0]]), space)
        self.assertLess(scores[0], 0.5)
        self.assertGreater(scores[1], 0.5)

    def test_full_chain_requires_detection_acceptance_and_correct_class(self) -> None:
        known = pd.DataFrame(
            {
                "model_id": ["A", "B"],
                "target_index": [0, 1],
            }
        )
        unknown = pd.DataFrame(
            {
                "model_id": ["X", "Y"],
                "target_index": [-1, -1],
            }
        )
        metrics, _, _ = summarize_seed(
            known_frame=known,
            unknown_frame=unknown,
            known_logits=np.asarray([[3.0, 0.0], [3.0, 0.0]]),
            unknown_logits=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            known_scores=np.asarray([0.9, 0.9]),
            unknown_scores=np.asarray([0.1, 0.1]),
            threshold=0.5,
            known_g7_probability=np.asarray([0.9, 0.4]),
            unknown_g7_probability=np.asarray([0.9, 0.4]),
            strict_threshold=0.7,
            balanced_threshold=0.3,
            known_models=["A", "B"],
        )
        self.assertEqual(metrics["known_closed_set_accuracy"], 0.5)
        self.assertEqual(metrics["unknown_recall"], 1.0)
        self.assertEqual(metrics["strict_known_full_chain_accuracy"], 0.5)
        self.assertEqual(metrics["strict_unknown_full_chain_recall"], 0.5)
        self.assertEqual(metrics["balanced_known_full_chain_accuracy"], 0.5)
        self.assertEqual(metrics["balanced_unknown_full_chain_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
