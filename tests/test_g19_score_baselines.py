from __future__ import annotations

import unittest

import numpy as np

from dads_crnn.probe_g19_score_baselines import (
    knownness_scores,
    threshold_from_known_only,
)


class G19KnownOnlyScoreTests(unittest.TestCase):
    def test_scores_have_common_higher_is_known_orientation(self) -> None:
        logits = np.asarray([[5.0, 0.0], [0.2, 0.1]])
        for method in ("msp", "maximum_logit", "energy"):
            scores = knownness_scores(logits, method)
            self.assertGreater(scores[0], scores[1])

    def test_known_only_threshold_honors_acceptance_constraint(self) -> None:
        scores = np.arange(10, dtype=np.float64)
        result = threshold_from_known_only(scores, 0.9)
        self.assertEqual(result["threshold"], 1.0)
        self.assertEqual(result["actual_acceptances"], 9)
        self.assertEqual(result["actual_acceptance_rate"], 0.9)

    def test_ties_are_conservative_for_known_acceptance(self) -> None:
        result = threshold_from_known_only(np.asarray([0.1, 0.1, 0.2]), 2 / 3)
        self.assertEqual(result["actual_acceptances"], 3)


if __name__ == "__main__":
    unittest.main()
