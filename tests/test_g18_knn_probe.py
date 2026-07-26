from __future__ import annotations

import unittest

import numpy as np

from dads_crnn.probe_g18_knn import (
    class_conditional_knn_scores,
    fit_knn_space,
)


class G18KnnProbeTests(unittest.TestCase):
    def test_fit_space_requires_k_per_class(self) -> None:
        values = np.eye(4)
        targets = np.asarray([0, 0, 1, 1])
        with self.assertRaises(ValueError):
            fit_knn_space(
                values,
                targets,
                components=2,
                classes=2,
                neighbors=3,
                seed=42,
            )

    def test_class_conditional_knn_respects_predicted_class(self) -> None:
        rng = np.random.default_rng(42)
        class_a = rng.normal([3, 0, 0, 0], 0.03, size=(10, 4))
        class_b = rng.normal([0, 3, 0, 0], 0.03, size=(10, 4))
        values = np.vstack([class_a, class_b])
        targets = np.asarray([0] * 10 + [1] * 10)
        pca, references, reference_targets = fit_knn_space(
            values,
            targets,
            components=2,
            classes=2,
            neighbors=5,
            seed=42,
        )
        close_score = class_conditional_knn_scores(
            class_a[:1],
            np.asarray([0]),
            pca=pca,
            references=references,
            reference_targets=reference_targets,
            neighbors=5,
        )[0]
        wrong_class_score = class_conditional_knn_scores(
            class_a[:1],
            np.asarray([1]),
            pca=pca,
            references=references,
            reference_targets=reference_targets,
            neighbors=5,
        )[0]
        self.assertGreater(close_score, wrong_class_score)
        self.assertGreaterEqual(close_score, 0.0)
        self.assertLessEqual(close_score, 1.0)


if __name__ == "__main__":
    unittest.main()
