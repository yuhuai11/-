from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dads_crnn.probe_g18_prototypes import (
    aggregate_recording_embeddings,
    fit_prototype_space,
    l2_normalize,
    prototype_scores,
)


class G18PrototypeProbeTests(unittest.TestCase):
    def test_recording_embeddings_are_averaged(self) -> None:
        frame = pd.DataFrame(
            {
                "audio_sha256": ["a", "a", "b"],
                "model_id": ["A", "A", "B"],
                "target_index": [0, 0, -1],
                "is_known": [True, True, False],
            }
        )
        rows, values = aggregate_recording_embeddings(
            frame, np.asarray([[1.0, 0.0], [3.0, 2.0], [0.0, 4.0]])
        )
        self.assertEqual(rows["audio_sha256"].tolist(), ["a", "b"])
        np.testing.assert_allclose(values, [[2.0, 1.0], [0.0, 4.0]])

    def test_l2_normalization(self) -> None:
        values = l2_normalize(np.asarray([[3.0, 4.0], [0.0, 2.0]]))
        np.testing.assert_allclose(np.linalg.norm(values, axis=1), [1.0, 1.0])

    def test_prototype_space_separates_simple_classes(self) -> None:
        rng = np.random.default_rng(42)
        class_a = rng.normal([3, 0, 0, 0], 0.05, size=(10, 4))
        class_b = rng.normal([0, 3, 0, 0], 0.05, size=(10, 4))
        values = np.vstack([class_a, class_b])
        targets = np.asarray([0] * 10 + [1] * 10)
        pca, prototypes = fit_prototype_space(
            values, targets, components=2, classes=2, seed=42
        )
        scores, predictions = prototype_scores(values, pca, prototypes)
        np.testing.assert_array_equal(predictions, targets)
        self.assertTrue(np.all((scores >= 0.0) & (scores <= 1.0)))

    def test_zero_vector_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            l2_normalize(np.zeros((1, 3)))


if __name__ == "__main__":
    unittest.main()
