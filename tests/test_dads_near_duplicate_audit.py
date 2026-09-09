from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dads_crnn.audit_dads_near_duplicates import (
    _numbered_adjacency,
    lsh_cross_split_nearest,
)


class DadsNearDuplicateAuditTests(unittest.TestCase):
    def test_lsh_retrieves_identical_cross_split_feature(self) -> None:
        rng = np.random.default_rng(7)
        features = rng.normal(size=(6, 48)).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        features[5] = features[1]
        scores, references, comparisons = lsh_cross_split_nearest(
            features,
            np.asarray([0, 1, 2, 3]),
            np.asarray([4, 5]),
            tables=4,
            bits=8,
            seed=3,
        )
        self.assertEqual(int(references[1]), 1)
        self.assertAlmostEqual(float(scores[1]), 1.0, places=6)
        self.assertGreater(int(comparisons[1]), 0)

    def test_numbered_adjacency_marks_cross_split_pairs(self) -> None:
        sources = pd.DataFrame(
            {
                "source_id": ["a", "b", "c"],
                "source_path": ["drone-1.wav", "drone-2.wav", "drone-4.wav"],
                "split": ["train", "test", "train"],
                "label": [1, 1, 1],
            }
        )
        features = np.zeros((3, 48), dtype=np.float32)
        features[:, 0] = 1.0
        result = _numbered_adjacency(sources, features)
        self.assertEqual(len(result), 1)
        self.assertTrue(bool(result.iloc[0]["cross_split"]))
        self.assertEqual(result.iloc[0]["pair_role"], "test__train")
        self.assertAlmostEqual(result.iloc[0]["spectral_cosine_similarity"], 1.0)



if __name__ == "__main__":
    unittest.main()
