from __future__ import annotations

import unittest

import numpy as np

from dads_crnn.analyze_g15_p4 import paired_bootstrap_ci


class G15P4AnalysisTests(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic_and_preserves_constant(self) -> None:
        values = np.full(20, 0.25)
        first = paired_bootstrap_ci(values, samples=100, seed=7)
        second = paired_bootstrap_ci(values, samples=100, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first, [0.25, 0.25])

    def test_paired_bootstrap_rejects_nonfinite_values(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap_ci(
                np.asarray([0.0, np.inf]), samples=10, seed=7
            )


if __name__ == "__main__":
    unittest.main()
