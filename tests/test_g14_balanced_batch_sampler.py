from __future__ import annotations

from collections import Counter
import unittest

import pandas as pd

from dads_crnn.sampling import ClassSourceBalancedBatchSampler


class G14BalancedBatchSamplerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = pd.DataFrame(
            {
                "label": [0] * 6 + [1] * 6,
                "source_group": ["n1"] * 2 + ["n2"] * 4 + ["p1"] * 3 + ["p2"] * 3,
            }
        )

    def test_every_batch_has_exact_class_balance(self) -> None:
        sampler = ClassSourceBalancedBatchSampler(
            self.rows,
            source_column="source_group",
            batch_size=4,
            num_batches=3,
            source_weight_exponent=0.5,
            seed=42,
        )
        observed = Counter()
        for batch in sampler:
            labels = self.rows.iloc[batch]["label"].tolist()
            self.assertEqual(Counter(labels), Counter({0: 2, 1: 2}))
            observed.update(self.rows.iloc[batch]["source_group"].tolist())
        planned = {
            source: count
            for values in sampler.source_draws.values()
            for source, count in values.items()
        }
        self.assertEqual(dict(observed), planned)

    def test_seed_and_epoch_control_order(self) -> None:
        sampler = ClassSourceBalancedBatchSampler(
            self.rows,
            source_column="source_group",
            batch_size=4,
            num_batches=3,
            source_weight_exponent=0.5,
            seed=42,
        )
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))


if __name__ == "__main__":
    unittest.main()
