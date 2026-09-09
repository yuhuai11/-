from __future__ import annotations

from collections import Counter
import unittest

import pandas as pd

from dads_crnn.sampling import ClassDomainQuotaBatchSampler


class G7R9DomainQuotaSamplerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = pd.DataFrame(
            {
                "label": [1] * 5 + [0] * 7 + [0] * 4,
                "domain_bucket": (
                    ["positive:dads"] * 5
                    + ["negative:dads"] * 7
                    + ["negative:tau"] * 4
                ),
            }
        )

    def _sampler(self) -> ClassDomainQuotaBatchSampler:
        return ClassDomainQuotaBatchSampler(
            self.rows,
            domain_column="domain_bucket",
            batch_size=10,
            positive_per_batch=4,
            num_batches=5,
            positive_domain="positive:dads",
            negative_domain_fractions={
                "negative:dads": 0.8,
                "negative:tau": 0.2,
            },
            seed=42,
        )

    def test_exact_class_and_epoch_domain_quotas(self) -> None:
        sampler = self._sampler()
        observed_domains = Counter()
        for batch in sampler:
            selected = self.rows.iloc[batch]
            self.assertEqual(Counter(selected["label"]), Counter({0: 6, 1: 4}))
            observed_domains.update(selected["domain_bucket"])
        self.assertEqual(observed_domains["positive:dads"], 20)
        self.assertEqual(observed_domains["negative:dads"], 24)
        self.assertEqual(observed_domains["negative:tau"], 6)
        self.assertEqual(
            sampler.expected_label_probabilities,
            {0: 0.6, 1: 0.4},
        )

    def test_seed_and_epoch_are_deterministic(self) -> None:
        sampler = self._sampler()
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))

    def test_rejects_unconfigured_negative_domain(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit quota"):
            ClassDomainQuotaBatchSampler(
                self.rows,
                domain_column="domain_bucket",
                batch_size=10,
                positive_per_batch=4,
                num_batches=2,
                positive_domain="positive:dads",
                negative_domain_fractions={"negative:dads": 1.0},
                seed=42,
            )


if __name__ == "__main__":
    unittest.main()
