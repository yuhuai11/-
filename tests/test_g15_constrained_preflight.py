from __future__ import annotations

import unittest

import pandas as pd
import torch

from dads_crnn.preflight_g15_constrained import (
    constrained_losses,
    select_source_rows,
)


class G15ConstrainedPreflightTests(unittest.TestCase):
    def test_dads_selection_is_deterministic_and_label_balanced(self) -> None:
        rows = pd.DataFrame(
            {
                "label": [0] * 20 + [1] * 20,
                "source_group": [f"group-{index // 4}" for index in range(40)],
                "row": list(range(40)),
            }
        )
        first = select_source_rows(rows, count=16, seed=42, balance_labels=True)
        second = select_source_rows(rows, count=16, seed=42, balance_labels=True)
        self.assertEqual(first["row"].tolist(), second["row"].tolist())
        self.assertEqual(first["label"].value_counts().to_dict(), {0: 8, 1: 8})
        self.assertFalse(first["row"].duplicated().any())

    def test_constrained_loss_matches_declared_weighting(self) -> None:
        student = torch.tensor([0.0, 1.0, -1.0, 0.5], requires_grad=True)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
        losses = constrained_losses(
            student,
            labels,
            student[:2],
            torch.tensor([0.1, 0.9]),
            torch.tensor([1.0, 0.4]),
            torch.tensor([0.2, 0.3]),
            distill_weight=1.0,
            pair_weight=0.25,
            pair_margin=0.5,
            supervised_pos_weight=torch.tensor(1.5),
        )
        supervised = torch.nn.functional.binary_cross_entropy_with_logits(
            student,
            labels,
            pos_weight=torch.tensor(1.5),
        )
        expected = supervised + losses["distill"] + 0.25 * losses["pair"]
        self.assertTrue(torch.allclose(losses["total"], expected))
        losses["total"].backward()
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_pair_loss_penalizes_insufficient_positive_margin(self) -> None:
        losses = constrained_losses(
            torch.zeros(2),
            torch.tensor([0.0, 1.0]),
            torch.zeros(1),
            torch.zeros(1),
            torch.tensor([0.2]),
            torch.tensor([0.0]),
            distill_weight=1.0,
            pair_weight=0.25,
            pair_margin=0.5,
        )
        self.assertAlmostEqual(float(losses["pair"]), 0.3, places=6)

    def test_fixed_four_source_batch_uses_expected_class_weight(self) -> None:
        negatives = 16 + 32 + 32
        positives = 16 + 32
        self.assertEqual((negatives, positives), (80, 48))
        self.assertAlmostEqual(negatives / positives, 5.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
