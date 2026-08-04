from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from dads_crnn.g20_model import (
    G20ClosedSetHead,
    MultiHeadAttentiveStatisticsPool,
    attention_regularization,
    normalized_attention_entropy,
)


class G20ModelTests(unittest.TestCase):
    def test_pool_returns_weighted_mean_and_deviation_per_head(self) -> None:
        pool = MultiHeadAttentiveStatisticsPool(2, 3, 2, dropout=0.0)
        with torch.no_grad():
            for parameter in pool.parameters():
                parameter.zero_()
        values = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]]
        )
        mask = torch.tensor([[True, True, False]])

        statistics, attention = pool(values, mask)

        self.assertEqual(tuple(statistics.shape), (1, 8))
        self.assertEqual(tuple(attention.shape), (1, 2, 3))
        torch.testing.assert_close(
            attention,
            torch.tensor([[[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]]),
        )
        expected_head = torch.tensor([2.0, 3.0, 1.0, 1.0])
        torch.testing.assert_close(statistics[0, :4], expected_head)
        torch.testing.assert_close(statistics[0, 4:], expected_head)

    def test_attention_regularization_detects_uniform_collapsed_heads(self) -> None:
        attention = torch.tensor(
            [[[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]]
        )
        mask = torch.tensor([[True, True, False]])

        diversity, focus = attention_regularization(
            attention, mask, maximum_normalized_entropy=0.8
        )

        self.assertAlmostEqual(float(diversity), 1.0, places=6)
        self.assertAlmostEqual(float(focus), 0.04, places=6)
        torch.testing.assert_close(
            normalized_attention_entropy(attention, mask),
            torch.ones(1, 2),
        )

    def test_closed_set_head_has_valid_outputs_and_complete_gradients(self) -> None:
        head = G20ClosedSetHead(
            g7_embedding_dim=8,
            projection_hidden_dim=12,
            embedding_dim=4,
            attention_hidden_dim=5,
            attention_heads=3,
            classes=3,
            dropout=0.0,
        )
        features = torch.randn(4, 5, 8)
        mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, False],
                [True, True, True, True, True],
                [True, True, False, False, False],
            ]
        )
        targets = torch.tensor([0, 0, 1, 1])

        output = head(features, mask)
        diversity, focus = attention_regularization(
            output.attention, mask, maximum_normalized_entropy=0.85
        )
        expanded_targets = targets[:, None].expand_as(mask)
        loss = F.cross_entropy(output.logits, targets)
        loss = loss + 0.2 * F.cross_entropy(
            output.segment_logits[mask], expanded_targets[mask]
        )
        loss = loss + 0.05 * diversity + 0.02 * focus
        loss.backward()

        self.assertEqual(tuple(output.logits.shape), (4, 3))
        self.assertEqual(tuple(output.embedding.shape), (4, 4))
        self.assertEqual(tuple(output.attention.shape), (4, 3, 5))
        self.assertEqual(tuple(output.segment_logits.shape), (4, 5, 3))
        self.assertEqual(output.attention[0, :, 3:].tolist(), [[0.0, 0.0]] * 3)
        torch.testing.assert_close(
            torch.linalg.vector_norm(output.embedding, dim=1),
            torch.ones(4),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        self.assertTrue(
            all(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                for parameter in head.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
