from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from dads_crnn.g19_model import (
    G19RecordingHead,
    GatedAttentionPool,
    masked_softmax,
    supervised_contrastive_loss,
)
from dads_crnn.g19_recording_data import (
    RecordingFeatureDataset,
    balanced_recording_batches,
    collate_recordings,
)


class G19ModelTests(unittest.TestCase):
    def test_supervised_contrastive_loss_matches_small_analytic_case(
        self,
    ) -> None:
        embeddings = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        )
        targets = torch.tensor([0, 0, 1, 1])

        observed = supervised_contrastive_loss(
            embeddings, targets, temperature=1.0
        )

        self.assertAlmostEqual(
            float(observed),
            math.log(math.e + 2.0) - 1.0,
            places=6,
        )

    def test_supervised_contrastive_loss_rejects_singleton_class(self) -> None:
        with self.assertRaisesRegex(ValueError, "same-class positive"):
            supervised_contrastive_loss(
                torch.eye(3),
                torch.tensor([0, 0, 1]),
            )

    def test_masked_softmax_zeros_padding_and_normalizes_valid_positions(
        self,
    ) -> None:
        logits = torch.tensor([[0.0, 0.0, 99.0], [1.0, 2.0, 3.0]])
        mask = torch.tensor([[True, True, False], [True, False, False]])

        probabilities = masked_softmax(logits, mask)

        self.assertEqual(
            probabilities.tolist(),
            [[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]],
        )

    def test_zero_score_attention_is_uniform_over_only_valid_segments(
        self,
    ) -> None:
        pool = GatedAttentionPool(input_dim=3, hidden_dim=2)
        with torch.no_grad():
            for parameter in pool.parameters():
                parameter.zero_()
        features = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [100.0, 100.0, 100.0],
                ],
                [
                    [0.0, 0.0, 3.0],
                    [50.0, 50.0, 50.0],
                    [60.0, 60.0, 60.0],
                ],
            ]
        )
        mask = torch.tensor([[True, True, False], [True, False, False]])

        pooled, weights = pool(features, mask)

        torch.testing.assert_close(
            weights,
            torch.tensor([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]]),
        )
        torch.testing.assert_close(
            pooled,
            torch.tensor([[0.5, 1.0, 0.0], [0.0, 0.0, 3.0]]),
        )

    def test_recording_head_returns_one_normalized_embedding_per_recording(
        self,
    ) -> None:
        head = G19RecordingHead(
            g7_embedding_dim=8,
            projection_hidden_dim=6,
            embedding_dim=4,
            attention_hidden_dim=3,
            classes=3,
            dropout=0.0,
        )
        features = torch.randn(2, 5, 8)
        mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, True],
            ]
        )

        output = head(features, mask)

        self.assertEqual(tuple(output.logits.shape), (2, 3))
        self.assertEqual(tuple(output.embedding.shape), (2, 4))
        self.assertEqual(tuple(output.attention.shape), (2, 5))
        torch.testing.assert_close(
            torch.linalg.vector_norm(output.embedding, dim=1),
            torch.ones(2),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        self.assertEqual(output.attention[0, 3:].tolist(), [0.0, 0.0])

    def test_balanced_recording_batch_supports_one_complete_train_step(
        self,
    ) -> None:
        rows = []
        for target in range(2):
            for recording in range(2):
                for segment in range(2):
                    rows.append(
                        {
                            "audio_sha256": f"{target}-{recording}",
                            "segment_index": segment,
                            "target_index": target,
                            "model_id": f"MODEL_{target}",
                            "is_known": True,
                        }
                    )
        frame = pd.DataFrame(rows)
        features = np.random.default_rng(19).normal(
            size=(len(frame), 8)
        ).astype(np.float32)
        dataset = RecordingFeatureDataset(frame, features)
        batch = balanced_recording_batches(
            dataset.examples,
            recordings_per_class=2,
            seed=42,
            epoch=1,
        )[0]
        padded, mask, targets, _ = collate_recordings(
            [dataset[int(index)] for index in batch]
        )
        head = G19RecordingHead(
            g7_embedding_dim=8,
            projection_hidden_dim=6,
            embedding_dim=4,
            attention_hidden_dim=3,
            classes=2,
            dropout=0.0,
        )

        output = head(padded, mask)
        loss = F.cross_entropy(output.logits, targets)
        loss = loss + 0.2 * supervised_contrastive_loss(
            output.embedding,
            targets,
            temperature=0.07,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                for parameter in head.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
