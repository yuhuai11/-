from __future__ import annotations

import unittest

import torch
from torch import nn

from dads_crnn.g21_model import G21PartialFineTuneIdentifier


class DummyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.frozen = nn.Linear(4, 4)
        self.backbone.fc1 = nn.Linear(4, 4)

    def extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        values = self.backbone.frozen(waveform)
        return self.backbone.fc1(values)


class G21ModelTests(unittest.TestCase):
    def test_only_registered_detector_layer_is_adapted(self) -> None:
        model = G21PartialFineTuneIdentifier(
            DummyDetector(), embedding_dim=4, classes=3
        )

        self.assertEqual(
            set(model.adapted_detector_parameter_names),
            {"backbone.fc1.weight", "backbone.fc1.bias"},
        )
        self.assertFalse(model.detector.backbone.frozen.weight.requires_grad)
        self.assertEqual(float(model.l2_sp_penalty()), 0.0)

    def test_penalty_tracks_departure_from_g7_initialization(self) -> None:
        model = G21PartialFineTuneIdentifier(
            DummyDetector(), embedding_dim=4, classes=3
        )
        with torch.no_grad():
            model.detector.backbone.fc1.weight.add_(1.0)

        self.assertGreater(float(model.l2_sp_penalty()), 0.0)

    def test_forward_preserves_batch_and_class_dimensions(self) -> None:
        model = G21PartialFineTuneIdentifier(
            DummyDetector(), embedding_dim=4, classes=3
        )
        embedding, logits = model(torch.randn(5, 4))

        self.assertEqual(embedding.shape, (5, 4))
        self.assertEqual(logits.shape, (5, 3))


if __name__ == "__main__":
    unittest.main()
