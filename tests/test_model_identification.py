from __future__ import annotations

import unittest

import torch
from torch import nn

from dads_crnn.model_identification import (
    G7ModelIdentifier,
    hierarchical_decisions,
)
from dads_crnn.preflight_g18_model_id import select_balanced_models

import pandas as pd


class DummyHead(nn.Linear):
    pass


class DummyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 8)
        self.backbone = nn.Module()
        self.backbone.fc_audioset = DummyHead(8, 1)

    def extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.encoder(waveform)


class ModelIdentificationTests(unittest.TestCase):
    def test_only_model_head_is_trainable(self) -> None:
        model = G7ModelIdentifier(DummyDetector(), embedding_dim=8, classes=3)
        self.assertEqual(
            model.trainable_parameter_names,
            [
                "embedding_norm.weight",
                "embedding_norm.bias",
                "classifier.weight",
                "classifier.bias",
            ],
        )
        self.assertFalse(any(parameter.requires_grad for parameter in model.detector.parameters()))

    def test_hierarchy_emits_background_unknown_and_known(self) -> None:
        drone_logits = torch.tensor([-10.0, 10.0, 10.0])
        model_logits = torch.tensor(
            [[10.0, 0.0], [0.0, 0.0], [10.0, 0.0]], dtype=torch.float32
        )
        decisions = hierarchical_decisions(
            drone_logits,
            model_logits,
            drone_threshold=0.5,
            known_confidence_threshold=0.8,
        )
        self.assertEqual(
            [decision.state for decision in decisions],
            ["background", "unknown_uav", "known_uav"],
        )
        self.assertEqual(decisions[2].model_index, 0)

    def test_frozen_detector_gets_no_gradients(self) -> None:
        model = G7ModelIdentifier(DummyDetector(), embedding_dim=8, classes=3)
        _, logits = model(torch.randn(5, 4))
        torch.nn.functional.cross_entropy(logits, torch.arange(5) % 3).backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.detector.parameters()))
        self.assertTrue(all(parameter.grad is not None for parameter in model.classifier.parameters()))

    def test_preflight_selection_balances_models(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "model_id": model,
                    "target_index": target,
                    "audio_sha256": f"{target}{index}",
                }
                for target, model in enumerate(("A", "B", "C"))
                for index in range(5)
            ]
        )
        selected = select_balanced_models(frame, samples_per_model=2, seed=42)
        self.assertEqual(selected["model_id"].value_counts().to_dict(), {"A": 2, "B": 2, "C": 2})


if __name__ == "__main__":
    unittest.main()
