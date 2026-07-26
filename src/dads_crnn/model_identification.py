from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HierarchicalDecision:
    state: str
    model_index: int | None
    drone_probability: float
    model_confidence: float | None


class G7ModelIdentifier(nn.Module):
    """Frozen G7 detector plus a trainable known-model classification head."""

    def __init__(self, detector: nn.Module, embedding_dim: int, classes: int) -> None:
        super().__init__()
        if classes < 2 or embedding_dim <= 0:
            raise ValueError("Invalid model-identifier dimensions")
        if not hasattr(detector, "extract_embedding"):
            raise TypeError("Detector must expose extract_embedding")
        self.detector = detector
        for parameter in self.detector.parameters():
            parameter.requires_grad = False
        self.detector.eval()
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, classes)

    def train(self, mode: bool = True) -> G7ModelIdentifier:
        super().train(mode)
        self.detector.eval()
        return self

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            embedding = self.detector.extract_embedding(waveform)
            drone_logit = self.detector.backbone.fc_audioset(embedding).squeeze(1)
        model_logits = self.classifier(self.embedding_norm(embedding.detach()))
        return drone_logit, model_logits

    @property
    def trainable_parameter_names(self) -> list[str]:
        return [
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        ]


def hierarchical_decisions(
    drone_logits: torch.Tensor,
    model_logits: torch.Tensor,
    *,
    drone_threshold: float,
    known_confidence_threshold: float,
) -> list[HierarchicalDecision]:
    if drone_logits.ndim != 1 or model_logits.ndim != 2:
        raise ValueError("Unexpected hierarchical-logit shapes")
    if len(drone_logits) != len(model_logits):
        raise ValueError("Detector and model-ID batch sizes differ")
    if not (0.0 < drone_threshold < 1.0):
        raise ValueError("drone_threshold must be inside (0, 1)")
    if not (0.0 < known_confidence_threshold < 1.0):
        raise ValueError("known_confidence_threshold must be inside (0, 1)")
    drone_probability = torch.sigmoid(drone_logits)
    model_probability = F.softmax(model_logits, dim=1)
    confidence, indices = model_probability.max(dim=1)
    result = []
    for drone, model_confidence, model_index in zip(
        drone_probability, confidence, indices, strict=True
    ):
        drone_value = float(drone)
        confidence_value = float(model_confidence)
        if drone_value < drone_threshold:
            result.append(
                HierarchicalDecision("background", None, drone_value, None)
            )
        elif confidence_value < known_confidence_threshold:
            result.append(
                HierarchicalDecision("unknown_uav", None, drone_value, confidence_value)
            )
        else:
            result.append(
                HierarchicalDecision(
                    "known_uav", int(model_index), drone_value, confidence_value
                )
            )
    return result
