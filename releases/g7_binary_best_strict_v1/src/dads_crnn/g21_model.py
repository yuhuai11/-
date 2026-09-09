from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


class G21PartialFineTuneIdentifier(nn.Module):
    """A private model-ID copy of G7 with narrowly scoped transfer learning."""

    def __init__(
        self,
        detector: nn.Module,
        *,
        embedding_dim: int,
        classes: int,
        trainable_detector_prefixes: Iterable[str] = ("backbone.fc1.",),
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or classes < 2:
            raise ValueError("Invalid G21 model dimensions")
        if not hasattr(detector, "extract_embedding"):
            raise TypeError("G21 detector must expose extract_embedding")
        prefixes = tuple(str(value) for value in trainable_detector_prefixes)
        if not prefixes or any(not value for value in prefixes):
            raise ValueError("G21 requires explicit detector parameter prefixes")

        self.detector = detector
        self.trainable_detector_prefixes = prefixes
        for name, parameter in self.detector.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))
        selected = [
            name
            for name, parameter in self.detector.named_parameters()
            if parameter.requires_grad
        ]
        if not selected:
            raise ValueError("G21 detector prefixes selected no parameters")

        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, classes)
        self._reference_buffers: dict[str, str] = {}
        for index, (name, parameter) in enumerate(self.detector.named_parameters()):
            if parameter.requires_grad:
                buffer_name = f"_l2_sp_reference_{index}"
                self.register_buffer(
                    buffer_name, parameter.detach().clone(), persistent=False
                )
                self._reference_buffers[name] = buffer_name

    def train(self, mode: bool = True) -> G21PartialFineTuneIdentifier:
        super().train(mode)
        # Batch-normalization statistics and all PANNs dropout/spec-augmentation
        # remain frozen. Only the explicitly selected affine layer is adapted.
        self.detector.eval()
        return self

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.detector.extract_embedding(waveform)
        model_logits = self.classifier(self.embedding_norm(embedding))
        return embedding, model_logits

    def l2_sp_penalty(self) -> torch.Tensor:
        """Squared distance from the trusted G7 initialization (L2-SP)."""
        terms = []
        for name, parameter in self.detector.named_parameters():
            if parameter.requires_grad:
                reference = getattr(self, self._reference_buffers[name])
                terms.append((parameter - reference).square().sum())
        if not terms:
            raise RuntimeError("G21 has no adapted detector parameters")
        return torch.stack(terms).sum()

    @property
    def adapted_detector_parameter_names(self) -> list[str]:
        return [
            name
            for name, parameter in self.detector.named_parameters()
            if parameter.requires_grad
        ]
