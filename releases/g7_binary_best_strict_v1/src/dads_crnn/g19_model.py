from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class G19RecordingOutput:
    logits: torch.Tensor
    embedding: torch.Tensor
    attention: torch.Tensor


def masked_softmax(
    logits: torch.Tensor, mask: torch.Tensor, *, dim: int = -1
) -> torch.Tensor:
    """Softmax that assigns exactly zero probability to padded positions."""
    if logits.shape != mask.shape:
        raise ValueError("masked_softmax logits and mask must have identical shapes")
    if mask.dtype != torch.bool:
        raise TypeError("masked_softmax mask must be boolean")
    if not torch.all(mask.any(dim=dim)):
        raise ValueError("Every masked_softmax row must contain a valid position")
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(masked, dim=dim)
    probabilities = probabilities.masked_fill(~mask, 0.0)
    return probabilities / probabilities.sum(dim=dim, keepdim=True)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Supervised contrastive loss over one embedding per recording.

    Each anchor must have at least one other recording of the same class in the
    batch. G19's balanced recording sampler enforces this contract.
    """
    if embeddings.ndim != 2 or targets.ndim != 1:
        raise ValueError("SupCon requires [batch, dim] embeddings and [batch] targets")
    if len(embeddings) != len(targets) or len(embeddings) < 2:
        raise ValueError("SupCon inputs are empty or misaligned")
    if targets.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("SupCon targets must contain integer class indices")
    if embeddings.device != targets.device:
        raise ValueError("SupCon embeddings and targets must share a device")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("SupCon temperature must be positive")
    if not torch.isfinite(embeddings).all():
        raise ValueError("SupCon embeddings must be finite")

    normalized = F.normalize(embeddings.float(), dim=1)
    similarities = normalized @ normalized.T / float(temperature)
    identity = torch.eye(len(normalized), device=normalized.device, dtype=torch.bool)
    positives = targets[:, None].eq(targets[None, :]) & ~identity
    if not torch.all(positives.any(dim=1)):
        raise ValueError("Every SupCon anchor needs a same-class positive")

    similarities = similarities - similarities.max(dim=1, keepdim=True).values.detach()
    denominator_mask = ~identity
    exp_logits = torch.exp(similarities) * denominator_mask
    log_probabilities = similarities - torch.log(
        exp_logits.sum(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    )
    mean_positive_log_probability = (
        (log_probabilities * positives).sum(dim=1) / positives.sum(dim=1)
    )
    loss = -mean_positive_log_probability.mean()
    if not torch.isfinite(loss):
        raise RuntimeError("SupCon produced a non-finite loss")
    return loss


class GatedAttentionPool(nn.Module):
    """Learn a quality-aware weight for every segment in a recording."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Attention dimensions must be positive")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("Attention dropout must be inside [0, 1)")
        self.value = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, segment_embeddings: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if segment_embeddings.ndim != 3 or mask.ndim != 2:
            raise ValueError("Attention pooling expects [B, S, D] and [B, S]")
        if segment_embeddings.shape[:2] != mask.shape:
            raise ValueError("Attention pooling feature and mask shapes differ")
        hidden = torch.tanh(self.value(segment_embeddings))
        hidden = hidden * torch.sigmoid(self.gate(segment_embeddings))
        scores = self.score(self.dropout(hidden)).squeeze(-1)
        weights = masked_softmax(scores, mask, dim=1)
        pooled = torch.sum(segment_embeddings * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class G19RecordingHead(nn.Module):
    """Projection, learned recording aggregation, and Known-model classifier."""

    def __init__(
        self,
        *,
        g7_embedding_dim: int,
        projection_hidden_dim: int,
        embedding_dim: int,
        attention_hidden_dim: int,
        classes: int,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        dimensions = (
            g7_embedding_dim,
            projection_hidden_dim,
            embedding_dim,
            attention_hidden_dim,
            classes,
        )
        if any(value <= 0 for value in dimensions) or classes < 2:
            raise ValueError("Invalid G19 recording-head dimensions")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("G19 dropout must be inside [0, 1)")
        self.g7_embedding_dim = int(g7_embedding_dim)
        self.embedding_dim = int(embedding_dim)
        self.classes = int(classes)
        self.projector = nn.Sequential(
            nn.LayerNorm(g7_embedding_dim),
            nn.Linear(g7_embedding_dim, projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden_dim, embedding_dim),
        )
        self.aggregator = GatedAttentionPool(
            embedding_dim,
            attention_hidden_dim,
            dropout=dropout,
        )
        self.classifier = nn.Linear(embedding_dim, classes)

    def forward(
        self, g7_embeddings: torch.Tensor, mask: torch.Tensor
    ) -> G19RecordingOutput:
        if g7_embeddings.ndim != 3:
            raise ValueError("G19 head expects [recordings, segments, embedding]")
        if g7_embeddings.shape[:2] != mask.shape:
            raise ValueError("G19 recording mask is misaligned")
        if g7_embeddings.shape[2] != self.g7_embedding_dim:
            raise ValueError("G19 received an unexpected G7 embedding dimension")
        if mask.dtype != torch.bool:
            raise TypeError("G19 recording mask must be boolean")
        if not torch.isfinite(g7_embeddings).all():
            raise ValueError("G19 G7 embeddings must be finite")

        projected = self.projector(g7_embeddings.float())
        projected = F.normalize(projected, dim=-1)
        pooled, attention = self.aggregator(projected, mask)
        recording_embedding = F.normalize(pooled, dim=-1)
        logits = self.classifier(recording_embedding)
        return G19RecordingOutput(logits, recording_embedding, attention)


class G19RecordingIdentifier(nn.Module):
    """Frozen G7 waveform encoder followed by the trainable G19 head."""

    def __init__(self, detector: nn.Module, head: G19RecordingHead) -> None:
        super().__init__()
        if not hasattr(detector, "extract_embedding"):
            raise TypeError("G19 detector must expose extract_embedding")
        self.detector = detector
        self.head = head
        for parameter in self.detector.parameters():
            parameter.requires_grad = False
        self.detector.eval()

    def train(self, mode: bool = True) -> G19RecordingIdentifier:
        super().train(mode)
        self.detector.eval()
        return self

    def forward(
        self, waveforms: torch.Tensor, mask: torch.Tensor
    ) -> G19RecordingOutput:
        if waveforms.ndim != 3 or waveforms.shape[:2] != mask.shape:
            raise ValueError("G19 waveforms must have shape [B, S, samples]")
        if mask.dtype != torch.bool:
            raise TypeError("G19 waveform mask must be boolean")
        batch, segments, samples = waveforms.shape
        flat_mask = mask.reshape(-1)
        if not flat_mask.any():
            raise ValueError("G19 waveform batch contains no valid segment")
        flat_waveforms = waveforms.reshape(batch * segments, samples)
        with torch.no_grad():
            valid_embeddings = self.detector.extract_embedding(
                flat_waveforms[flat_mask]
            )
        padded = valid_embeddings.new_zeros(
            (batch * segments, valid_embeddings.shape[-1])
        )
        padded[flat_mask] = valid_embeddings
        return self.head(padded.reshape(batch, segments, -1), mask)

    @property
    def trainable_parameter_names(self) -> list[str]:
        return [
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        ]
