from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .g19_model import masked_softmax


@dataclass(frozen=True)
class G20RecordingOutput:
    logits: torch.Tensor
    embedding: torch.Tensor
    attention: torch.Tensor
    segment_logits: torch.Tensor


class CosineClassifier(nn.Module):
    """A normalized classifier whose scale is learned from data."""

    def __init__(self, input_dim: int, classes: int, initial_scale: float = 16.0) -> None:
        super().__init__()
        if input_dim <= 0 or classes < 2 or initial_scale <= 0.0:
            raise ValueError("Invalid cosine-classifier configuration")
        self.weight = nn.Parameter(torch.empty(classes, input_dim))
        self.log_scale = nn.Parameter(torch.tensor(float(initial_scale)).log())
        nn.init.xavier_uniform_(self.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        scale = self.log_scale.exp().clamp(max=100.0)
        return scale * F.linear(
            F.normalize(values, dim=-1),
            F.normalize(self.weight, dim=-1),
        )


class MultiHeadAttentiveStatisticsPool(nn.Module):
    """Pool complementary segment evidence with weighted mean and deviation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, heads) <= 0:
            raise ValueError("Pooling dimensions and head count must be positive")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("Pooling dropout must be inside [0, 1)")
        self.heads = int(heads)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(hidden_dim, heads, bias=False)

    def forward(
        self,
        segment_embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if segment_embeddings.ndim != 3 or mask.ndim != 2:
            raise ValueError("G20 pooling expects [B, S, D] and [B, S]")
        if segment_embeddings.shape[:2] != mask.shape:
            raise ValueError("G20 pooling feature and mask shapes differ")
        if mask.dtype != torch.bool:
            raise TypeError("G20 pooling mask must be boolean")
        hidden = torch.tanh(self.value(segment_embeddings))
        hidden = hidden * torch.sigmoid(self.gate(segment_embeddings))
        scores = self.score(self.dropout(hidden)).transpose(1, 2)
        expanded_mask = mask[:, None, :].expand_as(scores)
        weights = masked_softmax(scores, expanded_mask, dim=2)
        means = torch.einsum("bhs,bsd->bhd", weights, segment_embeddings)
        second_moments = torch.einsum(
            "bhs,bsd->bhd", weights, segment_embeddings.square()
        )
        deviations = (
            second_moments - means.square()
        ).clamp_min(1.0e-5).sqrt()
        statistics = torch.cat((means, deviations), dim=-1).flatten(1)
        return statistics, weights


def normalized_attention_entropy(
    attention: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return one normalized entropy value per recording and attention head."""
    if attention.ndim != 3 or mask.ndim != 2:
        raise ValueError("Attention entropy expects [B, H, S] and [B, S]")
    if attention.shape[0] != mask.shape[0] or attention.shape[2] != mask.shape[1]:
        raise ValueError("Attention entropy shapes differ")
    expanded_mask = mask[:, None, :].expand_as(attention)
    entropy = -torch.sum(
        torch.where(
            expanded_mask,
            attention.clamp_min(torch.finfo(attention.dtype).tiny).log()
            * attention,
            torch.zeros_like(attention),
        ),
        dim=2,
    )
    counts = mask.sum(dim=1, keepdim=True)
    denominator = counts.float().log()
    return torch.where(
        counts > 1,
        entropy / denominator.clamp_min(torch.finfo(entropy.dtype).tiny),
        torch.zeros_like(entropy),
    )


def attention_regularization(
    attention: torch.Tensor,
    mask: torch.Tensor,
    *,
    maximum_normalized_entropy: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize collapsed heads and attention that remains nearly uniform."""
    if not (0.0 <= maximum_normalized_entropy <= 1.0):
        raise ValueError("Maximum normalized attention entropy must be in [0, 1]")
    entropy = normalized_attention_entropy(attention, mask)
    focus_loss = F.relu(entropy - maximum_normalized_entropy).square().mean()
    normalized_heads = F.normalize(attention, dim=2)
    similarities = normalized_heads @ normalized_heads.transpose(1, 2)
    heads = attention.shape[1]
    if heads == 1:
        diversity_loss = similarities.new_zeros(())
    else:
        identity = torch.eye(
            heads, device=attention.device, dtype=torch.bool
        )[None, :, :]
        diversity_loss = similarities.masked_select(~identity).mean()
    return diversity_loss, focus_loss


class G20ClosedSetHead(nn.Module):
    """Known-model identifier with multi-head attentive statistics pooling."""

    def __init__(
        self,
        *,
        g7_embedding_dim: int,
        projection_hidden_dim: int,
        embedding_dim: int,
        attention_hidden_dim: int,
        attention_heads: int,
        classes: int,
        dropout: float = 0.10,
        cosine_scale: float = 16.0,
    ) -> None:
        super().__init__()
        values = (
            g7_embedding_dim,
            projection_hidden_dim,
            embedding_dim,
            attention_hidden_dim,
            attention_heads,
            classes,
        )
        if any(value <= 0 for value in values) or classes < 2:
            raise ValueError("Invalid G20 head dimensions")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("G20 dropout must be inside [0, 1)")
        self.g7_embedding_dim = int(g7_embedding_dim)
        self.embedding_dim = int(embedding_dim)
        self.attention_heads = int(attention_heads)
        self.projector = nn.Sequential(
            nn.LayerNorm(g7_embedding_dim),
            nn.Linear(g7_embedding_dim, projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden_dim, embedding_dim),
        )
        self.segment_classifier = nn.Linear(embedding_dim, classes)
        self.aggregator = MultiHeadAttentiveStatisticsPool(
            embedding_dim,
            attention_hidden_dim,
            attention_heads,
            dropout=dropout,
        )
        statistics_dim = attention_heads * embedding_dim * 2
        self.recording_projector = nn.Sequential(
            nn.LayerNorm(statistics_dim),
            nn.Linear(statistics_dim, projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden_dim, embedding_dim),
        )
        self.classifier = CosineClassifier(
            embedding_dim, classes, initial_scale=cosine_scale
        )

    def forward(
        self, g7_embeddings: torch.Tensor, mask: torch.Tensor
    ) -> G20RecordingOutput:
        if g7_embeddings.ndim != 3 or g7_embeddings.shape[:2] != mask.shape:
            raise ValueError("G20 expects [recordings, segments, embedding]")
        if g7_embeddings.shape[2] != self.g7_embedding_dim:
            raise ValueError("G20 received an unexpected G7 embedding dimension")
        if mask.dtype != torch.bool:
            raise TypeError("G20 recording mask must be boolean")
        if not torch.all(mask.any(dim=1)):
            raise ValueError("Every G20 recording must contain a valid segment")
        if not torch.isfinite(g7_embeddings).all():
            raise ValueError("G20 G7 embeddings must be finite")

        segments = F.normalize(self.projector(g7_embeddings.float()), dim=-1)
        segment_logits = self.segment_classifier(segments)
        statistics, attention = self.aggregator(segments, mask)
        recording_embedding = F.normalize(
            self.recording_projector(statistics), dim=-1
        )
        logits = self.classifier(recording_embedding)
        return G20RecordingOutput(
            logits=logits,
            embedding=recording_embedding,
            attention=attention,
            segment_logits=segment_logits,
        )
