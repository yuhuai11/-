from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from numpy._core.multiarray import _reconstruct
from torch import nn
from torch.nn import functional as F


class CapturingBinaryHead(nn.Linear):
    """Linear head that exposes logits from an upstream probability-only model."""

    last_logits: torch.Tensor | None

    def __init__(self, in_features: int) -> None:
        super().__init__(in_features, 1)
        self.last_logits = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = super().forward(inputs)
        self.last_logits = output
        return output


class RecordingMeanFrequencyMasking(nn.Module):
    """Frequency-only masking for log-Mel tensors shaped B x 1 x T x F."""

    def __init__(
        self,
        *,
        probability: float,
        maximum_masks: int,
        maximum_width_fraction: float,
    ) -> None:
        super().__init__()
        if not 0.0 <= probability <= 1.0:
            raise ValueError("frequency masking probability must be in [0, 1]")
        if maximum_masks < 1:
            raise ValueError("frequency masking maximum_masks must be positive")
        if not 0.0 < maximum_width_fraction <= 1.0:
            raise ValueError(
                "frequency masking maximum_width_fraction must be in (0, 1]"
            )
        self.probability = float(probability)
        self.maximum_masks = int(maximum_masks)
        self.maximum_width_fraction = float(maximum_width_fraction)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return features
        if features.ndim != 4 or features.size(1) != 1:
            raise ValueError("frequency masking expects B x 1 x T x F features")
        frequency_bins = int(features.size(-1))
        maximum_width = max(
            1,
            min(
                frequency_bins,
                int(np.floor(frequency_bins * self.maximum_width_fraction)),
            ),
        )
        output = features.clone()
        apply_mask = torch.rand(features.size(0), device=features.device) < self.probability
        for batch_index in torch.nonzero(apply_mask, as_tuple=False).flatten().tolist():
            fill = features[batch_index].mean()
            mask_count = int(
                torch.randint(
                    1,
                    self.maximum_masks + 1,
                    (),
                    device=features.device,
                ).item()
            )
            for _ in range(mask_count):
                width = int(
                    torch.randint(
                        1,
                        maximum_width + 1,
                        (),
                        device=features.device,
                    ).item()
                )
                start = int(
                    torch.randint(
                        0,
                        frequency_bins - width + 1,
                        (),
                        device=features.device,
                    ).item()
                )
                output[batch_index, :, :, start : start + width] = fill
        return output


class FrequencyMixStyle(nn.Module):
    """Mix per-instance frequency statistics for B x C x T x F features."""

    def __init__(
        self,
        *,
        probability: float,
        beta_alpha: float,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 0.0 <= probability <= 1.0:
            raise ValueError("frequency MixStyle probability must be in [0, 1]")
        if not np.isfinite(beta_alpha) or beta_alpha <= 0.0:
            raise ValueError("frequency MixStyle beta_alpha must be positive")
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("frequency MixStyle epsilon must be positive")
        self.probability = float(probability)
        self.beta_alpha = float(beta_alpha)
        self.epsilon = float(epsilon)
        self.last_applied = False

    def forward(self, features: torch.Tensor, *, force: bool = False) -> torch.Tensor:
        self.last_applied = False
        if features.ndim != 4:
            raise ValueError("frequency MixStyle expects B x C x T x F features")
        if not self.training or features.size(0) < 2:
            return features
        if not force and (
            self.probability == 0.0
            or bool(torch.rand((), device=features.device) >= self.probability)
        ):
            return features

        # PANNs features are B x C x T x F. Following Freq-MixStyle, compute
        # instance statistics over the frequency axis while retaining channel
        # and temporal structure. Stop gradients through statistics as in the
        # original MixStyle formulation.
        mean = features.mean(dim=-1, keepdim=True).detach()
        variance = features.var(dim=-1, keepdim=True, unbiased=False)
        standard_deviation = torch.sqrt(variance + self.epsilon).detach()
        permutation = torch.randperm(features.size(0), device=features.device)
        concentration = torch.tensor(
            self.beta_alpha, dtype=features.dtype, device=features.device
        )
        mixing = torch.distributions.Beta(concentration, concentration).sample(
            (features.size(0), 1, 1, 1)
        )
        mixed_mean = mixing * mean + (1.0 - mixing) * mean[permutation]
        mixed_standard_deviation = (
            mixing * standard_deviation
            + (1.0 - mixing) * standard_deviation[permutation]
        )
        normalized = (features - mean) / standard_deviation
        self.last_applied = True
        return normalized * mixed_standard_deviation + mixed_mean


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_official_checkpoint(path: Path, expected_sha256: str) -> dict:
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            f"PANNs checkpoint SHA256 mismatch: expected {expected_sha256}, observed {observed}"
        )
    original_module = _reconstruct.__module__
    try:
        _reconstruct.__module__ = "numpy.core.multiarray"
        safe_types = [_reconstruct, np.ndarray, np.dtype, np.dtypes.Int64DType]
        with torch.serialization.safe_globals(safe_types):
            result = torch.load(path, map_location="cpu", weights_only=True)
    finally:
        _reconstruct.__module__ = original_module
    if not isinstance(result, dict) or not isinstance(result.get("model"), dict):
        raise ValueError("Unexpected official PANNs checkpoint structure")
    return result


class PannsCnn14Binary(nn.Module):
    """Official PANNs Cnn14_16k adapted to emit one binary logit."""

    def __init__(
        self,
        *,
        initialization: str,
        vendor_dir: str,
        checkpoint_path: str,
        checkpoint_sha256: str,
        spec_augment: bool,
        frontend_precision: str,
        binary_checkpoint_path: str | None = None,
        binary_checkpoint_sha256: str | None = None,
        trainable_scope: str = "full",
        frequency_masking: dict | None = None,
        frequency_mixstyle: dict | None = None,
    ) -> None:
        super().__init__()
        if initialization not in {"scratch", "audioset"}:
            raise ValueError("initialization must be 'scratch' or 'audioset'")
        if frontend_precision != "float32":
            raise ValueError("PANNs spectral frontend must use float32 to prevent AMP overflow")
        vendor = Path(vendor_dir).resolve()
        sys.path.insert(0, str(vendor))
        try:
            from models import Cnn14_16k
        finally:
            sys.path.pop(0)
        backbone = Cnn14_16k(
            sample_rate=16000,
            window_size=512,
            hop_size=160,
            mel_bins=64,
            fmin=50,
            fmax=8000,
            classes_num=527,
        )
        if initialization == "audioset":
            official = safe_official_checkpoint(Path(checkpoint_path), checkpoint_sha256)
            backbone.load_state_dict(official["model"], strict=True)
        # Both arms construct the original 527-class model first and replace
        # its head at the same RNG position. This keeps the new binary head
        # initialization identical for equal seeds.
        backbone.fc_audioset = CapturingBinaryHead(2048)
        if frequency_masking and bool(frequency_masking.get("enabled", False)):
            if spec_augment:
                raise ValueError(
                    "PANNs built-in spec_augment and frequency-only masking "
                    "cannot be enabled together"
                )
            backbone.spec_augmenter = RecordingMeanFrequencyMasking(
                probability=float(frequency_masking.get("probability", 0.5)),
                maximum_masks=int(frequency_masking.get("maximum_masks", 3)),
                maximum_width_fraction=float(
                    frequency_masking.get("maximum_width_fraction", 0.15)
                ),
            )
        elif not spec_augment:
            backbone.spec_augmenter = nn.Identity()
        if frequency_mixstyle and bool(frequency_mixstyle.get("enabled", False)):
            self.frequency_mixstyle = FrequencyMixStyle(
                probability=float(frequency_mixstyle.get("probability", 0.5)),
                beta_alpha=float(frequency_mixstyle.get("beta_alpha", 0.6)),
                epsilon=float(frequency_mixstyle.get("epsilon", 1e-6)),
            )
        else:
            self.frequency_mixstyle = nn.Identity()
        self.backbone = backbone
        self.initialization = initialization
        self.frontend_precision = frontend_precision
        if binary_checkpoint_path:
            if not binary_checkpoint_sha256:
                raise ValueError("binary_checkpoint_sha256 is required")
            checkpoint_path = Path(binary_checkpoint_path)
            observed = file_sha256(checkpoint_path)
            if observed != binary_checkpoint_sha256:
                raise ValueError(
                    "Binary PANNs checkpoint SHA256 mismatch: "
                    f"expected {binary_checkpoint_sha256}, observed {observed}"
                )
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(saved, dict) or not isinstance(saved.get("model"), dict):
                raise ValueError("Unexpected binary PANNs checkpoint structure")
            self.load_state_dict(saved["model"], strict=True)
        if trainable_scope not in {
            "full",
            "binary_head_only",
            "fc1_and_binary_head",
        }:
            raise ValueError(
                "trainable_scope must be 'full', 'binary_head_only' or "
                "'fc1_and_binary_head'"
            )
        self.trainable_scope = trainable_scope
        if trainable_scope != "full":
            for parameter in self.parameters():
                parameter.requires_grad = False
            for parameter in self.backbone.fc_audioset.parameters():
                parameter.requires_grad = True
            if trainable_scope == "fc1_and_binary_head":
                for parameter in self.backbone.fc1.parameters():
                    parameter.requires_grad = True

    def train(self, mode: bool = True) -> PannsCnn14Binary:
        super().train(mode)
        if mode and self.trainable_scope != "full":
            self.backbone.eval()
            if self.trainable_scope == "fc1_and_binary_head":
                self.backbone.fc1.train(True)
            self.backbone.fc_audioset.train(True)
        return self

    def extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return the frozen 2048-D G7 representation before its binary head."""
        # torchlibrosa computes real**2 + imag**2. In float16, tonal real-world
        # clips can overflow that square even when waveform samples are in
        # [-1, 1]. Keep only the spectral frontend and bn0 computation in float32;
        # the surrounding autocast context resumes for the trainable CNN.
        device_type = waveform.device.type
        with torch.amp.autocast(device_type, enabled=False):
            x = self.backbone.spectrogram_extractor(waveform.float())
            x = self.backbone.logmel_extractor(x)
            x = x.transpose(1, 3)
            x = self.backbone.bn0(x)
            x = x.transpose(1, 3)

        feature_training = (
            self.training and self.trainable_scope == "full"
        )
        if feature_training:
            x = self.backbone.spec_augmenter(x)
            x = self.frequency_mixstyle(x)
        x = self.backbone.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=feature_training)
        x = self.backbone.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=feature_training)
        x = self.backbone.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=feature_training)
        x = self.backbone.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=feature_training)
        x = self.backbone.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=feature_training)
        x = self.backbone.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=feature_training)
        x = torch.mean(x, dim=3)
        x = torch.max(x, dim=2).values + torch.mean(x, dim=2)
        x = F.dropout(x, p=0.5, training=feature_training)
        x = F.relu_(self.backbone.fc1(x))
        return x

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.extract_embedding(waveform)
        logits = self.backbone.fc_audioset(x)
        return logits.squeeze(1)
