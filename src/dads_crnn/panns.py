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
        if not spec_augment:
            backbone.spec_augmenter = nn.Identity()
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
        if trainable_scope not in {"full", "binary_head_only"}:
            raise ValueError("trainable_scope must be 'full' or 'binary_head_only'")
        self.trainable_scope = trainable_scope
        if trainable_scope == "binary_head_only":
            for parameter in self.parameters():
                parameter.requires_grad = False
            for parameter in self.backbone.fc_audioset.parameters():
                parameter.requires_grad = True

    def train(self, mode: bool = True) -> PannsCnn14Binary:
        super().train(mode)
        if mode and self.trainable_scope == "binary_head_only":
            self.backbone.eval()
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
            self.training and self.trainable_scope != "binary_head_only"
        )
        if feature_training:
            x = self.backbone.spec_augmenter(x)
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
