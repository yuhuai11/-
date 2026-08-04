from __future__ import annotations

import hashlib
import io
from typing import Final

import numpy as np
from scipy.io import wavfile

from .audio import ensure_sample_rate


CANONICAL_SAMPLE_RATE: Final = 16_000
IDENTITY_PROTOCOL: Final = "g7_audio_identity_v1"


def decode_wav_channel_variants(
    wav_bytes: bytes,
) -> tuple[dict[str, np.ndarray], int]:
    """Decode a WAV while preserving channel variants needed for identity checks."""
    sample_rate, values = wavfile.read(io.BytesIO(wav_bytes))
    samples = np.asarray(values)
    if samples.dtype == np.uint8:
        audio = (samples.astype(np.float32) - 128.0) / 128.0
    elif samples.dtype.kind == "i":
        scale = float(2 ** (samples.dtype.itemsize * 8 - 1))
        audio = samples.astype(np.float32) / scale
    elif samples.dtype.kind == "f":
        audio = samples.astype(np.float32)
    else:
        raise ValueError(f"Unsupported WAV dtype for identity audit: {samples.dtype}")
    if not np.isfinite(audio).all():
        raise ValueError("WAV identity input contains NaN or infinite samples")
    audio = np.clip(audio, -1.0, 1.0)

    if audio.ndim == 1:
        variants = {"mono": audio}
    elif audio.ndim == 2 and audio.shape[1] >= 1:
        variants = {
            "mono_mean": audio.mean(axis=1, dtype=np.float32),
            **{
                f"channel_{channel}": audio[:, channel].astype(np.float32, copy=False)
                for channel in range(audio.shape[1])
            },
        }
    else:
        raise ValueError(f"Unsupported WAV shape for identity audit: {audio.shape}")
    return variants, int(sample_rate)


def quantized_pcm_sha256(
    audio: np.ndarray,
    sample_rate: int,
    *,
    normalize_gain: bool,
    target_rate: int = CANONICAL_SAMPLE_RATE,
) -> str:
    """Hash deterministic mono, resampled, little-endian int16 PCM."""
    values = ensure_sample_rate(
        np.asarray(audio, dtype=np.float32), int(sample_rate), int(target_rate)
    )
    if not np.isfinite(values).all():
        raise ValueError("Non-finite samples cannot be hashed")
    values = np.clip(values, -1.0, 1.0 - 1.0 / 32768.0)
    if normalize_gain:
        peak = float(np.max(np.abs(values))) if values.size else 0.0
        if peak > 1e-8:
            values = values / peak
            values = np.clip(values, -1.0, 1.0 - 1.0 / 32768.0)
    quantized = np.rint(values * 32768.0).astype("<i2")
    header = (
        f"{IDENTITY_PROTOCOL}|rate={target_rate}|samples={quantized.size}|"
        f"gain_normalized={int(normalize_gain)}|"
    ).encode("ascii")
    return hashlib.sha256(header + quantized.tobytes()).hexdigest()


def spectral_fingerprint(
    audio: np.ndarray,
    sample_rate: int,
    *,
    bands: int = 48,
    target_rate: int = CANONICAL_SAMPLE_RATE,
) -> np.ndarray:
    """Return an amplitude-invariant spectral profile for candidate retrieval."""
    values = ensure_sample_rate(
        np.asarray(audio, dtype=np.float32), int(sample_rate), int(target_rate)
    )
    if values.size == 0:
        return np.zeros(bands, dtype=np.float32)
    values = values - float(values.mean())
    peak = float(np.max(np.abs(values)))
    if peak > 1e-8:
        values = values / peak
    frame_length = 512
    hop = 256
    if values.size < frame_length:
        values = np.pad(values, (0, frame_length - values.size))
    frame_count = 1 + (values.size - frame_length) // hop
    frames = np.lib.stride_tricks.sliding_window_view(values, frame_length)[
        : frame_count * hop : hop
    ]
    window = np.hanning(frame_length).astype(np.float32)
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    mean_power = power.mean(axis=0)
    edges = np.geomspace(50.0, target_rate / 2.0, bands + 1)
    frequencies = np.fft.rfftfreq(frame_length, 1.0 / target_rate)
    profile = np.empty(bands, dtype=np.float32)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        selected = (frequencies >= left) & (
            frequencies < right if index + 1 < bands else frequencies <= right
        )
        profile[index] = float(np.log1p(mean_power[selected].mean())) if selected.any() else 0.0
    profile -= float(profile.mean())
    norm = float(np.linalg.norm(profile))
    if norm > 1e-8:
        profile /= norm
    return profile
