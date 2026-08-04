from __future__ import annotations

import io
import wave

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def _decode_ieee_float_wav(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode IEEE-float WAV files that Python's ``wave`` module rejects."""
    sample_rate, values = wavfile.read(io.BytesIO(wav_bytes))
    audio = np.asarray(values)
    if audio.dtype.kind != "f":
        raise ValueError(
            "WAV was rejected by the PCM decoder and is not IEEE floating-point audio"
        )
    if not np.isfinite(audio).all():
        raise ValueError("IEEE-float WAV contains NaN or infinite samples")
    if audio.ndim == 2:
        audio = audio.astype(np.float32).mean(axis=1)
    elif audio.ndim != 1:
        raise ValueError(f"Unsupported IEEE-float WAV shape: {audio.shape}")
    audio = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
    return audio, int(sample_rate)


def decode_wav_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode PCM WAV bytes to mono float32 in [-1, 1]."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())
    except wave.Error:
        return _decode_ieee_float_wav(wav_bytes)

    if sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        packed = np.frombuffer(frames, dtype=np.uint8)
        if packed.size % 3:
            raise ValueError("Malformed 24-bit PCM WAV payload")
        triplets = packed.reshape(-1, 3).astype(np.int32)
        values = triplets[:, 0] | (triplets[:, 1] << 8) | (triplets[:, 2] << 16)
        values = (values ^ 0x800000) - 0x800000
        audio = values.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported PCM WAV sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32, copy=False), sample_rate


def ensure_sample_rate(audio: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    if sample_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    gcd = np.gcd(sample_rate, target_rate)
    resampled = resample_poly(audio, target_rate // gcd, sample_rate // gcd)
    return resampled.astype(np.float32, copy=False)


def to_fixed_length(
    audio: np.ndarray,
    target_samples: int,
    *,
    random_crop: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Crop longer clips and loop-pad shorter clips to exactly target_samples."""
    if audio.size == 0:
        return np.zeros(target_samples, dtype=np.float32)

    if audio.size >= target_samples:
        if random_crop and audio.size > target_samples:
            rng = rng or np.random.default_rng()
            start = int(rng.integers(0, audio.size - target_samples + 1))
        else:
            start = max(0, (audio.size - target_samples) // 2)
        return audio[start : start + target_samples].astype(np.float32, copy=False)

    repeats = int(np.ceil(target_samples / audio.size))
    tiled = np.tile(audio, repeats)
    return tiled[:target_samples].astype(np.float32, copy=False)


def peak_normalize(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < eps:
        return np.zeros_like(audio, dtype=np.float32)
    return (audio / peak).astype(np.float32, copy=False)
