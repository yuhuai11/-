from __future__ import annotations

import numpy as np


EPSILON = 1.0e-12


def _normalized_entropy(values: np.ndarray) -> float:
    probabilities = values / np.maximum(values.sum(), EPSILON)
    return float(
        -(probabilities * np.log(probabilities + EPSILON)).sum()
        / np.log(len(probabilities))
    )


def _rolloff_frequency(
    power: np.ndarray, frequencies: np.ndarray, fraction: float
) -> float:
    threshold = fraction * power.sum()
    index = int(np.searchsorted(np.cumsum(power), threshold, side="left"))
    return float(frequencies[min(index, len(frequencies) - 1)])


def _harmonic_profile(
    residual: np.ndarray,
    *,
    maximum_bin: int,
    minimum_f0: int,
    maximum_f0: int,
    harmonics: int,
) -> tuple[float, float, np.ndarray]:
    candidates = np.arange(minimum_f0, maximum_f0 + 1, dtype=np.int64)
    orders = np.arange(1, harmonics + 1, dtype=np.int64)
    indices = candidates[:, None] * orders[None, :]
    valid = indices <= maximum_bin
    clipped = np.minimum(indices, maximum_bin)
    values = residual[clipped] * valid
    weights = valid / np.sqrt(orders[None, :])
    scores = (values * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), EPSILON)
    best = int(np.argmax(scores))
    profile = np.where(valid[best], values[best], 0.0)
    return float(candidates[best]), float(scores[best]), profile


def extract_segment_harmonic_features(
    waveform: np.ndarray,
    *,
    sample_rate: int = 16000,
    n_fft: int = 16384,
    bands: int = 32,
    harmonics: int = 12,
) -> np.ndarray:
    """Extract deterministic rotor-oriented spectral and harmonic features."""
    values = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if values.size != sample_rate or not np.isfinite(values).all():
        raise ValueError("G22 expects one finite second of waveform")
    if n_fft < values.size or n_fft & (n_fft - 1):
        raise ValueError("G22 n_fft must be a power of two covering the waveform")
    centered = values - values.mean()
    spectrum = np.fft.rfft(centered * np.hanning(values.size), n=n_fft)
    power = np.square(np.abs(spectrum)) + EPSILON
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    valid = (frequencies >= 20.0) & (frequencies <= sample_rate / 2.0)
    selected_power = power[valid]
    selected_frequency = frequencies[valid]
    total = selected_power.sum()

    centroid = float((selected_frequency * selected_power).sum() / total)
    bandwidth = float(
        np.sqrt(
            (np.square(selected_frequency - centroid) * selected_power).sum() / total
        )
    )
    flatness = float(
        np.exp(np.log(selected_power).mean()) / np.maximum(selected_power.mean(), EPSILON)
    )
    global_features = np.asarray(
        [
            np.log10(np.mean(np.square(centered)) + EPSILON),
            centroid / 8000.0,
            bandwidth / 8000.0,
            flatness,
            _normalized_entropy(selected_power),
            _rolloff_frequency(selected_power, selected_frequency, 0.50) / 8000.0,
            _rolloff_frequency(selected_power, selected_frequency, 0.85) / 8000.0,
            _rolloff_frequency(selected_power, selected_frequency, 0.95) / 8000.0,
        ],
        dtype=np.float64,
    )

    edges = np.geomspace(20.0, 8000.0, bands + 1)
    band_features = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (frequencies >= low) & (frequencies < high)
        band_features.append(np.log10(power[mask].mean() + EPSILON))
    band_features = np.asarray(band_features, dtype=np.float64)
    band_features -= band_features.mean()

    log_power = np.log10(power)
    smooth = np.convolve(log_power, np.ones(41, dtype=np.float64) / 41.0, mode="same")
    residual = log_power - smooth
    maximum_bin = int(np.searchsorted(frequencies, 6000.0, side="right") - 1)
    bin_hz = sample_rate / n_fft
    f0, harmonic_score, profile = _harmonic_profile(
        residual,
        maximum_bin=maximum_bin,
        minimum_f0=max(1, int(np.ceil(40.0 / bin_hz))),
        maximum_f0=int(np.floor(500.0 / bin_hz)),
        harmonics=harmonics,
    )
    f0_hz = f0 * bin_hz
    profile_scale = np.maximum(np.std(residual[20:maximum_bin]), EPSILON)
    normalized_profile = profile / profile_scale
    harmonic_features = np.concatenate(
        (
            np.asarray(
                [
                    f0_hz / 500.0,
                    harmonic_score / profile_scale,
                    float((normalized_profile > 1.0).mean()),
                    float(np.mean(normalized_profile)),
                    float(np.std(normalized_profile)),
                ]
            ),
            normalized_profile,
        )
    )

    # Estimate short-time fundamental stability across four non-overlapping frames.
    frame_f0 = []
    frame_score = []
    frame_size = sample_rate // 4
    for start in range(0, sample_rate, frame_size):
        frame = centered[start : start + frame_size]
        frame_spectrum = np.fft.rfft(frame * np.hanning(frame_size), n=4096)
        frame_log_power = np.log10(np.square(np.abs(frame_spectrum)) + EPSILON)
        frame_smooth = np.convolve(
            frame_log_power, np.ones(21, dtype=np.float64) / 21.0, mode="same"
        )
        frame_residual = frame_log_power - frame_smooth
        frame_bin_hz = sample_rate / 4096
        value, score, _ = _harmonic_profile(
            frame_residual,
            maximum_bin=int(6000.0 / frame_bin_hz),
            minimum_f0=int(40.0 / frame_bin_hz),
            maximum_f0=int(500.0 / frame_bin_hz),
            harmonics=8,
        )
        frame_f0.append(value * frame_bin_hz)
        frame_score.append(score)
    temporal_features = np.asarray(
        [
            np.mean(frame_f0) / 500.0,
            np.std(frame_f0) / 500.0,
            np.mean(frame_score) / profile_scale,
            np.std(frame_score) / profile_scale,
        ]
    )
    result = np.concatenate(
        (global_features, band_features, harmonic_features, temporal_features)
    ).astype(np.float32)
    if result.shape != (61,) or not np.isfinite(result).all():
        raise RuntimeError(f"G22 produced invalid features: {result.shape}")
    return result


def aggregate_recording_features(
    frame,
    segment_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if len(frame) != len(segment_features) or segment_features.ndim != 2:
        raise ValueError("G22 recording aggregation inputs are misaligned")
    values = []
    targets = []
    identities = []
    for identity, indices in frame.groupby("audio_sha256", sort=True).indices.items():
        positions = np.asarray(indices, dtype=np.int64)
        labels = frame.iloc[positions]["target_index"].astype(int).unique()
        if len(labels) != 1:
            raise ValueError("G22 recording contains conflicting targets")
        current = segment_features[positions]
        values.append(np.concatenate((current.mean(axis=0), current.std(axis=0))))
        targets.append(int(labels[0]))
        identities.append(str(identity))
    result = np.asarray(values, dtype=np.float32)
    if result.shape[1] != segment_features.shape[1] * 2:
        raise RuntimeError("G22 recording feature shape changed")
    return result, np.asarray(targets, dtype=np.int64), identities


def normalize_class_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("G22 class scores must be a finite matrix")
    centered = values - values.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    return centered / np.maximum(scale, 1.0e-6)
