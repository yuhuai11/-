from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .audio import peak_normalize


class WaveformAugmenter:
    """Training-only acoustic augmentation with auditable operation metadata."""

    def __init__(self, config: dict[str, Any], sample_rate: int, seed: int) -> None:
        self.config = config
        self.sample_rate = int(sample_rate)
        self.rng = np.random.default_rng(seed)
        levels = np.asarray(config["mix_snr_db"], dtype=np.float64)
        weights = np.asarray(config["mix_snr_weights"], dtype=np.float64)
        if (
            len(levels) != len(weights)
            or len(levels) == 0
            or not np.isfinite(levels).all()
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or not np.isclose(weights.sum(), 1.0)
        ):
            raise ValueError("mix_snr_db and mix_snr_weights must have equal length and weights sum to 1")
        if len(np.unique(levels)) != len(levels):
            raise ValueError("mix_snr_db values must be unique")
        for key in (
            "positive_mix_probability",
            "frequency_response_probability",
            "reverb_probability",
            "colored_noise_probability",
            "time_shift_probability",
        ):
            value = float(config[key])
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{key} must be a finite probability in [0, 1]")
        for key in ("reverb_delay_ms", "reverb_gain", "colored_noise_snr_db"):
            interval = np.asarray(config[key], dtype=np.float64)
            if interval.shape != (2,) or not np.isfinite(interval).all() or interval[0] > interval[1]:
                raise ValueError(f"{key} must be a finite increasing pair")
        if int(config.get("frequency_response_anchors", 8)) < 2:
            raise ValueError("frequency_response_anchors must be at least 2")
        if float(config["frequency_response_std_db"]) < 0:
            raise ValueError("frequency_response_std_db must be non-negative")
        if float(config["frequency_response_limit_db"]) < 0:
            raise ValueError("frequency_response_limit_db must be non-negative")
        if float(config["max_time_shift_ms"]) < 0:
            raise ValueError("max_time_shift_ms must be non-negative")
        self.snr_levels = levels
        self.snr_weights = weights

    @staticmethod
    def _rms(audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))

    def _mix_background(
        self, audio: np.ndarray, background: np.ndarray, target_snr_db: float
    ) -> tuple[np.ndarray, float | None]:
        signal_rms = self._rms(audio)
        background_rms = self._rms(background)
        if signal_rms <= 1e-8 or background_rms <= 1e-8:
            return audio, None
        scale = signal_rms / (background_rms * (10.0 ** (target_snr_db / 20.0)))
        scaled_background = background * scale
        achieved = 20.0 * np.log10(signal_rms / self._rms(scaled_background))
        return audio + scaled_background, float(achieved)

    def _frequency_response(self, audio: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft(audio)
        bins = spectrum.size
        anchors = int(self.config.get("frequency_response_anchors", 8))
        gains_db = self.rng.normal(0.0, float(self.config["frequency_response_std_db"]), anchors)
        limit = float(self.config["frequency_response_limit_db"])
        gains_db = np.clip(gains_db, -limit, limit)
        curve_db = np.interp(np.arange(bins), np.linspace(0, bins - 1, anchors), gains_db)
        return np.fft.irfft(spectrum * (10.0 ** (curve_db / 20.0)), n=audio.size).astype(np.float32)

    def _reverberate(self, audio: np.ndarray) -> np.ndarray:
        output = audio.astype(np.float32, copy=True)
        for _ in range(int(self.rng.integers(1, 4))):
            delay_ms = self.rng.uniform(
                float(self.config["reverb_delay_ms"][0]),
                float(self.config["reverb_delay_ms"][1]),
            )
            delay = max(1, int(self.sample_rate * delay_ms / 1000.0))
            gain = float(self.rng.uniform(*self.config["reverb_gain"]))
            if delay < audio.size:
                output[delay:] += gain * audio[:-delay]
        return output

    def _colored_noise(self, samples: int) -> np.ndarray:
        white = self.rng.normal(0.0, 1.0, samples)
        spectrum = np.fft.rfft(white)
        frequencies = np.arange(spectrum.size, dtype=np.float64)
        frequencies[0] = 1.0
        spectrum /= np.sqrt(frequencies)
        noise = np.fft.irfft(spectrum, n=samples).astype(np.float32)
        return noise / max(self._rms(noise), 1e-8)

    def _add_noise(self, audio: np.ndarray, snr_db: float) -> np.ndarray:
        noise = self._colored_noise(audio.size)
        noise_scale = self._rms(audio) / (self._rms(noise) * (10.0 ** (snr_db / 20.0)))
        return audio + noise * noise_scale

    def _time_shift(self, audio: np.ndarray) -> np.ndarray:
        maximum = int(self.sample_rate * float(self.config["max_time_shift_ms"]) / 1000.0)
        shift = int(self.rng.integers(-maximum, maximum + 1))
        if shift == 0:
            return audio
        output = np.zeros_like(audio)
        if shift > 0:
            output[shift:] = audio[:-shift]
        else:
            output[:shift] = audio[-shift:]
        return output

    def apply(
        self,
        audio: np.ndarray,
        label: int,
        background_sampler: Callable[[], np.ndarray] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        output = audio.astype(np.float32, copy=True)
        metadata: dict[str, Any] = {
            "background_mix": False,
            "target_snr_db": None,
            "achieved_snr_db": None,
            "frequency_response": False,
            "reverb": False,
            "colored_noise": False,
            "noise_snr_db": None,
            "time_shift": False,
        }
        if (
            label == 1
            and background_sampler is not None
            and self.rng.random() < float(self.config["positive_mix_probability"])
        ):
            target_snr = float(self.rng.choice(self.snr_levels, p=self.snr_weights))
            output, achieved = self._mix_background(output, background_sampler(), target_snr)
            if achieved is not None:
                metadata.update(
                    background_mix=True,
                    target_snr_db=target_snr,
                    achieved_snr_db=achieved,
                )
        if self.rng.random() < float(self.config["frequency_response_probability"]):
            output = self._frequency_response(output)
            metadata["frequency_response"] = True
        if self.rng.random() < float(self.config["reverb_probability"]):
            output = self._reverberate(output)
            metadata["reverb"] = True
        if self.rng.random() < float(self.config["colored_noise_probability"]):
            noise_snr = float(self.rng.uniform(*self.config["colored_noise_snr_db"]))
            output = self._add_noise(output, noise_snr)
            metadata.update(colored_noise=True, noise_snr_db=noise_snr)
        if self.rng.random() < float(self.config["time_shift_probability"]):
            output = self._time_shift(output)
            metadata["time_shift"] = True
        return peak_normalize(output), metadata
