from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt

from .audio import peak_normalize


class WaveformAugmenter:
    """Training-only acoustic augmentation with auditable operation metadata."""

    def __init__(self, config: dict[str, Any], sample_rate: int, seed: int) -> None:
        self.config = config
        self.sample_rate = int(sample_rate)
        self.rng = np.random.default_rng(seed)
        self.frequency_response_mode = str(
            config.get("frequency_response_mode", "smooth_fft")
        ).strip()
        if self.frequency_response_mode not in {
            "smooth_fft",
            "structured_microphone_v1",
        }:
            raise ValueError(
                "frequency_response_mode must be smooth_fft or "
                "structured_microphone_v1"
            )
        uniform_snr = config.get("mix_snr_uniform_db")
        if uniform_snr is not None:
            if "mix_snr_db" in config or "mix_snr_weights" in config:
                raise ValueError(
                    "mix_snr_uniform_db is mutually exclusive with "
                    "mix_snr_db and mix_snr_weights"
                )
            interval = np.asarray(uniform_snr, dtype=np.float64)
            if (
                interval.shape != (2,)
                or not np.isfinite(interval).all()
                or interval[0] > interval[1]
            ):
                raise ValueError("mix_snr_uniform_db must be a finite increasing pair")
            self.snr_uniform_range: tuple[float, float] | None = (
                float(interval[0]),
                float(interval[1]),
            )
            levels = np.empty(0, dtype=np.float64)
            weights = np.empty(0, dtype=np.float64)
        else:
            try:
                levels = np.asarray(config["mix_snr_db"], dtype=np.float64)
                weights = np.asarray(config["mix_snr_weights"], dtype=np.float64)
            except KeyError as error:
                raise ValueError(
                    "configure either mix_snr_uniform_db or both "
                    "mix_snr_db and mix_snr_weights"
                ) from error
            if (
                len(levels) != len(weights)
                or len(levels) == 0
                or not np.isfinite(levels).all()
                or not np.isfinite(weights).all()
                or (weights < 0).any()
                or not np.isclose(weights.sum(), 1.0)
            ):
                raise ValueError(
                    "mix_snr_db and mix_snr_weights must have equal length "
                    "and weights sum to 1"
                )
            if len(np.unique(levels)) != len(levels):
                raise ValueError("mix_snr_db values must be unique")
            self.snr_uniform_range = None
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
        if self.frequency_response_mode == "smooth_fft":
            if int(config.get("frequency_response_anchors", 8)) < 2:
                raise ValueError("frequency_response_anchors must be at least 2")
            if float(config["frequency_response_std_db"]) < 0:
                raise ValueError("frequency_response_std_db must be non-negative")
            if float(config["frequency_response_limit_db"]) < 0:
                raise ValueError("frequency_response_limit_db must be non-negative")
        else:
            self._validate_structured_microphone_config()
        if float(config["max_time_shift_ms"]) < 0:
            raise ValueError("max_time_shift_ms must be non-negative")
        self.snr_levels = levels
        self.snr_weights = weights

    def _sample_mix_snr_db(self) -> float:
        if self.snr_uniform_range is not None:
            return float(self.rng.uniform(*self.snr_uniform_range))
        return float(self.rng.choice(self.snr_levels, p=self.snr_weights))

    def _config_pair(self, key: str) -> tuple[float, float]:
        if key not in self.config:
            raise ValueError(f"{key} is required for structured_microphone_v1")
        try:
            interval = np.asarray(self.config[key], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key} must be a finite increasing pair") from error
        if (
            interval.shape != (2,)
            or not np.isfinite(interval).all()
            or interval[0] > interval[1]
        ):
            raise ValueError(f"{key} must be a finite increasing pair")
        return float(interval[0]), float(interval[1])

    def _validate_structured_microphone_config(self) -> None:
        nyquist = self.sample_rate / 2.0
        highpass = self._config_pair("microphone_highpass_hz")
        lowpass = self._config_pair("microphone_lowpass_hz")
        centers = self._config_pair("microphone_peaking_center_hz")
        q_values = self._config_pair("microphone_peaking_q")
        gains = self._config_pair("microphone_peaking_gain_db")

        if highpass[0] < 60.0 or highpass[1] > 120.0 or highpass[0] <= 0.0:
            raise ValueError("microphone_highpass_hz must stay within [60, 120] Hz")
        if (
            lowpass[0] < 6000.0
            or lowpass[1] > 8000.0
            or lowpass[1] > nyquist
            or lowpass[0] >= nyquist
        ):
            raise ValueError(
                "microphone_lowpass_hz must stay within [6000, 8000] Hz "
                "and not exceed Nyquist"
            )
        if highpass[1] >= lowpass[0]:
            raise ValueError("microphone high-pass range must be below low-pass range")
        if centers[0] <= 0.0 or centers[1] >= nyquist:
            raise ValueError(
                "microphone_peaking_center_hz must be strictly between 0 and Nyquist"
            )
        if q_values[0] <= 0.0:
            raise ValueError("microphone_peaking_q values must be positive")
        if gains[0] < -4.0 or gains[1] > 4.0:
            raise ValueError("microphone_peaking_gain_db must stay within [-4, 4] dB")

        key = "microphone_peaking_sections"
        if key not in self.config:
            raise ValueError(f"{key} is required for structured_microphone_v1")
        raw_sections = np.asarray(self.config[key])
        try:
            sections = np.asarray(self.config[key], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key} must be the integer pair [2, 3]") from error
        if (
            raw_sections.shape != (2,)
            or sections.shape != (2,)
            or not np.isfinite(sections).all()
            or not np.equal(sections, np.floor(sections)).all()
            or sections[0] > sections[1]
            or sections[0] < 2
            or sections[1] > 3
        ):
            raise ValueError(f"{key} must be an increasing integer pair within [2, 3]")

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

    def _smooth_frequency_response(self, audio: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft(audio)
        bins = spectrum.size
        anchors = int(self.config.get("frequency_response_anchors", 8))
        gains_db = self.rng.normal(0.0, float(self.config["frequency_response_std_db"]), anchors)
        limit = float(self.config["frequency_response_limit_db"])
        gains_db = np.clip(gains_db, -limit, limit)
        curve_db = np.interp(np.arange(bins), np.linspace(0, bins - 1, anchors), gains_db)
        return np.fft.irfft(spectrum * (10.0 ** (curve_db / 20.0)), n=audio.size).astype(np.float32)

    @staticmethod
    def _peaking_eq_sos(
        center_hz: float,
        q_value: float,
        gain_db: float,
        sample_rate: int,
    ) -> np.ndarray:
        """Return one normalized RBJ peaking-EQ biquad as an SOS row."""
        amplitude = 10.0 ** (gain_db / 40.0)
        omega = 2.0 * np.pi * center_hz / sample_rate
        alpha = np.sin(omega) / (2.0 * q_value)
        cosine = np.cos(omega)
        b0 = 1.0 + alpha * amplitude
        b1 = -2.0 * cosine
        b2 = 1.0 - alpha * amplitude
        a0 = 1.0 + alpha / amplitude
        a1 = -2.0 * cosine
        a2 = 1.0 - alpha / amplitude
        return np.asarray(
            [b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0],
            dtype=np.float64,
        )

    def _sample_structured_microphone_parameters(self) -> dict[str, Any]:
        highpass = self._config_pair("microphone_highpass_hz")
        lowpass = self._config_pair("microphone_lowpass_hz")
        center_range = self._config_pair("microphone_peaking_center_hz")
        q_range = self._config_pair("microphone_peaking_q")
        gain_range = self._config_pair("microphone_peaking_gain_db")
        section_range = np.asarray(
            self.config["microphone_peaking_sections"], dtype=np.int64
        )

        highpass_hz = float(self.rng.uniform(*highpass))
        # The published range reaches 8 kHz, which is exactly Nyquist at the
        # G7 sample rate. scipy requires a digital cutoff strictly below it.
        lowpass_upper = min(lowpass[1], float(np.nextafter(self.sample_rate / 2.0, 0.0)))
        lowpass_hz = float(self.rng.uniform(lowpass[0], lowpass_upper))
        section_count = int(
            self.rng.integers(int(section_range[0]), int(section_range[1]) + 1)
        )
        peaking_eq = []
        for _ in range(section_count):
            center_hz = float(
                np.exp(
                    self.rng.uniform(
                        np.log(center_range[0]),
                        np.log(center_range[1]),
                    )
                )
            )
            peaking_eq.append(
                {
                    "center_hz": center_hz,
                    "q": float(self.rng.uniform(*q_range)),
                    "gain_db": float(self.rng.uniform(*gain_range)),
                }
            )
        return {
            "mode": "structured_microphone_v1",
            "highpass_hz": highpass_hz,
            "lowpass_hz": lowpass_hz,
            "peaking_center_distribution": "log_uniform",
            "peaking_eq": peaking_eq,
        }

    def _structured_microphone_response(
        self,
        audio: np.ndarray,
        parameters: dict[str, Any],
    ) -> np.ndarray:
        sos_rows = [
            butter(
                1,
                float(parameters["highpass_hz"]),
                btype="highpass",
                fs=self.sample_rate,
                output="sos",
            ),
            butter(
                1,
                float(parameters["lowpass_hz"]),
                btype="lowpass",
                fs=self.sample_rate,
                output="sos",
            ),
        ]
        for section in parameters["peaking_eq"]:
            sos_rows.append(
                self._peaking_eq_sos(
                    float(section["center_hz"]),
                    float(section["q"]),
                    float(section["gain_db"]),
                    self.sample_rate,
                )[None, :]
            )
        sos = np.concatenate(sos_rows, axis=0)
        filtered = sosfilt(sos, audio.astype(np.float64, copy=False))
        return filtered.astype(np.float32)

    def _frequency_response_with_metadata(
        self, audio: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any] | None]:
        if self.frequency_response_mode == "structured_microphone_v1":
            parameters = self._sample_structured_microphone_parameters()
            return self._structured_microphone_response(audio, parameters), parameters
        return self._smooth_frequency_response(audio), None

    def _frequency_response(self, audio: np.ndarray) -> np.ndarray:
        output, _ = self._frequency_response_with_metadata(audio)
        return output

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
            target_snr = self._sample_mix_snr_db()
            output, achieved = self._mix_background(output, background_sampler(), target_snr)
            if achieved is not None:
                metadata.update(
                    background_mix=True,
                    target_snr_db=target_snr,
                    achieved_snr_db=achieved,
                )
        if self.rng.random() < float(self.config["frequency_response_probability"]):
            output, response_metadata = self._frequency_response_with_metadata(output)
            metadata["frequency_response"] = True
            if response_metadata is not None:
                metadata["frequency_response_parameters"] = response_metadata
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
