from __future__ import annotations

import unittest

import numpy as np

from dads_crnn.augmentation import WaveformAugmenter


def _base_config() -> dict:
    return {
        "positive_mix_probability": 0.0,
        "mix_snr_db": [-5.0, 0.0, 5.0, 10.0],
        "mix_snr_weights": [0.10, 0.25, 0.35, 0.30],
        "frequency_response_probability": 1.0,
        "frequency_response_anchors": 8,
        "frequency_response_std_db": 3.0,
        "frequency_response_limit_db": 6.0,
        "reverb_probability": 0.0,
        "reverb_delay_ms": [12.0, 80.0],
        "reverb_gain": [0.05, 0.25],
        "colored_noise_probability": 0.0,
        "colored_noise_snr_db": [10.0, 30.0],
        "time_shift_probability": 0.0,
        "max_time_shift_ms": 100.0,
    }


def _structured_config() -> dict:
    config = _base_config()
    config.update(
        frequency_response_mode="structured_microphone_v1",
        microphone_highpass_hz=[60.0, 120.0],
        microphone_lowpass_hz=[6000.0, 8000.0],
        microphone_peaking_sections=[2, 3],
        microphone_peaking_center_hz=[120.0, 6000.0],
        microphone_peaking_q=[0.5, 2.0],
        microphone_peaking_gain_db=[-4.0, 4.0],
    )
    return config


class G7R1MicrophoneResponseTests(unittest.TestCase):
    @staticmethod
    def _signal() -> np.ndarray:
        time = np.arange(16000, dtype=np.float64) / 16000.0
        return (
            0.50 * np.sin(2.0 * np.pi * 40.0 * time)
            + 0.35 * np.sin(2.0 * np.pi * 700.0 * time)
            + 0.15 * np.sin(2.0 * np.pi * 7000.0 * time)
        ).astype(np.float32)

    def test_default_mode_is_bit_exact_with_legacy_smooth_fft(self) -> None:
        config = _base_config()
        seed = 1701
        audio = self._signal()
        reference_rng = np.random.default_rng(seed)
        spectrum = np.fft.rfft(audio)
        bins = spectrum.size
        gains_db = reference_rng.normal(
            0.0,
            float(config["frequency_response_std_db"]),
            int(config["frequency_response_anchors"]),
        )
        gains_db = np.clip(
            gains_db,
            -float(config["frequency_response_limit_db"]),
            float(config["frequency_response_limit_db"]),
        )
        curve_db = np.interp(
            np.arange(bins),
            np.linspace(0, bins - 1, int(config["frequency_response_anchors"])),
            gains_db,
        )
        expected = np.fft.irfft(
            spectrum * (10.0 ** (curve_db / 20.0)), n=audio.size
        ).astype(np.float32)

        augmenter = WaveformAugmenter(config, sample_rate=16000, seed=seed)
        observed = augmenter._frequency_response(audio)

        self.assertEqual(augmenter.frequency_response_mode, "smooth_fft")
        np.testing.assert_array_equal(observed, expected)
        _, legacy_metadata = WaveformAugmenter(
            config, sample_rate=16000, seed=seed
        ).apply(audio, label=0)
        self.assertNotIn("frequency_response_parameters", legacy_metadata)

    def test_structured_response_is_deterministic_finite_and_peak_normalized(self) -> None:
        audio = self._signal()
        first = WaveformAugmenter(_structured_config(), sample_rate=16000, seed=42)
        second = WaveformAugmenter(_structured_config(), sample_rate=16000, seed=42)

        first_output, first_metadata = first.apply(audio, label=0)
        second_output, second_metadata = second.apply(audio, label=0)

        np.testing.assert_array_equal(first_output, second_output)
        self.assertEqual(first_output.shape, audio.shape)
        self.assertEqual(first_output.dtype, np.float32)
        self.assertTrue(np.isfinite(first_output).all())
        self.assertLessEqual(float(np.max(np.abs(first_output))), 1.0)
        self.assertTrue(first_metadata["frequency_response"])
        self.assertEqual(first_metadata, second_metadata)
        self.assertFalse(np.array_equal(first_output, audio))

        parameters = first_metadata["frequency_response_parameters"]
        self.assertEqual(parameters["mode"], "structured_microphone_v1")
        self.assertEqual(parameters["peaking_center_distribution"], "log_uniform")
        self.assertIn(len(parameters["peaking_eq"]), (2, 3))
        self.assertGreaterEqual(parameters["highpass_hz"], 60.0)
        self.assertLessEqual(parameters["highpass_hz"], 120.0)
        self.assertGreaterEqual(parameters["lowpass_hz"], 6000.0)
        self.assertLess(parameters["lowpass_hz"], 8000.0)
        for section in parameters["peaking_eq"]:
            self.assertGreaterEqual(section["center_hz"], 120.0)
            self.assertLessEqual(section["center_hz"], 6000.0)
            self.assertGreaterEqual(section["q"], 0.5)
            self.assertLessEqual(section["q"], 2.0)
            self.assertGreaterEqual(section["gain_db"], -4.0)
            self.assertLessEqual(section["gain_db"], 4.0)

    def test_log_uniform_parameters_are_reproducible_and_auditable(self) -> None:
        seed = 1702
        augmenter = WaveformAugmenter(
            _structured_config(), sample_rate=16000, seed=seed
        )
        _, metadata = augmenter.apply(self._signal(), label=0)
        observed = metadata["frequency_response_parameters"]

        reference_rng = np.random.default_rng(seed)
        reference_rng.random()  # frequency_response_probability gate
        expected_highpass = float(reference_rng.uniform(60.0, 120.0))
        expected_lowpass = float(
            reference_rng.uniform(6000.0, np.nextafter(8000.0, 0.0))
        )
        expected_sections = int(reference_rng.integers(2, 4))
        expected_eq = []
        for _ in range(expected_sections):
            expected_eq.append(
                {
                    "center_hz": float(
                        np.exp(
                            reference_rng.uniform(
                                np.log(120.0),
                                np.log(6000.0),
                            )
                        )
                    ),
                    "q": float(reference_rng.uniform(0.5, 2.0)),
                    "gain_db": float(reference_rng.uniform(-4.0, 4.0)),
                }
            )

        self.assertEqual(observed["highpass_hz"], expected_highpass)
        self.assertEqual(observed["lowpass_hz"], expected_lowpass)
        self.assertEqual(observed["peaking_eq"], expected_eq)

    def test_structured_response_handles_silence(self) -> None:
        augmenter = WaveformAugmenter(
            _structured_config(), sample_rate=16000, seed=43
        )
        output, metadata = augmenter.apply(
            np.zeros(16000, dtype=np.float32), label=1
        )
        np.testing.assert_array_equal(output, np.zeros_like(output))
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue(metadata["frequency_response"])

    def test_structured_response_rejects_invalid_parameters(self) -> None:
        cases = (
            ("frequency_response_mode", "unknown"),
            ("microphone_highpass_hz", [59.0, 120.0]),
            ("microphone_lowpass_hz", [6000.0, 8001.0]),
            ("microphone_peaking_sections", [1, 3]),
            ("microphone_peaking_sections", [2.0, 3.5]),
            ("microphone_peaking_center_hz", [0.0, 6000.0]),
            ("microphone_peaking_center_hz", [120.0, 8000.0]),
            ("microphone_peaking_q", [0.0, 2.0]),
            ("microphone_peaking_q", "invalid"),
            ("microphone_peaking_gain_db", [-4.1, 4.0]),
        )
        for key, value in cases:
            with self.subTest(key=key, value=value):
                config = _structured_config()
                config[key] = value
                with self.assertRaises(ValueError):
                    WaveformAugmenter(config, sample_rate=16000, seed=42)

        missing = _structured_config()
        del missing["microphone_peaking_q"]
        with self.assertRaises(ValueError):
            WaveformAugmenter(missing, sample_rate=16000, seed=42)


if __name__ == "__main__":
    unittest.main()
