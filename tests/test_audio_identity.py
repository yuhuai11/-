from __future__ import annotations

import io
import unittest

import numpy as np
from scipy.io import wavfile

from dads_crnn.audio_identity import (
    decode_wav_channel_variants,
    quantized_pcm_sha256,
    spectral_fingerprint,
)


class AudioIdentityTests(unittest.TestCase):
    def test_channel_variants_include_mean_and_individual_channels(self) -> None:
        values = np.asarray([[0.25, -0.25], [0.75, 0.25]], dtype=np.float32)
        output = io.BytesIO()
        wavfile.write(output, 48_000, values)

        variants, rate = decode_wav_channel_variants(output.getvalue())

        self.assertEqual(rate, 48_000)
        self.assertEqual(set(variants), {"mono_mean", "channel_0", "channel_1"})
        np.testing.assert_allclose(variants["mono_mean"], [0.0, 0.5])

    def test_gain_normalized_hash_is_amplitude_invariant(self) -> None:
        audio = np.sin(np.linspace(0.0, 10.0, 16_000, dtype=np.float32))
        first = quantized_pcm_sha256(audio, 16_000, normalize_gain=True)
        second = quantized_pcm_sha256(audio * 0.5, 16_000, normalize_gain=True)
        self.assertEqual(first, second)
        self.assertNotEqual(
            quantized_pcm_sha256(audio, 16_000, normalize_gain=False),
            quantized_pcm_sha256(audio * 0.5, 16_000, normalize_gain=False),
        )

    def test_spectral_fingerprint_is_gain_invariant(self) -> None:
        audio = np.sin(
            2.0 * np.pi * 440.0 * np.arange(16_000, dtype=np.float32) / 16_000
        )
        first = spectral_fingerprint(audio, 16_000)
        second = spectral_fingerprint(audio * 0.1, 16_000)
        np.testing.assert_allclose(first, second, atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(first)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
