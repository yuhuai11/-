from __future__ import annotations

import unittest

import numpy as np

from dads_crnn.audit_g17_high_rate import (
    decode_pcm,
    select_stratified_rows,
    spectral_band_features,
)


class G17HighRateAuditTests(unittest.TestCase):
    def test_stratified_selection_is_deterministic_and_bounded(self) -> None:
        rows = [
            {
                "split": "train",
                "label": "1",
                "source_group": "source-a",
                "audio_sha256": f"{index:064x}",
                "archive_member": f"{index}.wav",
            }
            for index in range(10)
        ]
        first = select_stratified_rows(rows, 3)
        second = select_stratified_rows(list(reversed(rows)), 3)
        self.assertEqual(
            [row["audio_sha256"] for row in first],
            [row["audio_sha256"] for row in second],
        )
        self.assertEqual(len(first), 3)

    def test_24_bit_pcm_decoding(self) -> None:
        raw = bytes((0, 0, 0, 255, 255, 127, 0, 0, 128))
        decoded = decode_pcm(raw, sample_width=3, channels=1)
        self.assertAlmostEqual(float(decoded[0]), 0.0)
        self.assertGreater(float(decoded[1]), 0.99)
        self.assertAlmostEqual(float(decoded[2]), -1.0)

    def test_high_frequency_tone_is_detected(self) -> None:
        sample_rate = 44100
        time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        low = spectral_band_features(np.sin(2 * np.pi * 1000 * time), sample_rate)
        high = spectral_band_features(np.sin(2 * np.pi * 10000 * time), sample_rate)
        self.assertLess(low["above_8000_ratio"], 1.0e-6)
        self.assertGreater(high["above_8000_ratio"], 0.99)


if __name__ == "__main__":
    unittest.main()
