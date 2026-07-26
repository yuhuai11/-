from __future__ import annotations

import unittest

from dads_crnn.audit_g14_decode_preflight import (
    predicted_full_segments,
    predicted_resampled_samples,
)


class G14DecodePreflightTests(unittest.TestCase):
    def test_resample_length_uses_ceiling_like_resample_poly(self) -> None:
        self.assertEqual(predicted_resampled_samples(44101, 44100, 16000), 16001)

    def test_only_complete_nonoverlap_segments_are_counted(self) -> None:
        self.assertEqual(predicted_full_segments(441000, 44100, 16000, 16000), 10)
        self.assertEqual(predicted_full_segments(440997, 44100, 16000, 16000), 9)


if __name__ == "__main__":
    unittest.main()
