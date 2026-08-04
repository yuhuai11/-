import unittest

import numpy as np

from dads_crnn.prepare_g7_r6_dronenoise_cache import (
    complete_window_count,
    segment_sha256,
)


class DroneNoiseCacheTests(unittest.TestCase):
    def test_complete_native_halfsecond_windows_only(self) -> None:
        self.assertEqual(complete_window_count(7_999), 0)
        self.assertEqual(complete_window_count(8_000), 1)
        self.assertEqual(complete_window_count(16_123), 2)

    def test_segment_hash_is_deterministic_and_order_sensitive(self) -> None:
        waveform = np.arange(8_000, dtype=np.float32)
        self.assertEqual(segment_sha256(waveform), segment_sha256(waveform.copy()))
        self.assertNotEqual(segment_sha256(waveform), segment_sha256(waveform[::-1]))


if __name__ == "__main__":
    unittest.main()
