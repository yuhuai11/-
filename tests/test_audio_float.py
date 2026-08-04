from __future__ import annotations

import io
import unittest

import numpy as np
from scipy.io import wavfile

from dads_crnn.audio import decode_wav_bytes


class AudioFloatTests(unittest.TestCase):
    def test_decodes_float32_stereo_to_clipped_mono(self) -> None:
        stereo = np.asarray(
            [
                [-0.75, 0.25],
                [0.50, 1.50],
                [-1.50, -0.50],
            ],
            dtype=np.float32,
        )
        output = io.BytesIO()
        wavfile.write(output, 48_000, stereo)

        audio, rate = decode_wav_bytes(output.getvalue())

        self.assertEqual(rate, 48_000)
        self.assertEqual(audio.dtype, np.float32)
        np.testing.assert_allclose(
            audio,
            np.asarray([-0.25, 1.0, -1.0], dtype=np.float32),
            rtol=0.0,
            atol=1e-7,
        )

    def test_rejects_non_finite_float_audio(self) -> None:
        output = io.BytesIO()
        wavfile.write(output, 16_000, np.asarray([0.0, np.nan], dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            decode_wav_bytes(output.getvalue())


if __name__ == "__main__":
    unittest.main()
