from __future__ import annotations

import io
import unittest
import wave

import numpy as np

from dads_crnn.audio import decode_wav_bytes


def _pcm24(value: int) -> bytes:
    unsigned = value & 0xFFFFFF
    return bytes(
        (unsigned & 0xFF, (unsigned >> 8) & 0xFF, (unsigned >> 16) & 0xFF)
    )


class Audio24BitTests(unittest.TestCase):
    def test_signed_little_endian_pcm24_decode(self) -> None:
        values = [-8388608, -1, 0, 1, 8388607]
        output = io.BytesIO()
        with wave.open(output, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(3)
            handle.setframerate(44100)
            handle.writeframes(b"".join(_pcm24(value) for value in values))
        audio, rate = decode_wav_bytes(output.getvalue())
        self.assertEqual(rate, 44100)
        np.testing.assert_allclose(
            audio,
            np.asarray(values, dtype=np.float32) / 8388608.0,
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
