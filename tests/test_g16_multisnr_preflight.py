from __future__ import annotations

import unittest

import numpy as np
import torch

from dads_crnn.preflight_g16_multisnr import make_multisnr_pair_batch


class G16MultiSnrPreflightTests(unittest.TestCase):
    def test_multisnr_pairs_share_exact_negative_and_hit_snr(self) -> None:
        time = np.linspace(0, 1, 16000, endpoint=False)
        background = np.sin(2 * np.pi * 220 * time).astype(np.float32)
        uav = np.sin(2 * np.pi * 880 * time).astype(np.float32)
        negative, positive, diagnostics = make_multisnr_pair_batch(
            torch.from_numpy(np.stack((background, background * 0.8))),
            torch.from_numpy(np.stack((uav, uav * 0.7))),
            target_snr_db=[-15.0, -10.0, -5.0, 0.0],
            epsilon=1.0e-8,
            peak_limit=0.99,
        )
        self.assertEqual(negative.shape, (8, 16000))
        self.assertEqual(positive.shape, (8, 16000))
        for start in (0, 4):
            for offset in range(1, 4):
                self.assertTrue(
                    torch.equal(negative[start], negative[start + offset])
                )
        self.assertLess(
            max(
                abs(item["achieved_snr_db"] - item["target_snr_db"])
                for item in diagnostics
            ),
            1.0e-4,
        )

    def test_multisnr_grid_must_be_unique_and_increasing(self) -> None:
        values = torch.ones((1, 16), dtype=torch.float32)
        with self.assertRaises(ValueError):
            make_multisnr_pair_batch(
                values,
                values,
                target_snr_db=[-10.0, -15.0],
                epsilon=1.0e-8,
                peak_limit=0.99,
            )


if __name__ == "__main__":
    unittest.main()
