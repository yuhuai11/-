import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dads_crnn.dataset import DADSDataset


class G7R4DatasetTests(unittest.TestCase):
    def test_cache_offsets_expose_two_nonoverlapping_halves(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "audio.npy"
            np.save(cache, np.asarray([np.arange(16000, dtype=np.float32)]))
            manifest = root / "manifest.csv"
            pd.DataFrame(
                [
                    {"split": "train", "label": 1, "cache_path": str(cache), "cache_index": 0, "cache_start_sample": 0, "cache_end_sample": 8000},
                    {"split": "train", "label": 1, "cache_path": str(cache), "cache_index": 0, "cache_start_sample": 8000, "cache_end_sample": 16000},
                ]
            ).to_csv(manifest, index=False)
            dataset = DADSDataset(manifest, "train", sample_rate=16000, clip_seconds=0.5, training=False, seed=42)
            first = dataset._load_audio(dataset.rows.iloc[0])
            second = dataset._load_audio(dataset.rows.iloc[1])
            self.assertEqual(first.shape, (8000,))
            self.assertEqual(second.shape, (8000,))
            self.assertEqual(float(first[0]), 0.0)
            self.assertEqual(float(second[0]), 8000.0)

    def test_partial_offset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "audio.npy"
            np.save(cache, np.zeros((1, 16000), dtype=np.float32))
            manifest = root / "manifest.csv"
            pd.DataFrame([{"split": "train", "label": 0, "cache_path": str(cache), "cache_index": 0, "cache_start_sample": 0}]).to_csv(manifest, index=False)
            dataset = DADSDataset(manifest, "train", sample_rate=16000, clip_seconds=0.5, training=False, seed=42)
            with self.assertRaisesRegex(ValueError, "must be provided together"):
                dataset._load_audio(dataset.rows.iloc[0])


if __name__ == "__main__":
    unittest.main()
