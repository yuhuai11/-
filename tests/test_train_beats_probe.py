import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dads_crnn.train_beats_probe import _load_source_embeddings


class TrainBeatsProbeTests(unittest.TestCase):
    def test_source_embeddings_are_mean_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.zeros((3, 768), dtype=np.float32)
            values[0, 0] = 1.0
            values[1, 0] = 3.0
            values[2, 0] = 8.0
            np.save(root / "embeddings.npy", values)
            np.save(root / "labels.npy", np.array([0, 0, 1]))
            pd.DataFrame(
                {"source_path": ["a.wav", "a.wav", "b.wav"], "label": [0, 0, 1]}
            ).to_csv(root / "metadata.csv", index=False)
            embeddings, labels, rows = _load_source_embeddings(root)
            self.assertEqual(embeddings[:, 0].tolist(), [2.0, 8.0])
            self.assertEqual(labels.tolist(), [0, 1])
            self.assertEqual(rows["segments"].tolist(), [2, 1])


if __name__ == "__main__":
    unittest.main()
