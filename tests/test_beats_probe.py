import unittest

import pandas as pd

from dads_crnn.prepare_beats_probe import reject_locked_path, select_source_segments


class BeatsProbeTests(unittest.TestCase):
    def test_source_selection_uses_first_middle_last(self) -> None:
        frame = pd.DataFrame(
            {
                "split": ["train"] * 5,
                "source_path": ["source.wav"] * 5,
                "segment_index": [0, 1, 2, 3, 4],
                "start_sample": [0, 1, 2, 3, 4],
                "cache_path": [f"{index}.npy" for index in range(5)],
                "label": [1] * 5,
            }
        )
        selected = select_source_segments(frame, 3)
        self.assertEqual(selected["segment_index"].tolist(), [0, 2, 4])

    def test_locked_final_test_paths_are_rejected(self) -> None:
        from pathlib import Path

        for name in (
            "unseen_manifest.csv",
            "real_world_manifest.csv",
            "real-world.csv",
            "external_confirmation_v2_manifest.csv",
            "g13_external_confirmation.csv",
        ):
            with self.assertRaisesRegex(ValueError, "Locked"):
                reject_locked_path(Path(name))


if __name__ == "__main__":
    unittest.main()
