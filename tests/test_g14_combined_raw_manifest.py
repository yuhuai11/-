from __future__ import annotations

import unittest

from dads_crnn.prepare_g14_combined_raw_manifest import (
    _normalize_background,
    _normalize_positive,
)


class G14CombinedRawManifestTests(unittest.TestCase):
    def test_normalized_rows_share_the_same_schema(self) -> None:
        positive = _normalize_positive(
            {
                "dataset": "k",
                "archive_path": "k.zip",
                "archive_member": "a.wav",
                "split": "train",
                "source_group": "k:d1",
                "audio_sha256": "a" * 64,
                "uncompressed_bytes": "10",
                "location": "x",
                "vehicle": "v",
                "acquisition_date": "2023-01-01",
                "rotor_layout": "X4",
                "height_m": "5",
                "distance_m": "0",
                "license": "cc",
            }
        )
        background = _normalize_background(
            {
                "dataset": "t",
                "archive_path": "t.zip",
                "archive_member": "b.wav",
                "split": "train",
                "source_group": "t:s1",
                "audio_sha256": "b" * 64,
                "uncompressed_bytes": "20",
                "city": "c",
                "scene_label": "airport",
                "device": "a",
                "license": "nc",
            }
        )
        self.assertEqual(set(positive), set(background))
        self.assertEqual((positive["label"], background["label"]), (1, 0))


if __name__ == "__main__":
    unittest.main()
