from __future__ import annotations

import unittest

from dads_crnn.prepare_g18_model_id_registry import (
    cap_recording_segments,
    normalize_model,
    split_recordings,
)


class G18ModelIdRegistryTests(unittest.TestCase):
    def test_model_aliases_normalize(self) -> None:
        self.assertEqual(normalize_model("Mavic 3"), "MAVIC3")
        self.assertEqual(normalize_model("MAVIC3"), "MAVIC3")
        self.assertEqual(normalize_model("Yuneec-H520 ERTK"), "YUNEECH520ERTK")

    def test_recording_split_is_deterministic_and_disjoint(self) -> None:
        hashes = [f"{index:064x}" for index in range(20)]
        fractions = {"train": 0.7, "tune": 0.15, "holdout": 0.15}
        first = split_recordings(hashes, fractions, 42, "MAVIC3")
        second = split_recordings(list(reversed(hashes)), fractions, 42, "MAVIC3")
        self.assertEqual(first, second)
        self.assertFalse(first["train"] & first["tune"])
        self.assertFalse(first["train"] & first["holdout"])
        self.assertFalse(first["tune"] & first["holdout"])
        self.assertEqual(set().union(*first.values()), set(hashes))

    def test_long_recording_is_capped_deterministically(self) -> None:
        rows = [
            {"segment_index": str(index), "segment_sha256": f"{index:064x}"}
            for index in range(100)
        ]
        selected = cap_recording_segments(rows, 20)
        self.assertEqual(len(selected), 20)
        self.assertEqual(selected[0]["segment_index"], "0")
        self.assertEqual(selected[-1]["segment_index"], "99")


if __name__ == "__main__":
    unittest.main()
