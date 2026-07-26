from __future__ import annotations

import unittest

from dads_crnn.prepare_g14_tau_manifest import _group_rows


class G14TauManifestTests(unittest.TestCase):
    def test_group_rows_uses_requested_fields(self) -> None:
        rows = [
            {"split": "train", "source_group": "a"},
            {"split": "train", "source_group": "a"},
            {"split": "tune", "source_group": "b"},
        ]
        grouped = _group_rows(rows, ("split", "source_group"))
        self.assertEqual(len(grouped[("train", "a")]), 2)
        self.assertEqual(len(grouped[("tune", "b")]), 1)


if __name__ == "__main__":
    unittest.main()
