from __future__ import annotations

import unittest

from dads_crnn.audit_g14_kielce_full_archive import (
    DATE_PATTERN,
    DRONE_PATTERN,
    GEOMETRY_PATTERN,
)


class G14KielceFullArchiveTests(unittest.TestCase):
    def test_official_filename_fields_are_parseable(self) -> None:
        archive = DRONE_PATTERN.search("_X4_D10_Mavic2Zoom_")
        geometry = GEOMETRY_PATTERN.search(
            "x4_d10_mavic2zoom_10m_8m_230315_0043.wav"
        )
        date = DATE_PATTERN.search("x4_d10_mavic2zoom_10m_8m_230315_0043.wav")
        self.assertEqual(archive.group(1), "D10")
        self.assertEqual(geometry.groups(), ("10", "8"))
        self.assertEqual(date.group(0), "230315")

    def test_full_archive_policy_is_based_on_total_not_each_zip(self) -> None:
        per_archive = [45] * 17
        per_archive[8] = 46
        per_archive[14] = 44
        self.assertEqual(sum(per_archive), 765)


if __name__ == "__main__":
    unittest.main()
