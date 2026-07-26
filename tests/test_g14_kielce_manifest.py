from __future__ import annotations

import unittest

from dads_crnn.prepare_g14_kielce_manifest import (
    ARCHIVE_PATTERN,
    MEMBER_PATTERN,
)


class G14KielceManifestTests(unittest.TestCase):
    def test_archive_fields(self) -> None:
        match = ARCHIVE_PATTERN.match("X6_D13_YuneecH520ERTK.zip")
        self.assertEqual(
            match.groupdict(),
            {"rotor": "X6", "drone": "D13", "vehicle": "YuneecH520ERTK"},
        )

    def test_member_fields(self) -> None:
        match = MEMBER_PATTERN.search(
            "x4_d1_matrice300_10m_8m_230315_0043.wav"
        )
        self.assertEqual(
            match.groupdict(),
            {
                "height": "10",
                "distance": "8",
                "date": "230315",
                "measurement": "0043",
            },
        )

    def test_speech_suffix_is_parseable_but_remains_excludable(self) -> None:
        match = MEMBER_PATTERN.search(
            "x4_d10_mavic2zoom_10m_0m_230417_0026_sekwencja.WAV"
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.group("measurement"), "0026")
        double_separator = MEMBER_PATTERN.search(
            "x4_d6_mavic2pro_5m_8m_230415__0070_sekwencja.WAV"
        )
        self.assertEqual(double_separator.group("measurement"), "0070")


if __name__ == "__main__":
    unittest.main()
