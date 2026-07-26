from __future__ import annotations

import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from dads_crnn.audit_g14_kielce_pilot import DATE_PATTERN, _device, _safe_files


class G14KielcePilotTests(unittest.TestCase):
    def test_recorder_is_derived_from_archive_path(self) -> None:
        self.assertEqual(_device("D1/NORSONIC140/a.wav"), "NORSONIC_140")
        self.assertEqual(_device("D1/OLYMPUSLS11/a.wav"), "OLYMPUS_LS11")
        self.assertEqual(_device("D1/other/a.wav"), "unknown")

    def test_zip_traversal_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.zip"
            with zipfile.ZipFile(path, "w") as handle:
                handle.writestr("../escape.wav", b"audio")
            with zipfile.ZipFile(path) as handle:
                with self.assertRaisesRegex(ValueError, "Unsafe Kielce"):
                    _safe_files(handle)

    def test_compact_acquisition_date_is_recognized(self) -> None:
        match = DATE_PATTERN.search("x4_d1_matrice300_10m_0m_230315_0031.wav")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(0), "230315")


if __name__ == "__main__":
    unittest.main()
