from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from dads_crnn.audit_g14_tau_intake import _safe_members


class G14TauIntakeTests(unittest.TestCase):
    def test_safe_zip_members_are_returned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "safe.zip"
            with zipfile.ZipFile(path, "w") as handle:
                handle.writestr("TAU/audio/airport-file.wav", b"audio")
            with zipfile.ZipFile(path) as handle:
                self.assertEqual(
                    _safe_members(handle), ["TAU/audio/airport-file.wav"]
                )

    def test_parent_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.zip"
            with zipfile.ZipFile(path, "w") as handle:
                handle.writestr("../escape.wav", b"audio")
            with zipfile.ZipFile(path) as handle:
                with self.assertRaisesRegex(ValueError, "Unsafe TAU ZIP member"):
                    _safe_members(handle)


if __name__ == "__main__":
    unittest.main()
