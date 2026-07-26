from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dads_crnn.data_firewall import (
    audit_csv_rows,
    load_forbidden_hashes,
    reject_locked_path,
    reject_locked_value,
)


class DataFirewallTests(unittest.TestCase):
    def test_path_aliases_and_symlink_targets_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "EXTERNAL-CONFIRMATION-V2.csv",
                "g13 external confirmation.csv",
                "Real World.csv",
                "UNSEEN.csv",
                "test_augmented.csv",
                "raw-recorded-audios.csv",
            ):
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, "Locked final-test path"):
                        reject_locked_path(root / name)
            locked = root / "external_confirmation_v2.csv"
            locked.write_text("x\n", encoding="utf-8")
            alias = root / "apparently_safe.csv"
            alias.symlink_to(locked)
            with self.assertRaisesRegex(ValueError, "Locked final-test path"):
                reject_locked_path(alias)

    def test_value_normalization_rejects_path_variants(self) -> None:
        for value in (
            "External Confirmation V2",
            "G13-external_confirmation",
            "REAL/WORLD",
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Locked final-test value"):
                    reject_locked_value(value, context="unit test")

    def test_csv_rejects_consumed_hash_after_copy_or_rename(self) -> None:
        consumed_hash = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "hashes.txt"
            registry.write_text(f"{consumed_hash}\n", encoding="utf-8")
            manifest = root / "safe_name.csv"
            manifest.write_text(
                "dataset,path,label,sha256\n"
                f"new_dataset,/new/location/copied.wav,1,{consumed_hash}\n",
                encoding="utf-8",
            )
            hashes = load_forbidden_hashes(registry)
            with self.assertRaisesRegex(ValueError, "Consumed final-test audio hash"):
                audit_csv_rows(manifest, forbidden_hashes=hashes)

    def test_safe_csv_passes_and_returns_row_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            manifest.write_text(
                "dataset,path,label,sha256\n"
                f"g14_dev,/new/location/sample.wav,1,{'b' * 64}\n",
                encoding="utf-8",
            )
            self.assertEqual(
                audit_csv_rows(manifest, forbidden_hashes=frozenset()),
                1,
            )


if __name__ == "__main__":
    unittest.main()
