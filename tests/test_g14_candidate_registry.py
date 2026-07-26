from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from dads_crnn.prepare_g14_candidate_registry import prepare


def _config(root: Path) -> Path:
    config = {
        "protocol": "g14_multisource_intake_v1",
        "output_dir": "artifacts/g14_domain_generalization/intake",
        "sources": {
            "positive": {
                "role": "uav_primary_outdoor_multimodel",
                "intake_class": "positive_uav",
                "selected": True,
                "status": "conditional_license_review",
                "dataset_name": "positive",
                "archive_path": "data/g14/positive.zip",
                "expected_bytes": 10,
                "license": {
                    "access_level": "public",
                    "training_allowed": False,
                },
            },
            "background": {
                "role": "multidevice_background",
                "intake_class": "negative_background",
                "selected": True,
                "status": "metadata_pending",
                "dataset_name": "background",
                "metadata_path": "data/g14/background.zip",
                "metadata_expected_md5": "d41d8cd98f00b204e9800998ecf8427e",
                "license": {
                    "access_level": "open",
                    "training_allowed": True,
                },
            },
        },
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


class G14CandidateRegistryTests(unittest.TestCase):
    def test_missing_files_allow_download_but_not_intake_or_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _config(root)
            report = prepare(path, root)
            self.assertTrue(report["passed"])
            self.assertTrue(report["selection"]["ready_for_download"])
            self.assertFalse(report["selection"]["ready_for_intake"])
            self.assertFalse(report["selection"]["ready_for_training"])
            self.assertEqual(
                report["selection"]["unresolved_training_licenses"], ["positive"]
            )
            self.assertEqual(report["selection"]["coverage_blockers"], [])
            self.assertEqual(report["selection"]["pilot_only_sources"], [])

    def test_pilot_only_source_never_authorizes_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _config(root)
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            config["sources"]["positive"]["pilot_only"] = True
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            report = prepare(path, root)
            self.assertEqual(report["selection"]["pilot_only_sources"], ["positive"])
            self.assertFalse(report["selection"]["ready_for_training"])

    def test_source_stage_blocker_never_authorizes_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _config(root)
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            config["sources"]["positive"]["stage_blockers"] = ["manifest_pending"]
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            report = prepare(path, root)
            self.assertEqual(
                report["selection"]["source_stage_blockers"],
                {"positive": ["manifest_pending"]},
            )
            self.assertFalse(report["selection"]["ready_for_training"])

    def test_locked_destination_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _config(root)
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            config["sources"]["positive"]["archive_path"] = (
                "data/external_confirmation_v2/copied.zip"
            )
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Locked final-test path"):
                prepare(path, root)


if __name__ == "__main__":
    unittest.main()
