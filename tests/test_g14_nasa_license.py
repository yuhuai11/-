from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from dads_crnn.audit_g14_nasa_license import audit


def _write_inputs(root: Path) -> tuple[Path, Path]:
    config = {
        "sources": {
            "nasa_suas": {
                "license": {
                    "training_allowed": False,
                }
            }
        }
    }
    evidence = {
        "protocol": "g14_nasa_license_evidence_v1",
        "dataset": {
            "name": "NASA Small UAS Flyover Acoustics Data",
            "identifier": "6s8fb29q",
            "access_level": "public",
            "dataset_specific_license": "not_specified",
            "catalog_url": "https://example.invalid/catalog",
            "nasa_portal_url": "https://example.invalid/portal",
        },
        "policy": {
            "url": "https://example.invalid/policy",
            "relevant_rule": "conditional_cc0",
            "applicability_to_this_dataset": "not_explicitly_established",
        },
        "contact": {"name": "contact", "email": "contact@example.invalid"},
        "decision": {
            "status": "confirmation_required",
            "training_allowed": False,
            "extraction_for_integrity_audit_allowed": True,
            "acceptable_resolution": ["written_confirmation"],
            "prohibited_until_resolution": ["model_training"],
        },
    }
    config_path = root / "config.yaml"
    evidence_path = root / "evidence.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    evidence_path.write_text(yaml.safe_dump(evidence), encoding="utf-8")
    return config_path, evidence_path


class G14NasaLicenseTests(unittest.TestCase):
    def test_unresolved_evidence_keeps_training_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, evidence_path = _write_inputs(root)
            report = audit(config_path, evidence_path, root)
            self.assertTrue(report["passed"])
            self.assertFalse(report["ready_for_training"])
            self.assertEqual(report["decision"]["status"], "confirmation_required")

    def test_unresolved_evidence_cannot_authorize_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, evidence_path = _write_inputs(root)
            evidence = yaml.safe_load(evidence_path.read_text(encoding="utf-8"))
            evidence["decision"]["training_allowed"] = True
            evidence_path.write_text(yaml.safe_dump(evidence), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot authorize training"):
                audit(config_path, evidence_path, root)


if __name__ == "__main__":
    unittest.main()
