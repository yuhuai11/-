from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .data_firewall import file_sha256


PROTOCOL = "g14_nasa_license_gate_v1"


def audit(config_path: Path, evidence_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    evidence_path = evidence_path.resolve(strict=True)
    config = load_config(config_path)
    evidence = load_config(evidence_path)

    if evidence.get("protocol") != "g14_nasa_license_evidence_v1":
        raise ValueError("Unexpected NASA license evidence protocol")
    settings = config["sources"]["nasa_suas"]
    license_settings = settings["license"]
    dataset = evidence["dataset"]
    policy = evidence["policy"]
    decision = evidence["decision"]

    if dataset["identifier"] != "6s8fb29q":
        raise ValueError("NASA license evidence points to an unexpected dataset")
    if dataset["dataset_specific_license"] != "not_specified":
        raise ValueError("Update the license gate implementation before changing its status")
    if policy["applicability_to_this_dataset"] != "not_explicitly_established":
        raise ValueError("Policy applicability must be backed by dataset-specific evidence")
    if decision.get("training_allowed") is not False:
        raise ValueError("Unresolved NASA license evidence cannot authorize training")
    if license_settings.get("training_allowed") is not False:
        raise ValueError("Intake config conflicts with the unresolved NASA license gate")

    output_dir = root / "artifacts/g14_domain_generalization/intake"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": dataset["name"],
            "identifier": dataset["identifier"],
            "access_level": dataset["access_level"],
            "dataset_specific_license": dataset["dataset_specific_license"],
            "catalog_url": dataset["catalog_url"],
            "nasa_portal_url": dataset["nasa_portal_url"],
        },
        "policy": {
            "url": policy["url"],
            "relevant_rule": policy["relevant_rule"],
            "applicability_to_this_dataset": policy[
                "applicability_to_this_dataset"
            ],
        },
        "contact": evidence["contact"],
        "decision": {
            "status": decision["status"],
            "training_allowed": False,
            "integrity_audit_allowed": bool(
                decision["extraction_for_integrity_audit_allowed"]
            ),
            "acceptable_resolution": list(decision["acceptable_resolution"]),
            "prohibited_until_resolution": list(
                decision["prohibited_until_resolution"]
            ),
        },
        "inputs": {
            "config_path": config_path.relative_to(root).as_posix(),
            "config_sha256": file_sha256(config_path),
            "evidence_path": evidence_path.relative_to(root).as_posix(),
            "evidence_sha256": file_sha256(evidence_path),
        },
        "ready_for_training": False,
        "training_started": False,
        "model_inference_run": False,
    }
    output_path = output_dir / "nasa_license_audit.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "dataset_specific_license": dataset["dataset_specific_license"],
                "policy_applicability": policy["applicability_to_this_dataset"],
                "decision": decision["status"],
                "ready_for_training": False,
                "contact": evidence["contact"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the G14 NASA license gate.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g14_domain_generalization_intake.yaml"),
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("configs/g14_nasa_license_evidence.yaml"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    audit(args.config, args.evidence, args.root)


if __name__ == "__main__":
    main()
