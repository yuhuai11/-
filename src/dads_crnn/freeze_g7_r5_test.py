from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data_firewall import file_sha256
from .g7_r5_experiment import (
    load_experiment,
    predict_system,
    validation_freeze_report,
)
from .train import resolve_device


PROTOCOL = "g7_r5_pretest_freeze_contract_v1"


def freeze(config_path: Path, *, device_name: str, batch_size: int) -> dict:
    config = load_experiment(config_path)
    validation_manifest = Path(config["data"]["validation_manifest"])
    test_manifest = Path(config["data"]["locked_test_manifest"])
    access_lock_path = Path(config["data"]["access_lock"])
    access_lock = json.loads(access_lock_path.read_text(encoding="utf-8"))
    if access_lock.get("consumed") is not False or access_lock.get("status") != "locked_unconsumed":
        raise ValueError("Locked external test has already been consumed or unlocked")
    if file_sha256(test_manifest) != access_lock["manifest_sha256"]:
        raise ValueError("Locked test manifest identity changed")
    # Metadata-only validation of the test role; no cache/audio payload is opened.
    test_roles = set(pd.read_csv(test_manifest, usecols=["dataset_role"])["dataset_role"])
    if test_roles != {"locked_external_test"}:
        raise ValueError("Unexpected role in locked test manifest")

    device = resolve_device(device_name)
    systems = {}
    thresholds_by_system = {}
    for name, spec in config["systems"].items():
        checkpoint_paths = [Path(value) for value in spec["checkpoints"]]
        rows, probabilities, identities = predict_system(
            checkpoint_paths,
            validation_manifest,
            "validation",
            device=device,
            batch_size=batch_size,
        )
        report, thresholds = validation_freeze_report(
            rows, probabilities, [float(value) for value in config["thresholds"]["target_fprs"]]
        )
        systems[name] = {
            "aggregation": spec["aggregation"],
            "members": identities,
            "validation": report,
            "thresholds": thresholds,
        }
        thresholds_by_system[name] = thresholds

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "passed": True,
        "protocol": PROTOCOL,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": config["purpose"],
        "systems": systems,
        "recording_aggregation": config["recording_aggregation"],
        "metrics": config["metrics"],
        "post_test_policy": config["post_test_policy"],
        "inputs": {
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "validation_manifest": {
                "path": str(validation_manifest),
                "sha256": file_sha256(validation_manifest),
            },
            "locked_test_manifest": {
                "path": str(test_manifest),
                "sha256": file_sha256(test_manifest),
                "audio_payload_read": False,
            },
        },
        "test_inference_run": False,
        "locked_test_consumed": False,
    }
    contract_path = output_dir / "pretest_freeze_contract.json"
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    contract_sha256 = file_sha256(contract_path)
    access_lock.update(
        {
            "status": "authorized_frozen_not_consumed",
            "authorized_candidate": {
                name: [member["sha256"] for member in value["members"]]
                for name, value in systems.items()
            },
            "authorized_thresholds": thresholds_by_system,
            "authorized_aggregation": config["recording_aggregation"],
            "freeze_contract_path": str(contract_path),
            "freeze_contract_sha256": contract_sha256,
        }
    )
    access_lock_path.write_text(json.dumps(access_lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    contract["freeze_contract_sha256"] = contract_sha256
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze G7-R5 systems and validation thresholds")
    parser.add_argument("--config", type=Path, default=Path("configs/g7_r5_locked_test_experiment.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(freeze(args.config, device_name=args.device, batch_size=args.batch_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
