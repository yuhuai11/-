from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data_firewall import file_sha256
from .g7_r5_experiment import external_report, load_experiment, predict_system
from .train import resolve_device


PROTOCOL = "g7_r5_locked_external_test_execution_v1"


def evaluate(config_path: Path, *, device_name: str, batch_size: int) -> dict:
    config = load_experiment(config_path)
    test_manifest = Path(config["data"]["locked_test_manifest"])
    access_lock_path = Path(config["data"]["access_lock"])
    access_lock = json.loads(access_lock_path.read_text(encoding="utf-8"))
    if access_lock.get("status") != "authorized_frozen_not_consumed" or access_lock.get("consumed") is not False:
        raise ValueError("Locked test is not authorized for exactly one frozen execution")
    contract_path = Path(access_lock["freeze_contract_path"])
    if file_sha256(contract_path) != access_lock["freeze_contract_sha256"]:
        raise ValueError("Pretest freeze contract identity changed")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("test_inference_run") is not False or contract.get("locked_test_consumed") is not False:
        raise ValueError("Freeze contract is not in pretest state")
    if file_sha256(test_manifest) != contract["inputs"]["locked_test_manifest"]["sha256"]:
        raise ValueError("Locked test manifest changed after freeze")

    device = resolve_device(device_name)
    systems = {}
    for name, frozen in contract["systems"].items():
        checkpoint_paths = [Path(member["path"]) for member in frozen["members"]]
        observed_hashes = [file_sha256(path) for path in checkpoint_paths]
        expected_hashes = [member["sha256"] for member in frozen["members"]]
        if observed_hashes != expected_hashes:
            raise ValueError(f"Frozen checkpoint identity changed: {name}")
        rows, probabilities, _ = predict_system(
            checkpoint_paths,
            test_manifest,
            "locked_external_test",
            device=device,
            batch_size=batch_size,
        )
        systems[name] = external_report(rows, probabilities, frozen["thresholds"])

    baseline = systems["g7_r2_baseline"]
    candidate = systems["g7_r4c_ensemble"]
    comparison = {
        "segment_ranking_delta_candidate_minus_baseline": {
            metric: float(candidate["ranking"][metric] - baseline["ranking"][metric])
            for metric in baseline["ranking"]
        },
        "segment_operating_point_delta_candidate_minus_baseline": {
            point: {
                metric: float(
                    candidate["operating_points"][point][metric]
                    - baseline["operating_points"][point][metric]
                )
                for metric in ("accuracy", "precision", "recall", "f1", "false_positive_rate")
            }
            for point in baseline["operating_points"]
        },
    }
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "passed": True,
        "protocol": PROTOCOL,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": contract["purpose"],
        "systems": systems,
        "comparison": comparison,
        "freeze_contract": {
            "path": str(contract_path),
            "sha256": file_sha256(contract_path),
        },
        "locked_test_manifest": {
            "path": str(test_manifest),
            "sha256": file_sha256(test_manifest),
        },
        "test_used_for_training_or_threshold_selection": False,
        "repeat_test_allowed": False,
    }
    result_path = output_dir / "locked_test_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result_sha256 = file_sha256(result_path)

    access_lock.update(
        {
            "status": "consumed_read_once_complete",
            "consumed": True,
            "consumed_at_utc": result["executed_at_utc"],
            "result_path": str(result_path),
            "result_sha256": result_sha256,
            "access_log": [
                *list(access_lock.get("access_log", [])),
                {
                    "action": "single_frozen_test_execution",
                    "timestamp_utc": result["executed_at_utc"],
                    "result_sha256": result_sha256,
                },
            ],
        }
    )
    access_lock_path.write_text(json.dumps(access_lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute the frozen G7-R5 locked test once")
    parser.add_argument("--config", type=Path, default=Path("configs/g7_r5_locked_test_experiment.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.config, device_name=args.device, batch_size=args.batch_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
