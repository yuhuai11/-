from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config
from .data_firewall import file_sha256


def audit(candidate_path: Path, output_path: Path) -> dict:
    candidate = load_config(candidate_path)
    reference = candidate.get("control_reference", {})
    base_path = Path(str(reference["config_path"]))
    if file_sha256(base_path) != str(reference["config_sha256"]):
        raise ValueError("G7-R2 control config hash changed")
    checkpoint_path = Path(str(reference["seed42_checkpoint"]))
    if file_sha256(checkpoint_path) != str(reference["seed42_checkpoint_sha256"]):
        raise ValueError("G7-R2 seed42 control checkpoint hash changed")
    base = load_config(base_path)
    base_comparable = copy.deepcopy(base)
    candidate_comparable = copy.deepcopy(candidate)
    for values in (base_comparable, candidate_comparable):
        values.pop("protocol", None)
        values.pop("control_reference", None)
        values.pop("output_dir", None)
        values.pop("data", None)
    if candidate_comparable != base_comparable:
        raise ValueError("Model, features, training or evaluation changed from G7-R2 control")

    protocol_path = Path("artifacts/g7_r6_reusable_multicorpus/protocol_audit.json")
    cache_path = Path("artifacts/g7_r6_dronenoise_halfsec_cache/cache_audit.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if not protocol.get("passed") or protocol.get("protocol") != "g7_r6_reuter_reusable_multicorpus_v3":
        raise ValueError("G7-R6 v3 protocol audit did not pass")
    if not cache.get("passed"):
        raise ValueError("DroneNoise cache audit did not pass")
    fit = protocol["outputs"]["fit"]
    fit_path = Path(str(fit["path"]))
    if file_sha256(fit_path) != str(fit["sha256"]):
        raise ValueError("G7-R6 fit manifest changed after protocol audit")

    report = {
        "passed": True,
        "protocol": "g7_r6_dronenoise_control_preflight_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "single_changed_factor": "training_and_model_validation_data",
        "model_features_train_eval_identical_to_g7_r2_pt_mic_bg_freq": True,
        "dronenoise_test_used_for_training_or_early_stopping": False,
        "fit_roles": fit["roles"],
        "inputs": {
            "candidate_config": {"path": str(candidate_path), "sha256": file_sha256(candidate_path)},
            "control_config": {"path": str(base_path), "sha256": file_sha256(base_path)},
            "control_checkpoint": {"path": str(checkpoint_path), "sha256": file_sha256(checkpoint_path)},
            "protocol_audit": {"path": str(protocol_path), "sha256": file_sha256(protocol_path)},
            "cache_audit": {"path": str(cache_path), "sha256": file_sha256(cache_path)},
            "fit_manifest": {"path": str(fit_path), "sha256": file_sha256(fit_path)},
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit G7-R6 single-variable training control")
    parser.add_argument("--config", type=Path, default=Path("configs/g7_r6_dronenoise_control.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/g7_r6_dronenoise_control/preflight/config_delta_audit.json"),
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.config, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
