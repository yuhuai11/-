from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data_firewall import file_sha256
from .dataset import DADSDataset


PROTOCOL = "g7_r8_urban_negative_single_variable_preflight_v1"
BASELINE_CONFIG = Path("configs/g7_r7_freq_mixstyle.yaml")
CANDIDATE_CONFIG = Path("configs/g7_r8_urban_negatives.yaml")
MANIFEST = Path("artifacts/g7_r8_urban_negatives/data/manifest.csv")
AUDIT = Path("artifacts/g7_r8_urban_negatives/data/audit.json")


def audit() -> dict:
    baseline = load_config(BASELINE_CONFIG)
    candidate = load_config(CANDIDATE_CONFIG)
    manifest_audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    rows = pd.read_csv(MANIFEST, low_memory=False)
    if manifest_audit.get("passed") is not True:
        raise ValueError("R8 manifest audit has not passed")
    if file_sha256(MANIFEST) != manifest_audit["output"]["manifest"]["sha256"]:
        raise ValueError("R8 manifest no longer matches its audit")

    expected = json.loads(json.dumps(baseline))
    expected["protocol"] = candidate["protocol"]
    expected["data"] = candidate["data"]
    expected["train"]["pos_weight"] = candidate["train"]["pos_weight"]
    expected["output_dir"] = candidate["output_dir"]
    if candidate != expected:
        raise ValueError("R8 config changes fields beyond audited data replay/control fields")
    if "sampling" in candidate["train"]:
        raise ValueError("R8 seed42 screen must retain natural shuffle without replacement")

    train = rows[rows["split"].eq("train")].reset_index(drop=True)
    tau = train[train["dataset_origin"].eq("tau_urban_2022")]
    if len(tau) != 43320 or tau["recording_group"].nunique() != 21660:
        raise ValueError("Unexpected TAU training replay population")
    if not bool(tau["label"].eq(0).all()) or bool(tau["background_mix_eligible"].any()):
        raise ValueError("TAU replay role is not pure direct-negative")

    dataset = DADSDataset(
        MANIFEST, "train", sample_rate=16000, clip_seconds=0.5,
        training=False, seed=42,
    )
    tau_index = int(dataset.rows.index[dataset.rows["dataset_origin"].eq("tau_urban_2022")][0])
    dads_index = int(dataset.rows.index[dataset.rows["dataset_origin"].eq("dads_halfsec")][0])
    tau_waveform, tau_label = dataset[tau_index]
    dads_waveform, dads_label = dataset[dads_index]
    finite_load = bool(
        tau_waveform.shape == dads_waveform.shape == (8000,)
        and np.isfinite(tau_waveform.numpy()).all()
        and np.isfinite(dads_waveform.numpy()).all()
        and float(tau_label) == 0.0
    )

    result = {
        "passed": finite_load,
        "protocol": PROTOCOL,
        "experimental_variable": "direct_replay_of_tau_training_urban_negatives",
        "controls": {
            "model_and_frequency_mixstyle_identical_to_r7": True,
            "optimizer_schedule_and_augmentation_identical_to_r7": True,
            "r7_effective_pos_weight_preserved": float(candidate["train"]["pos_weight"]),
            "natural_shuffle_without_replacement": True,
            "validation_and_test_identical_to_r7": True,
            "tau_never_used_as_positive_background_mixer": True,
        },
        "real_loader_check": {
            "passed": finite_load,
            "waveform_shape": list(tau_waveform.shape),
            "tau_label": float(tau_label),
            "dads_label_example": float(dads_label),
        },
        "counts": manifest_audit["counts"],
        "firewall": manifest_audit["firewall"],
        "inputs": {
            "baseline_config": {"path": str(BASELINE_CONFIG), "sha256": file_sha256(BASELINE_CONFIG)},
            "candidate_config": {"path": str(CANDIDATE_CONFIG), "sha256": file_sha256(CANDIDATE_CONFIG)},
            "manifest": {"path": str(MANIFEST), "sha256": file_sha256(MANIFEST)},
            "manifest_audit": {"path": str(AUDIT), "sha256": file_sha256(AUDIT)},
        },
        "locked_datasets_read": [],
        "formal_training_started": False,
    }
    if not result["passed"]:
        raise RuntimeError("R8 real-loader preflight failed")
    output = Path("artifacts/g7_r8_urban_negatives/preflight/audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    print(json.dumps(audit(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
