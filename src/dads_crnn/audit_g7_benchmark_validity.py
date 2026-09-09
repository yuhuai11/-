from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from .data_firewall import file_sha256
from .evaluate_g7_r6_external_suite import (
    _dads_dataset,
    _esc50_dataset,
    _g13_dataset,
    _idmt_dataset,
)
from .g7_benchmark_validity import build_validity_report


PROTOCOL = "g7_reusable_benchmark_model_input_identity_audit_v1"


def model_input_hashes(
    dataset: Dataset, name: str, batch_size: int
) -> tuple[set[str], dict[str, Any]]:
    identities: set[str] = set()
    duplicates = 0
    rows = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch_index, waveforms in enumerate(loader, start=1):
        values = waveforms.numpy().astype("<f4", copy=False)
        for waveform in values:
            identity = hashlib.sha256(waveform.tobytes(order="C")).hexdigest()
            duplicates += int(identity in identities)
            identities.add(identity)
            rows += 1
        if batch_index % 200 == 0 or batch_index == len(loader):
            print(f"{name}: {rows}/{len(dataset)}", flush=True)
    return identities, {
        "rows": rows,
        "unique_model_inputs": len(identities),
        "within_dataset_duplicate_rows": duplicates,
    }


def audit(output: Path, batch_size: int) -> dict[str, Any]:
    paths = {
        "development_fit": Path("artifacts/g7_r6_reusable_multicorpus/fit_manifest.csv"),
        "calibration": Path("artifacts/g7_r6_reusable_multicorpus/threshold_calibration_manifest.csv"),
        "kielce_tau": Path("artifacts/g7_r5_train_val_test/locked_unseen_external_test/manifest.csv"),
        "g13": Path("artifacts/g13_external_confirmation/intake/external_confirmation_v2_manifest.csv"),
        "idmt": Path("artifacts/g7_improvement/stage_b/development_segments_dedup.csv"),
        "esc50": Path("artifacts/g9_hard_negatives/manifests/hn_guard.csv"),
    }
    frames = {name: pd.read_csv(path, low_memory=False) for name, path in paths.items()}
    metadata = build_validity_report(
        frames["development_fit"],
        frames["calibration"],
        frames["kielce_tau"],
        frames["g13"],
        frames["idmt"],
        frames["esc50"],
    )
    datasets = {
        "train": _dads_dataset(paths["development_fit"], "train"),
        "model_validation": _dads_dataset(
            paths["development_fit"], "model_validation"
        ),
        "calibration": _dads_dataset(paths["calibration"], "threshold_calibration"),
        "kielce_tau": _dads_dataset(paths["kielce_tau"], "locked_external_test"),
        "g13": _g13_dataset(paths["g13"]),
        "idmt": _idmt_dataset(paths["idmt"]),
        "esc50": _esc50_dataset(paths["esc50"]),
    }
    identities: dict[str, set[str]] = {}
    summaries = {}
    for name, dataset in datasets.items():
        identities[name], summaries[name] = model_input_hashes(
            dataset, name, batch_size
        )

    development = identities["train"] | identities["model_validation"]
    development_overlap = {
        name: len(values & development)
        for name, values in identities.items()
        if name not in {"train", "model_validation"}
    }
    train_validation_overlap = len(
        identities["train"] & identities["model_validation"]
    )
    passed = bool(
        train_validation_overlap == 0
        and all(value == 0 for value in development_overlap.values())
    )
    report = {
        "passed": passed,
        "protocol": PROTOCOL,
        "metadata_validity": metadata,
        "model_input_identity": {
            "hash_definition": "sha256_little_endian_float32_native_halfsecond_after_evaluation_preprocessing",
            "datasets": summaries,
            "train_vs_model_validation_overlap": train_validation_overlap,
            "development_vs_non_development_overlap": development_overlap,
            "near_duplicate_fingerprint_audit_completed": False,
            "near_duplicate_limitation": (
                "Exact identities do not exclude gain-shifted, cropped, re-encoded, or acoustically near-duplicate recordings."
            ),
        },
        "inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit corrected G7 reusable benchmark")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/g7_r6_dronenoise_control/external_suite/validity_audit.json"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    report = audit(args.output, args.batch_size)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "overall_risk": report["metadata_validity"]["overall_risk"],
                "model_input_identity": report["model_input_identity"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
