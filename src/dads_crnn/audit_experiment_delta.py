from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ensure_dirs, load_config


ALLOWED_CHANGES = {
    "output_dir",
    "train.augmentation.mix_snr_db",
    "train.augmentation.mix_snr_weights",
}
EXPECTED_BASELINE_LEVELS = [-5, 0, 5, 10]
EXPECTED_BASELINE_WEIGHTS = [0.10, 0.25, 0.35, 0.30]
EXPECTED_CANDIDATE_LEVELS = [-10, -5, 0, 5, 10]
EXPECTED_CANDIDATE_WEIGHTS = [0.05, 0.10, 0.25, 0.35, 0.25]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(child, child_prefix))
        return result
    return {prefix: value}


def config_differences(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    left = _flatten(baseline)
    right = _flatten(candidate)
    differences = {}
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            differences[key] = {"baseline": left.get(key), "candidate": right.get(key)}
    return differences


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit that G4 differs from G2 only as approved")
    parser.add_argument("--baseline", default="configs/crnn_dads_full_augmented_g2.yaml")
    parser.add_argument("--candidate", default="configs/crnn_dads_full_augmented_g4_low_snr.yaml")
    parser.add_argument("--guardrails", default="configs/g4_low_snr_guardrails.yaml")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts_full/manifests/dads_all_seed42.csv"))
    parser.add_argument("--mode", choices=("preflight", "verify"), default="preflight")
    parser.add_argument("--inventory", type=Path, default=Path("artifacts/g4_low_snr/baseline_inventory.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/g4_low_snr/delta_audit.json"))
    args = parser.parse_args()

    baseline = load_config(args.baseline)
    candidate = load_config(args.candidate)
    guardrails = load_config(args.guardrails)
    differences = config_differences(baseline, candidate)
    baseline_aug = baseline["train"]["augmentation"]
    candidate_aug = candidate["train"]["augmentation"]
    protected_paths = {
        "baseline_config": Path(guardrails["baseline"]["config"]),
        "baseline_checkpoint": Path(guardrails["baseline"]["checkpoint"]),
        "baseline_metrics": Path(guardrails["baseline"]["metrics"]),
        "baseline_calibration": Path(guardrails["baseline"]["calibration"]),
    }
    missing_protected = [name for name, path in protected_paths.items() if not path.is_file()]
    current_inventory = (
        {name: {"path": path.as_posix(), "sha256": sha256_file(path)} for name, path in protected_paths.items()}
        if not missing_protected
        else {}
    )
    inventory_match = True
    if args.inventory.exists():
        saved_inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
        inventory_match = saved_inventory == current_inventory
    elif args.mode == "verify":
        inventory_match = False

    train_rows = pd.read_csv(args.manifest, usecols=["split", "label"])
    train_rows = train_rows[train_rows["split"] == "train"]
    positive_rate = float(train_rows["label"].mean())
    moved_probability = 0.05
    changed_train_fraction = (
        positive_rate * float(candidate_aug["positive_mix_probability"]) * moved_probability
    )
    output_dir = Path(candidate["output_dir"])
    output_is_empty = not output_dir.exists() or not any(output_dir.rglob("*"))

    checks = {
        "only_approved_config_changes": set(differences) == ALLOWED_CHANGES,
        "baseline_snr_exact": baseline_aug["mix_snr_db"] == EXPECTED_BASELINE_LEVELS
        and baseline_aug["mix_snr_weights"] == EXPECTED_BASELINE_WEIGHTS,
        "candidate_snr_exact": candidate_aug["mix_snr_db"] == EXPECTED_CANDIDATE_LEVELS
        and candidate_aug["mix_snr_weights"] == EXPECTED_CANDIDATE_WEIGHTS,
        "natural_sampling_preserved": "sampling" not in candidate["train"],
        "single_worker_reproducibility": int(candidate["train"]["num_workers"]) == 0,
        "seed42_only": candidate["train"]["seeds"] == [42],
        "independent_output_directory": candidate["output_dir"] != baseline["output_dir"],
        "candidate_output_empty_before_training": args.mode == "verify" or output_is_empty,
        "protected_artifacts_exist": not missing_protected,
        "protected_artifacts_unchanged": inventory_match,
        "changed_fraction_below_2_1_percent": changed_train_fraction <= 0.021,
    }
    passed = all(checks.values())
    report = {
        "passed": passed,
        "mode": args.mode,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "baseline_config_sha256": sha256_file(Path(args.baseline)),
        "candidate_config_sha256": sha256_file(Path(args.candidate)),
        "guardrails_sha256": sha256_file(Path(args.guardrails)),
        "differences": differences,
        "allowed_changes": sorted(ALLOWED_CHANGES),
        "train_positive_rate": positive_rate,
        "expected_changed_train_fraction": changed_train_fraction,
        "protected_inventory": current_inventory,
        "checks": checks,
    }
    ensure_dirs(args.output.parent, args.inventory.parent)
    if args.mode == "preflight" and not args.inventory.exists() and not missing_protected:
        args.inventory.write_text(json.dumps(current_inventory, indent=2), encoding="utf-8")
        checks["protected_artifacts_unchanged"] = True
        report["passed"] = all(checks.values())
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("G4 experiment delta audit failed closed")


if __name__ == "__main__":
    main()
