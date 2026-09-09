from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OOD calibration for frozen full models")
    parser.add_argument("--config", default="configs/val_ood.yaml")
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    calibration = config["calibration"]
    experiments = args.experiments or list(calibration["experiments"])
    seeds = args.seeds or [int(value) for value in calibration["seeds"]]
    root = Path(config["output_dir"])
    tune_manifest = root / "manifests" / "val_ood_tune_manifest.csv"
    holdout_manifest = root / "manifests" / "val_ood_holdout_manifest.csv"
    commands = []
    for experiment in experiments:
        run_dir = Path(calibration["experiments"][experiment]["run_dir"])
        for seed in seeds:
            checkpoint = run_dir / f"seed_{seed}" / "best.pt"
            command = [
                sys.executable,
                "-m",
                "dads_crnn.calibrate_ood",
                "--checkpoint",
                str(checkpoint),
                "--tune-manifest",
                str(tune_manifest),
                "--holdout-manifest",
                str(holdout_manifest),
                "--output-dir",
                str(root),
                "--experiment",
                experiment,
                "--batch-size",
                str(calibration["batch_size"]),
                "--num-workers",
                str(calibration["num_workers"]),
                "--device",
                str(calibration["device"]),
                "--target-recall",
                str(calibration["target_recall"]),
                "--target-specificity",
                str(calibration["target_specificity"]),
                "--ece-bins",
                str(calibration["ece_bins"]),
            ]
            if not args.dry_run:
                for required in (checkpoint, tune_manifest, holdout_manifest):
                    if not required.exists():
                        raise FileNotFoundError(required)
            commands.append(command)
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
