from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen external-evaluation matrix")
    parser.add_argument("--config", default="configs/external_evaluation.yaml")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    datasets = args.datasets or list(config["datasets"])
    experiments = args.experiments or list(config["experiments"])
    eval_cfg = config["evaluation"]
    output_root = Path(config["output_dir"])

    commands = []
    for dataset in datasets:
        manifest = output_root / "manifests" / f"{dataset}_manifest.csv"
        if not args.dry_run and not manifest.exists():
            raise FileNotFoundError(f"Build the external manifest first: {manifest}")
        for experiment in experiments:
            experiment_cfg = config["experiments"][experiment]
            for seed in args.seeds:
                checkpoint = Path(experiment_cfg["run_dir"]) / f"seed_{seed}" / "best.pt"
                if not args.dry_run and not checkpoint.exists():
                    raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
                command = [
                    sys.executable,
                    "-m",
                    "dads_crnn.evaluate_external",
                    "--checkpoint",
                    str(checkpoint),
                    "--manifest",
                    str(manifest),
                    "--dataset-name",
                    dataset,
                    "--experiment-name",
                    experiment,
                    "--training-scale",
                    str(experiment_cfg["training_scale"]),
                    "--output-dir",
                    str(output_root / "predictions"),
                    "--primary-threshold",
                    str(eval_cfg["primary_threshold"]),
                    "--batch-size",
                    str(eval_cfg["batch_size"]),
                    "--num-workers",
                    str(eval_cfg["num_workers"]),
                    "--bootstrap-samples",
                    str(eval_cfg["bootstrap_samples"]),
                    "--bootstrap-seed",
                    str(eval_cfg["bootstrap_seed"]),
                    "--device",
                    str(eval_cfg["device"]),
                    "--thresholds",
                    *[str(value) for value in eval_cfg["thresholds"]],
                ]
                commands.append(command)

    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
