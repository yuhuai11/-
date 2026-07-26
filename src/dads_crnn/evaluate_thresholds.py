from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .metrics import binary_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved probabilities at configured thresholds.")
    parser.add_argument("--config", default="configs/crnn_dads.yaml")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = Path(args.run_dir)
    y_prob = np.load(run_dir / "test_probabilities.npy")
    y_true = np.load(run_dir / "test_labels.npy").astype(np.int64)
    metrics = [binary_metrics(y_true, y_prob, threshold=t) for t in config["eval"]["thresholds"]]
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
