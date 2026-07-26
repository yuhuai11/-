from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def collect_runs(run_root: Path, expected_seeds: list[int]) -> list[dict[str, Any]]:
    results = []
    reference_inputs = None
    for seed in expected_seeds:
        metrics_path = run_root / f"seed_{seed}" / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing completed seed metrics: {metrics_path}")
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        if int(result.get("seed", -1)) != seed:
            raise ValueError(f"Seed identity mismatch in {metrics_path}")
        if result.get("locked_datasets_read") != []:
            raise ValueError(f"Locked final-test access reported by {metrics_path}")
        training_inputs = result.get("training_inputs")
        if not isinstance(training_inputs, dict):
            raise ValueError(f"Missing training input identity in {metrics_path}")
        if reference_inputs is None:
            reference_inputs = training_inputs
        elif training_inputs != reference_inputs:
            raise ValueError("Multi-seed runs do not share an identical training input identity")
        results.append(result)
    return results


def write_summary(run_root: Path, results: list[dict[str, Any]]) -> Path:
    summary_path = run_root / "summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    temporary.replace(summary_path)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild a verified PANNs multi-seed summary")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    seeds = sorted(set(args.expected_seeds))
    if len(seeds) != len(args.expected_seeds):
        parser.error("--expected-seeds must not contain duplicates")
    results = collect_runs(args.run_root, seeds)
    summary = write_summary(args.run_root, results)
    print(f"Verified seeds {seeds}; wrote {summary}")


if __name__ == "__main__":
    main()
