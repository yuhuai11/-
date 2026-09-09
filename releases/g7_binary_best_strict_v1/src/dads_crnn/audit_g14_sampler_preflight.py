from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .data_firewall import file_sha256, reject_locked_path
from .dataset import DADSDataset
from .sampling import ClassSourceBalancedBatchSampler


PROTOCOL = "g14_source_balanced_sampler_preflight_v1"


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    reject_locked_path(path, context="G14 sampler preflight")
    return path.resolve(strict=True)


def _planned_ratios(
    sampler: ClassSourceBalancedBatchSampler, rows: pd.DataFrame
) -> list[dict[str, Any]]:
    output = []
    counts = rows.groupby(["label", sampler.source_column]).size()
    for label in (0, 1):
        for source, draws in sampler.source_draws[label].items():
            available = int(counts.loc[(label, source)])
            output.append(
                {
                    "label": label,
                    "source_group": source,
                    "available_segments": available,
                    "planned_draws": draws,
                    "draw_to_available_ratio": draws / available,
                }
            )
    return output


def audit(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected sampler-preflight protocol")
    manifest_path = _resolve(root, config["segment_manifest"])
    cache_audit_path = _resolve(root, config["segment_cache_audit"])
    cache_audit = json.loads(cache_audit_path.read_text(encoding="utf-8"))
    if not (
        cache_audit.get("passed") is True
        and cache_audit.get("ready_for_sampler_preflight") is True
        and cache_audit.get("outputs", {}).get("segment_manifest", {}).get("sha256")
        == file_sha256(manifest_path)
    ):
        raise ValueError("Segment manifest no longer matches its cache audit")

    rows = pd.read_csv(manifest_path, low_memory=False)
    train = rows.loc[rows["split"] == "train"].reset_index(drop=True)
    sampler_config = config["sampler"]
    batch_size = int(sampler_config["batch_size"])
    num_batches = int(sampler_config["batches_per_epoch"])
    samples_per_epoch = int(sampler_config["samples_per_epoch"])
    if batch_size * num_batches != samples_per_epoch:
        raise ValueError("Sampler epoch-size contract is inconsistent")
    seeds = [int(seed) for seed in sampler_config["seeds"]]
    sampler = ClassSourceBalancedBatchSampler(
        train,
        source_column=str(sampler_config["source_column"]),
        batch_size=batch_size,
        num_batches=num_batches,
        source_weight_exponent=float(sampler_config["source_weight_exponent"]),
        seed=seeds[0],
    )
    planned = _planned_ratios(sampler, train)
    maximum_ratio = max(row["draw_to_available_ratio"] for row in planned)
    minimum_ratio = min(row["draw_to_available_ratio"] for row in planned)
    guardrails = config["guardrails"]
    if maximum_ratio > float(guardrails["maximum_source_repeat_ratio"]):
        raise ValueError(f"Source repeat guardrail failed: {maximum_ratio}")
    if minimum_ratio < float(guardrails["minimum_source_coverage_ratio"]):
        raise ValueError(f"Source coverage guardrail failed: {minimum_ratio}")

    observed_sources: Counter[tuple[int, str]] = Counter()
    class_counts: Counter[int] = Counter()
    first_batch = None
    for batch_number, batch in enumerate(sampler, start=1):
        if len(batch) != batch_size:
            raise ValueError(f"Sampler produced a short batch: {batch_number}")
        labels = train.iloc[batch]["label"].to_numpy(dtype=np.int64)
        counts = Counter(int(value) for value in labels)
        expected_class = int(sampler_config["class_samples_per_batch"])
        if counts != Counter({0: expected_class, 1: expected_class}):
            raise ValueError(f"Batch class balance failed at batch {batch_number}")
        class_counts.update(int(value) for value in labels)
        selected = train.iloc[batch]
        observed_sources.update(
            (int(row.label), str(row.source_group))
            for row in selected[["label", "source_group"]].itertuples(index=False)
        )
        if first_batch is None:
            first_batch = list(batch)
    if len(observed_sources) != sum(sampler.source_counts.values()):
        raise ValueError("Sampler omitted one or more source groups")
    for row in planned:
        key = (int(row["label"]), str(row["source_group"]))
        if observed_sources[key] != int(row["planned_draws"]):
            raise ValueError(f"Observed source draws differ from plan: {key}")

    repeat = ClassSourceBalancedBatchSampler(
        train,
        source_column=str(sampler_config["source_column"]),
        batch_size=batch_size,
        num_batches=num_batches,
        source_weight_exponent=float(sampler_config["source_weight_exponent"]),
        seed=seeds[0],
    )
    if first_batch != next(iter(repeat)):
        raise ValueError("Sampler is not deterministic for the same seed")
    different = ClassSourceBalancedBatchSampler(
        train,
        source_column=str(sampler_config["source_column"]),
        batch_size=batch_size,
        num_batches=num_batches,
        source_weight_exponent=float(sampler_config["source_weight_exponent"]),
        seed=seeds[1],
    )
    if first_batch == next(iter(different)):
        raise ValueError("Different sampler seeds produced the same first batch")

    dataset = DADSDataset(
        manifest_path,
        "train",
        sample_rate=16000,
        clip_seconds=1.0,
        training=False,
        seed=seeds[0],
    )
    waveforms = []
    targets = []
    for index in first_batch:
        waveform, target = dataset[index]
        waveforms.append(waveform.numpy())
        targets.append(float(target))
    batch_array = np.stack(waveforms)
    if batch_array.shape != (batch_size, 16000) or not np.isfinite(batch_array).all():
        raise ValueError("Real memmap batch is invalid")
    if Counter(int(value) for value in targets) != Counter({0: 64, 1: 64}):
        raise ValueError("Real memmap batch labels are not balanced")

    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "source_draw_plan.csv"
    with plan_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(planned[0]))
        writer.writeheader()
        writer.writerows(planned)

    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_segments_available": len(train),
        "samples_per_epoch": samples_per_epoch,
        "batches_per_epoch": num_batches,
        "batch_size": batch_size,
        "class_samples_per_batch": int(sampler_config["class_samples_per_batch"]),
        "epoch_class_draws": dict(sorted(class_counts.items())),
        "source_groups": sampler.source_counts,
        "source_weight_exponent": sampler.source_weight_exponent,
        "minimum_source_draw_to_available_ratio": minimum_ratio,
        "maximum_source_draw_to_available_ratio": maximum_ratio,
        "same_seed_deterministic": True,
        "different_seed_changes_order": True,
        "real_memmap_batch_passed": True,
        "automatic_pos_weight": 1.0,
        "tune_segments_natural": int((rows["split"] == "tune").sum()),
        "dev_holdout_segments_natural": int(
            (rows["split"] == "dev_holdout").sum()
        ),
        "ready_for_g14_model_preflight": True,
        "ready_for_training": False,
        "training_blockers": ["g14_head_only_model_preflight_not_completed"],
        "model_inference_run": False,
        "training_started": False,
        "inputs": {
            "config_sha256": file_sha256(config_path),
            "segment_manifest_sha256": file_sha256(manifest_path),
            "segment_cache_audit_sha256": file_sha256(cache_audit_path),
        },
        "outputs": {
            "source_draw_plan": {
                "path": plan_path.relative_to(root).as_posix(),
                "sha256": file_sha256(plan_path),
                "rows": len(planned),
            }
        },
    }
    report_path = output_dir / "sampler_preflight_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit exact class-balanced, tempered source sampling for G14."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g14_sampler_preflight.yaml")
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    audit(args.config, args.root)


if __name__ == "__main__":
    main()
