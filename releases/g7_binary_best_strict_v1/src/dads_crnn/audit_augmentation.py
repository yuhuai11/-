from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .audio import peak_normalize, to_fixed_length
from .augmentation import WaveformAugmenter
from .config import ensure_dirs, load_config
from .dataset import DADSDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit stochastic training augmentation without training")
    parser.add_argument("--config", default="configs/crnn_dads_full_augmented_g2.yaml")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output", type=Path, default=Path("artifacts/augmentation_audit/g2_audit.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    manifest_path = args.manifest or next(Path(data_cfg["manifest_dir"]).glob("dads_*.csv"))
    dataset = DADSDataset(
        manifest_path,
        "train",
        sample_rate=int(data_cfg["sample_rate"]),
        clip_seconds=float(data_cfg["clip_seconds"]),
        training=False,
        seed=args.seed,
    )
    augmenter = WaveformAugmenter(
        config["train"]["augmentation"], int(data_cfg["sample_rate"]), args.seed
    )
    rng = np.random.default_rng(args.seed)
    selected = []
    for label in (0, 1):
        indices = dataset.rows.index[dataset.rows["label"] == label].to_numpy(dtype=np.int64)
        selected.extend(rng.choice(indices, size=min(args.samples_per_class, len(indices)), replace=False))
    negative_indices = dataset.rows.index[dataset.rows["label"] == 0].to_numpy(dtype=np.int64)

    def load(index: int) -> np.ndarray:
        audio = dataset._load_audio(dataset.rows.iloc[index])
        return peak_normalize(to_fixed_length(audio, dataset.target_samples, random_crop=False))

    def sample_background() -> np.ndarray:
        output = np.zeros(dataset.target_samples, dtype=np.float32)
        for _ in range(8):
            output = load(int(rng.choice(negative_indices)))
            if float(np.sqrt(np.mean(np.square(output, dtype=np.float64)))) > 1e-8:
                break
        return output

    operations: Counter[str] = Counter()
    target_snr: Counter[str] = Counter()
    snr_errors = []
    maximum_peak = 0.0
    finite = True
    negative_mix_count = 0
    selected_label_counts: Counter[str] = Counter()
    for index in selected:
        row = dataset.rows.iloc[int(index)]
        audio = load(int(index))
        label = int(row["label"])
        selected_label_counts[str(label)] += 1

        augmented, metadata = augmenter.apply(audio, label, sample_background)
        for operation in ("background_mix", "frequency_response", "reverb", "colored_noise", "time_shift"):
            operations[operation] += int(bool(metadata[operation]))
        if metadata["target_snr_db"] is not None:
            target_snr[str(metadata["target_snr_db"])] += 1
            snr_errors.append(abs(metadata["achieved_snr_db"] - metadata["target_snr_db"]))
            negative_mix_count += int(label == 0)
        maximum_peak = max(maximum_peak, float(np.max(np.abs(augmented))))
        finite = finite and bool(np.isfinite(augmented).all())
    total = len(selected)
    positive_samples = selected_label_counts["1"]
    mix_count = operations["background_mix"]
    observed_mix_probability = mix_count / max(positive_samples, 1)
    expected_mix_probability = float(config["train"]["augmentation"]["positive_mix_probability"])
    mix_tolerance = max(
        0.04,
        4.0
        * np.sqrt(
            expected_mix_probability * (1.0 - expected_mix_probability) / max(positive_samples, 1)
        ),
    )
    configured_levels = [float(value) for value in config["train"]["augmentation"]["mix_snr_db"]]
    configured_weights = [float(value) for value in config["train"]["augmentation"]["mix_snr_weights"]]
    observed_snr_distribution = {
        str(level): target_snr[str(level)] / max(mix_count, 1) for level in configured_levels
    }
    snr_tolerances = {
        str(level): max(0.04, 4.0 * np.sqrt(weight * (1.0 - weight) / max(mix_count, 1)))
        for level, weight in zip(configured_levels, configured_weights)
    }
    probability_checks = {
        "positive_mix_probability": abs(observed_mix_probability - expected_mix_probability)
        <= mix_tolerance,
        "snr_distribution": all(
            abs(observed_snr_distribution[str(level)] - weight) <= snr_tolerances[str(level)]
            for level, weight in zip(configured_levels, configured_weights)
        ),
        "negative_never_background_mixed": negative_mix_count == 0,
    }
    train_check = DADSDataset(
        manifest_path,
        "train",
        sample_rate=int(data_cfg["sample_rate"]),
        clip_seconds=float(data_cfg["clip_seconds"]),
        training=True,
        seed=args.seed,
        augmentation=config["train"]["augmentation"],
    )
    val_check = DADSDataset(
        manifest_path,
        "val",
        sample_rate=int(data_cfg["sample_rate"]),
        clip_seconds=float(data_cfg["clip_seconds"]),
        training=False,
        seed=args.seed,
        augmentation=config["train"]["augmentation"],
    )
    test_check = DADSDataset(
        manifest_path,
        "test",
        sample_rate=int(data_cfg["sample_rate"]),
        clip_seconds=float(data_cfg["clip_seconds"]),
        training=False,
        seed=args.seed,
        augmentation=config["train"]["augmentation"],
    )
    split_checks = {
        "train_augmented": train_check.augmenter is not None,
        "validation_not_augmented": val_check.augmenter is None,
        "test_not_augmented": test_check.augmenter is None,
    }
    config_hash = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    report = {
        "passed": bool(
            finite
            and maximum_peak <= 1.000001
            and (max(snr_errors) if snr_errors else 0.0) < 1e-4
            and all(probability_checks.values())
            and all(split_checks.values())
        ),
        "config": args.config,
        "config_sha256": config_hash,
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": manifest_hash,
        "samples": total,
        "samples_per_class": min(args.samples_per_class, total // 2),
        "operation_counts": dict(operations),
        "operation_rates": {key: value / total for key, value in operations.items()},
        "selected_label_counts": dict(selected_label_counts),
        "expected_positive_mix_probability": expected_mix_probability,
        "observed_positive_mix_probability": observed_mix_probability,
        "positive_mix_tolerance": mix_tolerance,
        "mix_snr_counts": dict(target_snr),
        "expected_snr_distribution": {
            str(level): weight for level, weight in zip(configured_levels, configured_weights)
        },
        "observed_snr_distribution": observed_snr_distribution,
        "snr_distribution_tolerances": snr_tolerances,
        "probability_checks": probability_checks,
        "max_snr_error_db": max(snr_errors) if snr_errors else None,
        "max_output_peak": maximum_peak,
        "all_finite": finite,
        "split_checks": split_checks,
        "background_pool": "DADS train label=0 only",
    }
    ensure_dirs(args.output.parent)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("Augmentation audit failed")


if __name__ == "__main__":
    main()
