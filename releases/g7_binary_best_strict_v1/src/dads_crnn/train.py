from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ensure_dirs, load_config
from .dataset import DADSDataset
from .features import LogMelSpectrogram, MFCCSpectrogram
from .metrics import binary_metrics, file_level_metrics
from .model import CRNN, ResNet10CBAM
from .sampling import (
    ClassDomainQuotaBatchSampler,
    ClassSourceBalancedBatchSampler,
    ClassSourceBalancedSampler,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(config_device: str) -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def _build_feature_extractor(config: dict) -> nn.Module:
    feature_cfg = dict(config["features"])
    feature_type = str(feature_cfg.pop("type", "log_mel"))
    sample_rate = int(config["data"]["sample_rate"])
    if feature_type == "log_mel":
        return LogMelSpectrogram(sample_rate=sample_rate, **feature_cfg)
    if feature_type == "mfcc":
        return MFCCSpectrogram(sample_rate=sample_rate, **feature_cfg)
    raise ValueError(f"Unsupported feature type: {feature_type!r}")


def _build_model(config: dict) -> nn.Module:
    model_cfg = dict(config["model"])
    model_type = str(model_cfg.pop("type", "crnn"))
    if model_type == "crnn":
        feature_cfg = config["features"]
        feature_bins = (
            int(feature_cfg["n_mfcc"])
            if str(feature_cfg.get("type", "log_mel")) == "mfcc"
            else int(feature_cfg["n_mels"])
        )
        return CRNN(n_mels=feature_bins, **model_cfg)
    if model_type == "resnet10_cbam":
        return ResNet10CBAM(**model_cfg)
    raise ValueError(f"Unsupported model type: {model_type!r}")


def _training_criterion(config: dict, train_loader: DataLoader, device: torch.device) -> nn.Module:
    pos_weight = config["train"].get("pos_weight")
    if pos_weight != "auto":
        if pos_weight is None:
            return nn.BCEWithLogitsLoss()
        weight = torch.tensor(float(pos_weight), device=device)
        return nn.BCEWithLogitsLoss(pos_weight=weight)

    expected = getattr(train_loader.sampler, "expected_label_probabilities", None)
    if expected is None:
        expected = getattr(
            train_loader.batch_sampler, "expected_label_probabilities", None
        )
    if expected is not None:
        ratio = float(expected[0]) / float(expected[1])
        source = "sampling distribution"
    else:
        labels = train_loader.dataset.rows["label"].to_numpy(dtype=np.int64)
        positives = int(labels.sum())
        negatives = int(labels.size - positives)
        if positives == 0:
            raise ValueError("Cannot compute automatic pos_weight without positive training samples")
        ratio = negatives / positives
        source = "manifest distribution"
    weight = torch.tensor(ratio, dtype=torch.float32, device=device)
    print(f"Using automatic positive-class weight from {source}: {float(weight):.4f}")
    return nn.BCEWithLogitsLoss(pos_weight=weight)


def _build_loaders(config: dict, manifest_path: Path, seed: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = config["data"]
    if bool(data_cfg.get("require_segment_guard", False)) and bool(
        data_cfg.get("require_leakage_fixed_guard", False)
    ):
        raise ValueError(
            "Only one DADS manifest guard protocol may be enabled"
        )
    if bool(data_cfg.get("require_segment_guard", False)):
        from .prepare_segment_guarded_manifest import validate_guarded_manifest

        guard = validate_guarded_manifest(
            manifest_path,
            expected_segments_per_class=(
                int(data_cfg["expected_segments_per_class"])
                if data_cfg.get("expected_segments_per_class") is not None
                else None
            ),
            split_ratios=data_cfg.get("splits"),
            verify_cache_content=bool(
                data_cfg.get("verify_segment_cache_hashes", False)
            ),
            root=Path("."),
            target_samples=int(
                int(data_cfg["sample_rate"])
                * float(data_cfg["clip_seconds"])
            ),
        )
        print(
            "Validated segment leakage guard: "
            f"protocol={guard['protocol']}, rows={guard['counts']['total']}"
        )
    if bool(data_cfg.get("require_leakage_fixed_guard", False)):
        from .prepare_dads_leakage_fixed import validate_manifest

        if not bool(data_cfg.get("verify_leakage_fixed_cache_file", False)):
            raise ValueError(
                "Leakage-fixed DADS training requires full cache-file and "
                "cache-row verification"
            )
        split_names = data_cfg.get(
            "split_names", {"train": "train", "val": "val", "test": "test"}
        )
        if split_names != {
            "train": "train",
            "val": "val",
            "test": "test",
        }:
            raise ValueError(
                "Leakage-fixed DADS roles must remain train/val/test"
            )
        audit_value = data_cfg.get("leakage_fixed_audit_path")
        if not audit_value:
            raise ValueError(
                "require_leakage_fixed_guard requires leakage_fixed_audit_path"
            )
        guard = validate_manifest(
            manifest_path,
            Path(str(audit_value)),
            verify_cache_file=bool(
                data_cfg.get("verify_leakage_fixed_cache_file", False)
            ),
            expected_sample_rate=int(data_cfg["sample_rate"]),
            expected_clip_seconds=float(data_cfg["clip_seconds"]),
            expected_split_ratios=data_cfg.get("splits"),
        )
        print(
            "Validated native half-second leakage guard: "
            f"protocol={guard['protocol']}, rows={guard['counts']['total']}"
        )
    batch_size = int(config["train"]["batch_size"])
    num_workers = int(config["train"]["num_workers"])
    split_names = data_cfg.get(
        "split_names", {"train": "train", "val": "val", "test": "test"}
    )
    train_ds = DADSDataset(
        manifest_path,
        str(split_names["train"]),
        sample_rate=int(data_cfg["sample_rate"]),
        clip_seconds=float(data_cfg["clip_seconds"]),
        training=True,
        seed=seed,
        augmentation=config["train"].get("augmentation"),
    )
    val_ds = DADSDataset(
        manifest_path,
        str(split_names["val"]),
        sample_rate=int(data_cfg["sample_rate"]),
        clip_seconds=float(data_cfg["clip_seconds"]),
        training=False,
        seed=seed,
    )
    test_ds = DADSDataset(
        manifest_path,
        str(split_names["test"]),
        sample_rate=int(data_cfg["sample_rate"]),
        clip_seconds=float(data_cfg["clip_seconds"]),
        training=False,
        seed=seed,
    )
    generator = torch.Generator().manual_seed(seed)
    sampling_cfg = config["train"].get("sampling")
    sampler = None
    batch_sampler = None
    if sampling_cfg:
        sampling_type = str(sampling_cfg.get("type", ""))
        requested_samples = sampling_cfg.get("samples_per_epoch", "dataset")
        num_samples = len(train_ds) if requested_samples == "dataset" else int(requested_samples)
        source_column = str(sampling_cfg.get("source_column", "source_path"))
        if sampling_type == "class_source_balanced":
            sampler = ClassSourceBalancedSampler(
                train_ds.rows,
                source_column=source_column,
                num_samples=num_samples,
                seed=seed,
            )
            print(
                f"Using class-source balanced sampling: {num_samples} samples/epoch, "
                f"sources={sampler.source_counts}"
            )
        elif sampling_type == "class_source_sqrt_balanced_batch":
            if num_samples % batch_size:
                raise ValueError("samples_per_epoch must be divisible by batch_size")
            batch_sampler = ClassSourceBalancedBatchSampler(
                train_ds.rows,
                source_column=source_column,
                batch_size=batch_size,
                num_batches=num_samples // batch_size,
                source_weight_exponent=float(
                    sampling_cfg.get("source_weight_exponent", 0.5)
                ),
                seed=seed,
            )
            print(
                f"Using exact class-balanced batches: {num_samples} samples/epoch, "
                f"sources={batch_sampler.source_counts}, "
                f"source_weight_exponent={batch_sampler.source_weight_exponent}"
            )
        elif sampling_type == "class_domain_quota_batch":
            if num_samples % batch_size:
                raise ValueError("samples_per_epoch must be divisible by batch_size")
            batch_sampler = ClassDomainQuotaBatchSampler(
                train_ds.rows,
                domain_column=str(sampling_cfg.get("domain_column", "domain_bucket")),
                batch_size=batch_size,
                positive_per_batch=int(sampling_cfg["positive_per_batch"]),
                num_batches=num_samples // batch_size,
                positive_domain=str(sampling_cfg["positive_domain"]),
                negative_domain_fractions={
                    str(key): float(value)
                    for key, value in sampling_cfg["negative_domain_fractions"].items()
                },
                seed=seed,
            )
            print(
                f"Using exact class/domain quota batches: {num_samples} samples/epoch, "
                f"positive_per_batch={batch_sampler.positive_per_batch}, "
                f"negative_domain_draws={batch_sampler.negative_domain_draws}"
            )
        else:
            raise ValueError(f"Unsupported training sampling type: {sampling_type!r}")
    if batch_sampler is not None:
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            generator=generator,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            generator=generator,
        )
    return (
        train_loader,
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def _predict(
    model: nn.Module,
    feature_extractor: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for waveform, target in loader:
            waveform = waveform.to(device)
            target = target.to(device)
            logits = model(feature_extractor(waveform))
            loss = criterion(logits, target)
            total_loss += float(loss.item()) * waveform.size(0)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(target.cpu().numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(labels).astype(np.int64)
    return y_true, y_prob, total_loss / len(loader.dataset)


def train_one_seed(config: dict, manifest_path: Path, seed: int) -> dict:
    set_seed(seed)
    device = resolve_device(str(config["train"]["device"]))
    run_dir = Path(config["output_dir"]) / f"seed_{seed}"
    ensure_dirs(run_dir)

    train_loader, val_loader, test_loader = _build_loaders(config, manifest_path, seed)
    feature_extractor = _build_feature_extractor(config).to(device)
    model = _build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["learning_rate"]))
    criterion = _training_criterion(config, train_loader, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {parameter_count:,}")

    best_f1 = -1.0
    best_epoch = 0
    bad_epochs = 0
    history_path = run_dir / "history.csv"
    start_time = time.time()

    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy", "val_f1", "val_auc"])
        writer.writeheader()
        for epoch in range(1, int(config["train"]["epochs"]) + 1):
            if hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch - 1)
            model.train()
            train_loss = 0.0
            progress = tqdm(train_loader, desc=f"seed {seed} epoch {epoch}", leave=False)
            for waveform, target in progress:
                waveform = waveform.to(device)
                target = target.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(feature_extractor(waveform))
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item()) * waveform.size(0)
                progress.set_postfix(loss=float(loss.item()))

            y_true, y_prob, val_loss = _predict(model, feature_extractor, val_loader, device)
            val_metrics = binary_metrics(y_true, y_prob, threshold=0.50)
            train_loss /= len(train_loader.dataset)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1": val_metrics["f1"],
                    "val_auc": val_metrics["auc"],
                }
            )
            handle.flush()

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_epoch = epoch
                bad_epochs = 0
                torch.save(
                    {
                        "seed": seed,
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "config": config,
                    },
                    run_dir / "best.pt",
                )
            else:
                bad_epochs += 1

            if bad_epochs >= int(config["train"]["patience"]):
                break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    val_true, val_prob, best_val_loss = _predict(model, feature_extractor, val_loader, device)
    y_true, y_prob, test_loss = _predict(model, feature_extractor, test_loader, device)
    val_metrics = [binary_metrics(val_true, val_prob, threshold=t) for t in config["eval"]["thresholds"]]
    test_metrics = [binary_metrics(y_true, y_prob, threshold=t) for t in config["eval"]["thresholds"]]
    val_file_metrics = {
        aggregation: file_level_metrics(
            val_loader.dataset.rows,
            val_prob,
            config["eval"]["thresholds"],
            aggregation=aggregation,
        )
        for aggregation in ("mean", "max")
    }
    test_file_metrics = {
        aggregation: file_level_metrics(
            test_loader.dataset.rows,
            y_prob,
            config["eval"]["thresholds"],
            aggregation=aggregation,
        )
        for aggregation in ("mean", "max")
    }
    elapsed_seconds = time.time() - start_time
    result = {
        "seed": seed,
        "device": str(device),
        "model_type": str(config["model"].get("type", "crnn")),
        "feature_type": str(config["features"].get("type", "log_mel")),
        "temporal_pooling": (
            str(config["model"].get("temporal_pooling", "mean"))
            if str(config["model"].get("type", "crnn")) == "crnn"
            else None
        ),
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "val_loss": best_val_loss,
        "val_threshold_metrics": val_metrics,
        "val_file_metrics": val_file_metrics,
        "test_loss": test_loss,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_seconds / 60.0,
        "threshold_metrics": test_metrics,
        "test_file_metrics": test_file_metrics,
    }
    np.save(run_dir / "val_probabilities.npy", val_prob)
    np.save(run_dir / "val_labels.npy", val_true)
    np.save(run_dir / "test_probabilities.npy", y_prob)
    np.save(run_dir / "test_labels.npy", y_true)
    result["checkpoint_sha256"] = _sha256(run_dir / "best.pt")
    result["prediction_sha256"] = {
        name: _sha256(run_dir / f"{name}.npy")
        for name in (
            "val_probabilities",
            "val_labels",
            "test_probabilities",
            "test_labels",
        )
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(
        f"seed {seed} complete: best_epoch={best_epoch}, "
        f"best_val_f1={best_f1:.4f}, elapsed={elapsed_seconds / 60.0:.2f} min"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a configured DADS acoustic classifier.")
    parser.add_argument("--config", default="configs/crnn_dads.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Explicitly permit replacing files in an existing seed run directory.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = Path(args.manifest) if args.manifest else next(Path(config["data"]["manifest_dir"]).glob("dads_*.csv"))
    seeds = args.seeds or [int(seed) for seed in config["train"]["seeds"]]
    if not args.allow_overwrite:
        occupied = []
        for seed in seeds:
            run_dir = Path(config["output_dir"]) / f"seed_{seed}"
            if run_dir.exists() and any(run_dir.iterdir()):
                occupied.append(run_dir.as_posix())
        if occupied:
            raise FileExistsError(
                "Refusing to overwrite existing runs: "
                + ", ".join(occupied)
                + ". Use a new output_dir or pass --allow-overwrite explicitly."
            )
    results = [train_one_seed(config, manifest_path, seed) for seed in seeds]

    summary_path = Path(config["output_dir"]) / "summary.json"
    ensure_dirs(summary_path.parent)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
