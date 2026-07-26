from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.nn import functional as F

from .config import load_config
from .data_firewall import audit_csv_rows, file_sha256, reject_locked_path
from .model_identification import G7ModelIdentifier
from .panns import PannsCnn14Binary
from .train import resolve_device, set_seed


PROTOCOL = "g18_p2_seed42_frozen_g7_model_id_feasibility_v1"
HEAD_PREFIXES = ("embedding_norm.", "classifier.")
HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "tune_loss",
    "segment_accuracy",
    "segment_macro_f1",
    "recording_accuracy",
    "recording_macro_f1",
    "recording_min_recall",
    "improved",
    "bad_epochs",
]


def balanced_epoch_indices(
    targets: np.ndarray, *, seed: int, epoch: int
) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.int64)
    classes = np.unique(targets)
    if len(classes) < 2:
        raise ValueError("G18 training requires multiple known models")
    positions = [np.flatnonzero(targets == value) for value in classes]
    count = min(len(values) for values in positions)
    if count <= 0:
        raise ValueError("G18 contains an empty model class")
    rng = np.random.default_rng(seed + epoch * 1009)
    selected = np.concatenate(
        [rng.choice(values, size=count, replace=False) for values in positions]
    )
    rng.shuffle(selected)
    return selected


def aggregate_recording_logits(
    frame: pd.DataFrame, logits: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(frame) != len(logits) or logits.ndim != 2:
        raise ValueError("G18 recording aggregation inputs are misaligned")
    targets = []
    values = []
    for _, indices in frame.groupby("audio_sha256", sort=True).indices.items():
        positions = np.asarray(indices, dtype=np.int64)
        group_targets = frame.iloc[positions]["target_index"].astype(int).unique()
        if len(group_targets) != 1:
            raise ValueError("One G18 raw recording has conflicting model targets")
        targets.append(int(group_targets[0]))
        values.append(np.asarray(logits[positions], dtype=np.float64).mean(axis=0))
    return np.asarray(targets, dtype=np.int64), np.stack(values)


def classification_metrics(
    targets: np.ndarray, logits: np.ndarray, classes: int
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (len(targets), classes) or not np.isfinite(logits).all():
        raise ValueError("Invalid G18 classification logits")
    predictions = logits.argmax(axis=1)
    labels = list(range(classes))
    recalls = recall_score(
        targets, predictions, labels=labels, average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(
                targets, predictions, labels=labels, average="macro", zero_division=0
            )
        ),
        "minimum_recall": float(np.min(recalls)),
        "per_class_recall": [float(value) for value in recalls],
    }


class RegistryDataset:
    def __init__(self, path: Path, target_samples: int) -> None:
        self.frame = pd.read_csv(path)
        self.target_samples = int(target_samples)
        self.maps: OrderedDict[str, np.ndarray] = OrderedDict()

    def batch(self, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        waveforms = []
        targets = []
        for position in indices:
            row = self.frame.iloc[int(position)]
            cache_path = str(row["cache_path"])
            reject_locked_path(Path(cache_path), context="G18 training cache")
            if cache_path not in self.maps:
                self.maps[cache_path] = np.load(cache_path, mmap_mode="r")
            waveform = np.asarray(
                self.maps[cache_path][int(row["cache_index"])], dtype=np.float32
            ).reshape(-1)
            if waveform.size != self.target_samples or not np.isfinite(waveform).all():
                raise ValueError("Invalid G18 training waveform")
            waveforms.append(waveform)
            targets.append(int(row["target_index"]))
        return (
            torch.from_numpy(np.stack(waveforms)),
            torch.tensor(targets, dtype=torch.long),
        )


def _batches(indices: np.ndarray, batch_size: int):
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def _head_state(model: G7ModelIdentifier) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.startswith(HEAD_PREFIXES)
    }


def _load_head(model: G7ModelIdentifier, state: dict[str, torch.Tensor]) -> None:
    expected = {name for name in model.state_dict() if name.startswith(HEAD_PREFIXES)}
    if set(state) != expected:
        raise ValueError("G18 head checkpoint keys do not match")
    model.load_state_dict(state, strict=False)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    reject_locked_path(path, context="G18 P2 input")
    return path.resolve(strict=True)


def _verify_inputs(
    config: dict[str, Any], root: Path
) -> tuple[dict[str, Path], dict[str, str], dict[str, Any]]:
    paths = {
        "known_train": _resolve(root, config["inputs"]["known_train"]["path"]),
        "known_tune": _resolve(root, config["inputs"]["known_tune"]["path"]),
        "registry_audit": _resolve(root, config["inputs"]["registry_audit"]["path"]),
        "p1_preflight": _resolve(root, config["inputs"]["p1_preflight"]["path"]),
        "official_checkpoint": _resolve(root, config["model"]["checkpoint_path"]),
        "g7_checkpoint": _resolve(root, config["model"]["binary_checkpoint_path"]),
    }
    expected = {
        "known_train": str(config["inputs"]["known_train"]["sha256"]),
        "known_tune": str(config["inputs"]["known_tune"]["sha256"]),
        "registry_audit": str(config["inputs"]["registry_audit"]["sha256"]),
        "p1_preflight": str(config["inputs"]["p1_preflight"]["sha256"]),
        "official_checkpoint": str(config["model"]["checkpoint_sha256"]),
        "g7_checkpoint": str(config["model"]["binary_checkpoint_sha256"]),
    }
    observed = {name: file_sha256(path) for name, path in paths.items()}
    mismatches = {
        name: (expected[name], observed[name])
        for name in expected
        if expected[name] != observed[name]
    }
    if mismatches:
        raise ValueError(f"G18 P2 input SHA256 mismatch: {mismatches}")
    registry = json.loads(paths["registry_audit"].read_text(encoding="utf-8"))
    preflight = json.loads(paths["p1_preflight"].read_text(encoding="utf-8"))
    if not (
        registry.get("passed") is True
        and registry.get("locked_datasets_read") == []
        and preflight.get("passed") is True
        and preflight.get("frozen_g7_gradients_absent") is True
        and preflight.get("trainable_gradients_finite") is True
        and preflight.get("ready_for_seed42_feasibility_training") is True
    ):
        raise ValueError("G18 P0/P1 prerequisites are not valid")
    return paths, observed, registry


@torch.no_grad()
def evaluate(
    model: G7ModelIdentifier,
    dataset: RegistryDataset,
    *,
    batch_size: int,
    device: torch.device,
    classes: int,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    model.eval()
    all_logits = []
    losses = []
    indices = np.arange(len(dataset.frame), dtype=np.int64)
    for batch_indices in _batches(indices, batch_size):
        waveforms, targets = dataset.batch(batch_indices)
        _, logits = model(waveforms.to(device))
        losses.append(
            F.cross_entropy(logits.float(), targets.to(device), reduction="sum")
            .cpu()
            .item()
        )
        all_logits.append(logits.float().cpu().numpy())
    logits = np.concatenate(all_logits)
    targets = dataset.frame["target_index"].to_numpy(dtype=np.int64)
    segment = classification_metrics(targets, logits, classes)
    recording_targets, recording_logits = aggregate_recording_logits(
        dataset.frame, logits
    )
    recording = classification_metrics(recording_targets, recording_logits, classes)
    return sum(losses) / len(dataset.frame), segment, recording


def train(config_path: Path, root: Path, *, resume: bool) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected G18 P2 protocol")
    paths, observed, registry = _verify_inputs(config, root)
    forbidden_inputs = ("unknown_tune", "known_holdout", "unknown_holdout")
    if any(name in config["inputs"] for name in forbidden_inputs):
        raise ValueError("G18 P2 must not bind Unknown or Holdout inputs")
    for name in ("known_train", "known_tune"):
        audit_csv_rows(
            paths[name],
            required_columns=(
                "model_id",
                "target_index",
                "audio_sha256",
                "cache_path",
                "cache_index",
            ),
        )

    seed = int(config["train"]["seed"])
    set_seed(seed)
    device = resolve_device(str(config["train"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("G18 P2 training requires the server CUDA GPU")
    known_models = list(registry["known_models"])
    classes = len(known_models)
    target_samples = int(
        config["data"]["sample_rate"] * config["data"]["clip_seconds"]
    )
    train_data = RegistryDataset(paths["known_train"], target_samples)
    tune_data = RegistryDataset(paths["known_tune"], target_samples)
    expected_targets = set(range(classes))
    if set(train_data.frame["target_index"].astype(int)) != expected_targets:
        raise ValueError("G18 training targets are incomplete")
    if set(tune_data.frame["target_index"].astype(int)) != expected_targets:
        raise ValueError("G18 tune targets are incomplete")

    detector = PannsCnn14Binary(
        initialization=str(config["model"]["initialization"]),
        vendor_dir=str(config["model"]["vendor_dir"]),
        checkpoint_path=paths["official_checkpoint"].as_posix(),
        checkpoint_sha256=observed["official_checkpoint"],
        spec_augment=False,
        frontend_precision="float32",
        binary_checkpoint_path=paths["g7_checkpoint"].as_posix(),
        binary_checkpoint_sha256=observed["g7_checkpoint"],
        trainable_scope="binary_head_only",
    )
    model = G7ModelIdentifier(
        detector,
        embedding_dim=int(config["model_id_head"]["embedding_dim"]),
        classes=classes,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    output_dir = root / str(config["output_dir"]) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.csv"
    summary_path = output_dir / "summary.json"
    identity = {
        "config_sha256": file_sha256(config_path),
        **{f"{name}_sha256": value for name, value in observed.items()},
    }
    start_epoch = 1
    best_recording_f1 = -1.0
    best_segment_f1 = -1.0
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    if resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"G18 resume checkpoint not found: {latest_path}")
        saved = torch.load(latest_path, map_location="cpu", weights_only=True)
        if saved.get("identity") != identity:
            raise ValueError("G18 resume input identity mismatch")
        _load_head(model, saved["head_state"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["epoch"]) + 1
        best_recording_f1 = float(saved["best_recording_f1"])
        best_segment_f1 = float(saved["best_segment_f1"])
        best_epoch = int(saved["best_epoch"])
        bad_epochs = int(saved["bad_epochs"])
        history = list(saved["history"])

    batch_size = int(config["train"]["batch_size"])
    epochs = int(config["train"]["epochs"])
    patience = int(config["train"]["patience"])
    label_smoothing = float(config["train"]["label_smoothing"])
    targets_np = train_data.frame["target_index"].to_numpy(dtype=np.int64)
    bn_buffers = {
        name: value.detach().clone()
        for name, value in model.detector.named_buffers()
        if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        indices = balanced_epoch_indices(targets_np, seed=seed, epoch=epoch)
        running_loss = 0.0
        processed = 0
        for step, batch_indices in enumerate(_batches(indices, batch_size), start=1):
            waveforms, targets = train_data.batch(batch_indices)
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(waveforms.to(device))
            loss = F.cross_entropy(
                logits.float(),
                targets.to(device),
                label_smoothing=label_smoothing,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("G18 produced a non-finite loss")
            loss.backward()
            head_gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
            if not all(
                gradient is not None and torch.isfinite(gradient).all()
                for gradient in head_gradients
            ):
                raise RuntimeError("G18 produced non-finite head gradients")
            if any(parameter.grad is not None for parameter in model.detector.parameters()):
                raise RuntimeError("G18 produced gradients in frozen G7")
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(config["train"]["gradient_clip_norm"]),
            )
            optimizer.step()
            count = len(batch_indices)
            running_loss += float(loss.detach().cpu()) * count
            processed += count
            if step % int(config["train"]["log_every_steps"]) == 0:
                print(
                    f"G18 seed {seed} epoch {epoch} step {step} "
                    f"loss={float(loss.detach().cpu()):.6f}",
                    flush=True,
                )

        tune_loss, segment, recording = evaluate(
            model,
            tune_data,
            batch_size=batch_size,
            device=device,
            classes=classes,
        )
        for name, expected_buffer in bn_buffers.items():
            observed_buffer = dict(model.detector.named_buffers())[name]
            if not torch.equal(observed_buffer, expected_buffer):
                raise RuntimeError(f"G18 modified frozen G7 buffer: {name}")
        improved = (
            recording["macro_f1"] > best_recording_f1 + 1.0e-8
            or (
                abs(recording["macro_f1"] - best_recording_f1) <= 1.0e-8
                and segment["macro_f1"] > best_segment_f1 + 1.0e-8
            )
        )
        if improved:
            best_recording_f1 = recording["macro_f1"]
            best_segment_f1 = segment["macro_f1"]
            best_epoch = epoch
            bad_epochs = 0
            _atomic_torch_save(
                {
                    "protocol": PROTOCOL,
                    "seed": seed,
                    "epoch": epoch,
                    "head_state": _head_state(model),
                    "known_models": known_models,
                    "identity": identity,
                    "tune_segment_metrics": segment,
                    "tune_recording_metrics": recording,
                    "locked_datasets_read": [],
                },
                best_path,
            )
        else:
            bad_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / processed,
                "tune_loss": tune_loss,
                "segment_accuracy": segment["accuracy"],
                "segment_macro_f1": segment["macro_f1"],
                "recording_accuracy": recording["accuracy"],
                "recording_macro_f1": recording["macro_f1"],
                "recording_min_recall": recording["minimum_recall"],
                "improved": improved,
                "bad_epochs": bad_epochs,
            }
        )
        _write_history(history_path, history)
        _atomic_torch_save(
            {
                "protocol": PROTOCOL,
                "seed": seed,
                "epoch": epoch,
                "head_state": _head_state(model),
                "optimizer": optimizer.state_dict(),
                "best_recording_f1": best_recording_f1,
                "best_segment_f1": best_segment_f1,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "history": history,
                "identity": identity,
            },
            latest_path,
        )
        print(
            f"G18 seed {seed} epoch {epoch} complete: "
            f"recording_macro_f1={recording['macro_f1']:.6f} "
            f"recording_min_recall={recording['minimum_recall']:.6f} "
            f"improved={improved} bad_epochs={bad_epochs}",
            flush=True,
        )
        if bad_epochs >= patience:
            print(f"G18 early stopping at epoch {epoch}", flush=True)
            break

    best = torch.load(best_path, map_location="cpu", weights_only=True)
    feasibility = best["tune_recording_metrics"]["macro_f1"] >= float(
        config["gates"]["minimum_tune_recording_macro_f1"]
    ) and best["tune_recording_metrics"]["minimum_recall"] >= float(
        config["gates"]["minimum_tune_recording_recall"]
    )
    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "decision": (
            "proceed_to_unknown_calibration"
            if feasibility
            else "stop_model_identification_branch"
        ),
        "best_epoch": best_epoch,
        "best_tune_segment_metrics": best["tune_segment_metrics"],
        "best_tune_recording_metrics": best["tune_recording_metrics"],
        "feasibility_gate_passed": feasibility,
        "g7_frozen": True,
        "g7_checkpoint_sha256": observed["g7_checkpoint"],
        "known_models": known_models,
        "balanced_samples_per_class_per_epoch": int(
            min(Counter(targets_np).values())
        ),
        "unknown_tune_read": False,
        "known_holdout_read": False,
        "unknown_holdout_read": False,
        "locked_datasets_read": [],
        "formal_external_evaluation_run": False,
        "outputs": {
            "best_checkpoint": {
                "path": best_path.relative_to(root).as_posix(),
                "sha256": file_sha256(best_path),
            },
            "history": {
                "path": history_path.relative_to(root).as_posix(),
                "sha256": file_sha256(history_path),
            },
        },
        "inputs": identity,
    }
    summary_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the seed-42 frozen-G7 model-ID head.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g18_model_id_seed42.yaml")
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args.config, args.root, resume=args.resume)


if __name__ == "__main__":
    main()
