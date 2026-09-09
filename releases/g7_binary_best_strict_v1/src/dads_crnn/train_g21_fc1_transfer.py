from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .config import load_config
from .data_firewall import audit_csv_rows, file_sha256
from .g21_model import G21PartialFineTuneIdentifier
from .panns import PannsCnn14Binary
from .train import resolve_device, set_seed
from .train_g18_model_id import (
    RegistryDataset,
    _batches,
    _verify_inputs,
    evaluate,
    balanced_epoch_indices,
)


PROTOCOL = "g21_p1_known_fc1_transfer_v1"
SCHEMA_VERSION = 1
HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "train_cross_entropy",
    "train_l2_sp",
    "tune_loss",
    "tune_segment_accuracy",
    "tune_segment_macro_f1",
    "tune_recording_accuracy",
    "tune_recording_macro_f1",
    "tune_recording_minimum_recall",
    "improved",
    "bad_epochs",
)


def _verify_config(config: dict[str, Any]) -> None:
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"G21 requires protocol {PROTOCOL}")
    required = {
        ("data", "split_unit"): "audio_sha256",
        ("model", "classifier"): "layer_norm_linear",
    }
    for (section, key), value in required.items():
        if config.get(section, {}).get(key) != value:
            raise ValueError(f"G21 requires {section}.{key}={value!r}")
    prefixes = config.get("model", {}).get("trainable_detector_prefixes")
    if prefixes != ["backbone.fc1."]:
        raise ValueError("G21 v1 only permits narrow PANNs fc1 adaptation")
    serialized = json.dumps(config, ensure_ascii=False).lower()
    forbidden = ("unknown_tune", "known_holdout", "unknown_holdout", "x6d", "y6")
    if any(value in serialized for value in forbidden):
        raise ValueError("G21 must not bind Unknown or Holdout inputs")
    for key in ("head_learning_rate", "detector_learning_rate", "l2_sp_weight"):
        if float(config["train"][key]) <= 0.0:
            raise ValueError(f"G21 requires positive train.{key}")
    if float(config["train"]["detector_learning_rate"]) >= float(
        config["train"]["head_learning_rate"]
    ):
        raise ValueError("G21 detector learning rate must be lower than head rate")


def _build_model(
    config: dict[str, Any],
    paths: dict[str, Path],
    observed: dict[str, str],
    classes: int,
) -> G21PartialFineTuneIdentifier:
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
    return G21PartialFineTuneIdentifier(
        detector,
        embedding_dim=int(config["model"]["embedding_dim"]),
        classes=classes,
        trainable_detector_prefixes=config["model"]["trainable_detector_prefixes"],
    )


def _transfer_state(model: G21PartialFineTuneIdentifier) -> dict[str, torch.Tensor]:
    allowed = ("detector.backbone.fc1.", "embedding_norm.", "classifier.")
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.startswith(allowed)
    }


def _load_transfer_state(
    model: G21PartialFineTuneIdentifier, state: dict[str, torch.Tensor]
) -> None:
    expected = set(_transfer_state(model))
    if set(state) != expected:
        raise ValueError("G21 transfer checkpoint keys do not match")
    model.load_state_dict(state, strict=False)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_history(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _identity(
    config_path: Path, observed: dict[str, str]
) -> dict[str, Any]:
    source = Path(__file__).parent
    return {
        "protocol": PROTOCOL,
        "config_sha256": file_sha256(config_path),
        "input_sha256": observed,
        "source_sha256": {
            "g21_model.py": file_sha256(source / "g21_model.py"),
            "train_g21_fc1_transfer.py": file_sha256(
                source / "train_g21_fc1_transfer.py"
            ),
            "panns.py": file_sha256(source / "panns.py"),
        },
    }


def preflight(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    _verify_config(config)
    paths, observed, registry = _verify_inputs(config, root)
    for split in ("known_train", "known_tune"):
        audit_csv_rows(
            paths[split],
            required_columns=(
                "model_id",
                "target_index",
                "audio_sha256",
                "cache_path",
                "cache_index",
            ),
        )
    model = _build_model(config, paths, observed, len(registry["known_models"]))
    adapted = model.adapted_detector_parameter_names
    frozen = [
        name
        for name, parameter in model.detector.named_parameters()
        if not parameter.requires_grad
    ]
    if adapted != ["backbone.fc1.weight", "backbone.fc1.bias"]:
        raise ValueError(f"G21 adapted an unexpected parameter set: {adapted}")
    if not frozen or any(
        name.startswith("backbone.fc1.") for name in frozen
    ):
        raise ValueError("G21 detector freeze boundary is invalid")

    train_rows = audit_csv_rows(
        paths["known_train"],
        required_columns=("audio_sha256", "target_index"),
    )
    tune_rows = audit_csv_rows(
        paths["known_tune"],
        required_columns=("audio_sha256", "target_index"),
    )
    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "adapted_detector_parameters": adapted,
        "frozen_detector_parameter_count": len(frozen),
        "known_train_rows": int(train_rows),
        "known_tune_rows": int(tune_rows),
        "optimization_inputs": ["known_train"],
        "model_selection_inputs": ["known_tune"],
        "known_holdout_read": False,
        "unknown_inputs_read": False,
        "locked_datasets_read": [],
        "identity": _identity(config_path, observed),
    }
    output = root / str(config["preflight_output_dir"]) / "report.json"
    _atomic_json(report, output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def train(config_path: Path, root: Path, *, resume: bool) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    _verify_config(config)
    paths, observed, registry = _verify_inputs(config, root)
    preflight_path = root / str(config["preflight_output_dir"]) / "report.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("G21 preflight report is missing")
    preflight_report = json.loads(preflight_path.read_text(encoding="utf-8"))
    current_identity = _identity(config_path, observed)
    if (
        preflight_report.get("passed") is not True
        or preflight_report.get("identity") != current_identity
    ):
        raise ValueError("G21 preflight is missing, failed, or stale")

    seed = int(config["train"]["seed"])
    set_seed(seed)
    device = resolve_device(str(config["train"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("G21 transfer training requires the server CUDA GPU")
    classes = len(registry["known_models"])
    target_samples = int(
        float(config["data"]["sample_rate"]) * float(config["data"]["clip_seconds"])
    )
    train_data = RegistryDataset(paths["known_train"], target_samples)
    tune_data = RegistryDataset(paths["known_tune"], target_samples)
    expected_targets = set(range(classes))
    for name, dataset in (("known_train", train_data), ("known_tune", tune_data)):
        if set(dataset.frame["target_index"].astype(int)) != expected_targets:
            raise ValueError(f"G21 {name} does not contain all Known classes")
    if set(train_data.frame["audio_sha256"]) & set(
        tune_data.frame["audio_sha256"]
    ):
        raise ValueError("G21 detected train/tune recording leakage")

    model = _build_model(config, paths, observed, classes).to(device)
    detector_parameters = [
        parameter
        for parameter in model.detector.parameters()
        if parameter.requires_grad
    ]
    head_parameters = list(model.embedding_norm.parameters()) + list(
        model.classifier.parameters()
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": detector_parameters,
                "lr": float(config["train"]["detector_learning_rate"]),
            },
            {
                "params": head_parameters,
                "lr": float(config["train"]["head_learning_rate"]),
            },
        ],
        weight_decay=float(config["train"]["weight_decay"]),
    )

    output_dir = root / str(config["output_dir"])
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.csv"
    summary_path = output_dir / "summary.json"
    identity = current_identity
    start_epoch = 1
    best_f1 = -1.0
    best_minimum_recall = -1.0
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    if resume:
        saved = torch.load(latest_path, map_location="cpu", weights_only=True)
        if saved.get("identity") != identity:
            raise ValueError("G21 resume identity mismatch")
        _load_transfer_state(model, saved["transfer_state"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["epoch"]) + 1
        best_f1 = float(saved["best_f1"])
        best_minimum_recall = float(saved["best_minimum_recall"])
        best_epoch = int(saved["best_epoch"])
        bad_epochs = int(saved["bad_epochs"])
        history = list(saved["history"])

    batch_size = int(config["train"]["batch_size"])
    patience = int(config["train"]["patience"])
    epochs = int(config["train"]["epochs"])
    smoothing = float(config["train"]["label_smoothing"])
    l2_sp_weight = float(config["train"]["l2_sp_weight"])
    targets_np = train_data.frame["target_index"].to_numpy(dtype=np.int64)
    frozen_buffers = {
        name: value.detach().clone()
        for name, value in model.detector.named_buffers()
        if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        indices = balanced_epoch_indices(targets_np, seed=seed, epoch=epoch)
        total_loss = 0.0
        total_ce = 0.0
        total_l2 = 0.0
        processed = 0
        for step, batch_indices in enumerate(
            _batches(indices, batch_size), start=1
        ):
            waveforms, targets = train_data.batch(batch_indices)
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(waveforms.to(device))
            cross_entropy = F.cross_entropy(
                logits.float(),
                targets.to(device),
                label_smoothing=smoothing,
            )
            l2_sp = model.l2_sp_penalty()
            loss = cross_entropy + l2_sp_weight * l2_sp
            if not torch.isfinite(loss):
                raise RuntimeError("G21 produced a non-finite loss")
            loss.backward()
            trainable_gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
            if not all(
                gradient is not None and torch.isfinite(gradient).all()
                for gradient in trainable_gradients
            ):
                raise RuntimeError("G21 trainable gradients are missing or non-finite")
            if any(
                parameter.grad is not None
                for parameter in model.detector.parameters()
                if not parameter.requires_grad
            ):
                raise RuntimeError("G21 crossed its detector adaptation boundary")
            torch.nn.utils.clip_grad_norm_(
                detector_parameters + head_parameters,
                float(config["train"]["gradient_clip_norm"]),
            )
            optimizer.step()
            count = len(batch_indices)
            processed += count
            total_loss += float(loss.detach().cpu()) * count
            total_ce += float(cross_entropy.detach().cpu()) * count
            total_l2 += float(l2_sp.detach().cpu()) * count
            if step % int(config["train"]["log_every_steps"]) == 0:
                print(
                    f"G21 epoch {epoch} step {step}: "
                    f"loss={float(loss.detach().cpu()):.6f} "
                    f"ce={float(cross_entropy.detach().cpu()):.6f} "
                    f"l2sp={float(l2_sp.detach().cpu()):.6f}",
                    flush=True,
                )

        tune_loss, segment, recording = evaluate(
            model,
            tune_data,
            batch_size=batch_size,
            device=device,
            classes=classes,
        )
        for name, reference in frozen_buffers.items():
            if not torch.equal(dict(model.detector.named_buffers())[name], reference):
                raise RuntimeError(f"G21 modified frozen detector buffer {name}")
        improved = (
            recording["macro_f1"] > best_f1 + 1.0e-8
            or (
                abs(recording["macro_f1"] - best_f1) <= 1.0e-8
                and recording["minimum_recall"] > best_minimum_recall + 1.0e-8
            )
        )
        if improved:
            best_f1 = float(recording["macro_f1"])
            best_minimum_recall = float(recording["minimum_recall"])
            best_epoch = epoch
            bad_epochs = 0
            _atomic_torch_save(
                {
                    "protocol": PROTOCOL,
                    "epoch": epoch,
                    "seed": seed,
                    "transfer_state": _transfer_state(model),
                    "known_models": registry["known_models"],
                    "tune_segment_metrics": segment,
                    "tune_recording_metrics": recording,
                    "identity": identity,
                    "known_holdout_read": False,
                    "locked_datasets_read": [],
                },
                best_path,
            )
        else:
            bad_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / processed,
                "train_cross_entropy": total_ce / processed,
                "train_l2_sp": total_l2 / processed,
                "tune_loss": tune_loss,
                "tune_segment_accuracy": segment["accuracy"],
                "tune_segment_macro_f1": segment["macro_f1"],
                "tune_recording_accuracy": recording["accuracy"],
                "tune_recording_macro_f1": recording["macro_f1"],
                "tune_recording_minimum_recall": recording["minimum_recall"],
                "improved": improved,
                "bad_epochs": bad_epochs,
            }
        )
        _write_history(history, history_path)
        _atomic_torch_save(
            {
                "protocol": PROTOCOL,
                "epoch": epoch,
                "transfer_state": _transfer_state(model),
                "optimizer": optimizer.state_dict(),
                "best_f1": best_f1,
                "best_minimum_recall": best_minimum_recall,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "history": history,
                "identity": identity,
            },
            latest_path,
        )
        print(
            f"G21 epoch {epoch}: accuracy={recording['accuracy']:.6f} "
            f"macro_f1={recording['macro_f1']:.6f} "
            f"minimum_recall={recording['minimum_recall']:.6f}",
            flush=True,
        )
        if bad_epochs >= patience:
            break

    best = torch.load(best_path, map_location="cpu", weights_only=True)
    metrics = best["tune_recording_metrics"]
    checks = {
        "accuracy": {
            "observed": metrics["accuracy"],
            "minimum": float(config["gates"]["minimum_tune_recording_accuracy"]),
        },
        "macro_f1": {
            "observed": metrics["macro_f1"],
            "minimum": float(config["gates"]["minimum_tune_recording_macro_f1"]),
        },
        "minimum_recall": {
            "observed": metrics["minimum_recall"],
            "minimum": float(config["gates"]["minimum_tune_recording_recall"]),
        },
    }
    for value in checks.values():
        value["passed"] = value["observed"] >= value["minimum"]
    passed = all(value["passed"] for value in checks.values())
    report = {
        "passed": passed,
        "execution_completed": True,
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": (
            "retain_g21_for_multiseed_ablation"
            if passed
            else "retain_g18_and_stop_g21"
        ),
        "best_epoch": best_epoch,
        "best_metrics": metrics,
        "baseline": config["baseline"],
        "gate_checks": checks,
        "adapted_detector_parameters": model.adapted_detector_parameter_names,
        "learning_rates": {
            "detector": float(config["train"]["detector_learning_rate"]),
            "head": float(config["train"]["head_learning_rate"]),
        },
        "l2_sp_weight": l2_sp_weight,
        "balanced_samples_per_class_per_epoch": int(
            min(Counter(targets_np).values())
        ),
        "identity": identity,
        "optimization_inputs": ["known_train"],
        "model_selection_inputs": ["known_tune"],
        "known_holdout_read": False,
        "unknown_inputs_read": False,
        "locked_datasets_read": [],
    }
    _atomic_json(report, summary_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the G21 narrow PANNs fc1 transfer experiment."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g21_fc1_transfer.yaml")
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight(args.config, args.root)
    else:
        train(args.config, args.root, resume=args.resume)


if __name__ == "__main__":
    main()
