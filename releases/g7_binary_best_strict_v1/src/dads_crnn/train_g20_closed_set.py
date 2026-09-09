from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .config import load_config
from .data_firewall import audit_csv_rows, file_sha256, reject_locked_path
from .g19_model import supervised_contrastive_loss
from .g19_recording_data import (
    RecordingExample,
    RecordingFeatureDataset,
    balanced_recording_batches,
    build_recording_examples,
    collate_recordings,
)
from .g20_model import (
    G20ClosedSetHead,
    attention_regularization,
    normalized_attention_entropy,
)
from .train import resolve_device, set_seed
from .train_g18_model_id import classification_metrics
from .train_g19_representation import (
    _capture_rng_state,
    _restore_rng_state,
)


PROTOCOL = "g20_p1_known_multiscale_attention_v1"
SCHEMA_VERSION = 1
REQUIRED_COLUMNS = (
    "model_id",
    "target_index",
    "is_known",
    "audio_sha256",
    "segment_index",
)
HISTORY_FIELDS = (
    "epoch",
    "learning_rate",
    "train_loss",
    "train_recording_ce",
    "train_supcon",
    "train_segment_ce",
    "train_attention_diversity",
    "train_attention_focus",
    "tune_loss",
    "tune_accuracy",
    "tune_macro_f1",
    "tune_minimum_recall",
    "tune_attention_entropy",
    "tune_attention_similarity",
    "improved",
    "bad_epochs",
)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_history(rows: list[dict[str, Any]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _resolve(root: Path, raw: object, *, context: str) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = root / path
    reject_locked_path(path, context=context)
    return path.resolve(strict=True)


def _output(root: Path, raw: object, *, context: str) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{context} must remain inside the workspace") from error
    reject_locked_path(path, context=context)
    return path


def _source_sha256() -> dict[str, str]:
    source = Path(__file__).parent
    return {
        name: file_sha256(source / name)
        for name in (
            "g19_model.py",
            "g19_recording_data.py",
            "g20_model.py",
            "train_g20_closed_set.py",
        )
    }


def _verify_config(config: dict[str, Any]) -> None:
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"Expected protocol {PROTOCOL}")
    expected = {
        ("data", "split_unit"): "audio_sha256",
        ("data", "g7_embedding_dim"): 2048,
        ("model", "aggregation"): "multi_head_attentive_statistics",
        ("model", "classifier"): "cosine",
        ("reporting", "primary_unit"): "recording",
    }
    for (section, key), required in expected.items():
        if config.get(section, {}).get(key) != required:
            raise ValueError(f"G20 requires {section}.{key}={required!r}")
    serialized = json.dumps(config, ensure_ascii=False).lower()
    forbidden = ("unknown_tune", "known_holdout", "unknown_holdout", "x6d", "y6")
    if any(token in serialized for token in forbidden):
        raise ValueError("G20 closed-set config must not bind Unknown or Holdout data")
    weights = (
        "supcon_weight",
        "segment_ce_weight",
        "attention_diversity_weight",
        "attention_focus_weight",
    )
    if any(float(config["train"][name]) < 0.0 for name in weights):
        raise ValueError("G20 auxiliary loss weights cannot be negative")
    entropy = float(config["train"]["maximum_normalized_attention_entropy"])
    if not 0.0 <= entropy <= 1.0:
        raise ValueError("G20 maximum attention entropy must be in [0, 1]")


def _verify_inputs(
    config: dict[str, Any], root: Path
) -> tuple[dict[str, Path], dict[str, str], list[str]]:
    _verify_config(config)
    paths: dict[str, Path] = {}
    for split in ("known_train", "known_tune"):
        spec = config["inputs"][split]
        paths[split] = _resolve(root, spec["path"], context=f"G20 {split}")
        paths[f"{split}_feature"] = _resolve(
            root, spec["feature_path"], context=f"G20 {split} features"
        )
        paths[f"{split}_feature_metadata"] = _resolve(
            root,
            spec["feature_metadata_path"],
            context=f"G20 {split} feature metadata",
        )
    for name in (
        "registry_audit",
        "segment_cache_audit",
        "official_checkpoint",
        "g7_checkpoint",
    ):
        paths[name] = _resolve(
            root, config["inputs"][name]["path"], context=f"G20 {name}"
        )
    expected = {}
    for split in ("known_train", "known_tune"):
        spec = config["inputs"][split]
        expected[split] = str(spec["sha256"])
        expected[f"{split}_feature"] = str(spec["feature_sha256"])
        expected[f"{split}_feature_metadata"] = str(
            spec["feature_metadata_sha256"]
        )
    for name in (
        "registry_audit",
        "segment_cache_audit",
        "official_checkpoint",
        "g7_checkpoint",
    ):
        expected[name] = str(config["inputs"][name]["sha256"])
    observed = {name: file_sha256(paths[name]) for name in expected}
    mismatches = {
        name: {"expected": expected[name], "observed": observed[name]}
        for name in expected
        if expected[name] != observed[name]
    }
    if mismatches:
        raise ValueError(f"G20 input SHA256 mismatch: {mismatches}")

    registry = json.loads(paths["registry_audit"].read_text(encoding="utf-8"))
    if registry.get("passed") is not True or registry.get("locked_datasets_read") != []:
        raise ValueError("G20 requires a valid registry audit")
    known_models = [str(value) for value in registry["known_models"]]
    if known_models != [str(value) for value in config["reporting"]["known_models"]]:
        raise ValueError("G20 Known model order differs from the registry")
    for split in ("known_train", "known_tune"):
        if registry["outputs"][split]["sha256"] != observed[split]:
            raise ValueError(f"G20 {split} is not registry-bound")
        audit_csv_rows(paths[split], required_columns=REQUIRED_COLUMNS)
        metadata = json.loads(
            paths[f"{split}_feature_metadata"].read_text(encoding="utf-8")
        )
        if (
            metadata.get("split_name") != split
            or metadata.get("manifest_sha256") != observed[split]
            or metadata.get("feature_sha256") != observed[f"{split}_feature"]
            or metadata.get("g7_checkpoint_sha256") != observed["g7_checkpoint"]
            or metadata.get("official_checkpoint_sha256")
            != observed["official_checkpoint"]
            or metadata.get("segment_cache_audit_sha256")
            != observed["segment_cache_audit"]
            or int(metadata.get("dimension", -1))
            != int(config["data"]["g7_embedding_dim"])
            or metadata.get("dtype") != "float32"
        ):
            raise ValueError(f"G20 {split} feature-cache identity mismatch")
    return paths, observed, known_models


def _load_data(
    config: dict[str, Any],
    paths: dict[str, Path],
    known_models: list[str],
) -> tuple[RecordingFeatureDataset, RecordingFeatureDataset, dict[str, Any]]:
    datasets = {}
    report = {}
    recording_sets = {}
    for split in ("known_train", "known_tune"):
        frame = pd.read_csv(paths[split])
        examples = build_recording_examples(frame)
        if not examples or not all(example.is_known for example in examples):
            raise ValueError(f"G20 {split} must contain only Known recordings")
        mapping = sorted({(example.target, example.model_id) for example in examples})
        if [item[0] for item in mapping] != list(range(len(known_models))):
            raise ValueError(f"G20 {split} target indices changed")
        if [item[1] for item in mapping] != known_models:
            raise ValueError(f"G20 {split} model mapping changed")
        if len(frame) != int(config["inputs"][split]["rows"]):
            raise ValueError(f"G20 {split} row count changed")
        if len(examples) != int(config["inputs"][split]["recordings"]):
            raise ValueError(f"G20 {split} recording count changed")
        maximum = int(config["data"]["maximum_segments_per_recording"])
        if any(len(example.indices) > maximum for example in examples):
            raise ValueError(f"G20 {split} recording exceeds segment maximum")
        features = np.load(
            paths[f"{split}_feature"], mmap_mode="r", allow_pickle=False
        )
        if (
            features.shape
            != (len(frame), int(config["data"]["g7_embedding_dim"]))
            or features.dtype != np.float32
            or not np.isfinite(features).all()
        ):
            raise ValueError(f"G20 {split} features are invalid")
        datasets[split] = RecordingFeatureDataset(frame, features)
        recording_sets[split] = set(frame["audio_sha256"].astype(str))
        report[split] = {
            "segments": len(frame),
            "recordings": len(examples),
            "recordings_per_class": (
                frame.groupby("target_index")["audio_sha256"]
                .nunique()
                .sort_index()
                .astype(int)
                .tolist()
            ),
        }
    if recording_sets["known_train"] & recording_sets["known_tune"]:
        raise ValueError("G20 train/tune recording leakage detected")
    return datasets["known_train"], datasets["known_tune"], report


def build_head(config: dict[str, Any], classes: int) -> G20ClosedSetHead:
    model = config["model"]
    return G20ClosedSetHead(
        g7_embedding_dim=int(config["data"]["g7_embedding_dim"]),
        projection_hidden_dim=int(model["projection_hidden_dim"]),
        embedding_dim=int(model["embedding_dim"]),
        attention_hidden_dim=int(model["attention_hidden_dim"]),
        attention_heads=int(model["attention_heads"]),
        classes=classes,
        dropout=float(model["dropout"]),
        cosine_scale=float(model["cosine_scale"]),
    )


def preflight(config_path: Path, root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    paths, observed, known_models = _verify_inputs(config, root)
    train_data, _, data_report = _load_data(config, paths, known_models)
    per_class = int(config["train"]["recordings_per_class"])
    if per_class < 2 or min(data_report["known_train"]["recordings_per_class"]) < per_class:
        raise ValueError("G20 balanced SupCon batch cannot be populated")
    batches = balanced_recording_batches(
        train_data.examples,
        recordings_per_class=per_class,
        seed=int(config["train"]["seed"]),
        epoch=0,
    )
    head = build_head(config, len(known_models))
    report = {
        "passed": True,
        "ready_for_training": True,
        "execution_completed": True,
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": file_sha256(config_path),
        "input_sha256": observed,
        "source_sha256": _source_sha256(),
        "known_models": known_models,
        "data": data_report,
        "balanced_batches_per_epoch": len(batches),
        "balanced_batch_recordings": len(batches[0]),
        "trainable_parameter_count": sum(p.numel() for p in head.parameters()),
        "optimization_inputs": ["known_train"],
        "model_selection_inputs": ["known_tune"],
        "unseen_model_inputs": [],
        "known_holdout_read": False,
        "locked_datasets_read": [],
    }
    output = _output(
        root, config["preflight_output_dir"], context="G20 preflight output"
    )
    _atomic_json(report, output / "report.json")
    return report


def _collate(
    dataset: RecordingFeatureDataset, indices: Iterable[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[RecordingExample, ...]]:
    return collate_recordings([dataset[int(index)] for index in indices])


def _segment_cross_entropy(
    logits: torch.Tensor,
    mask: torch.Tensor,
    targets: torch.Tensor,
    *,
    label_smoothing: float,
) -> torch.Tensor:
    expanded_targets = targets[:, None].expand(mask.shape)
    return F.cross_entropy(
        logits[mask].float(),
        expanded_targets[mask],
        label_smoothing=label_smoothing,
    )


@torch.no_grad()
def evaluate(
    head: G20ClosedSetHead,
    dataset: RecordingFeatureDataset,
    *,
    batch_size: int,
    device: torch.device,
    classes: int,
) -> tuple[float, dict[str, Any]]:
    head.eval()
    logits_parts = []
    target_parts = []
    total_loss = 0.0
    entropy_sum = 0.0
    similarity_sum = 0.0
    for start in range(0, len(dataset), batch_size):
        indices = range(start, min(start + batch_size, len(dataset)))
        features, mask, targets, _ = _collate(dataset, indices)
        mask_device = mask.to(device)
        targets_device = targets.to(device)
        output = head(features.to(device), mask_device)
        total_loss += float(
            F.cross_entropy(
                output.logits.float(), targets_device, reduction="sum"
            ).cpu()
        )
        entropy_sum += float(
            normalized_attention_entropy(output.attention, mask_device)
            .mean(dim=1)
            .sum()
            .cpu()
        )
        diversity, _ = attention_regularization(
            output.attention,
            mask_device,
            maximum_normalized_entropy=1.0,
        )
        similarity_sum += float(diversity.cpu()) * len(targets)
        logits_parts.append(output.logits.float().cpu().numpy())
        target_parts.append(targets.numpy())
    logits = np.concatenate(logits_parts)
    targets = np.concatenate(target_parts)
    metrics = classification_metrics(targets, logits, classes)
    metrics["mean_normalized_attention_entropy"] = entropy_sum / len(dataset)
    metrics["mean_attention_head_similarity"] = similarity_sum / len(dataset)
    return total_loss / len(dataset), metrics


def _scheduler_lambda(epoch_index: int, *, warmup: int, epochs: int) -> float:
    epoch = epoch_index + 1
    if warmup > 0 and epoch <= warmup:
        return epoch / warmup
    progress = (epoch - warmup) / max(epochs - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train(config_path: Path, root: Path, *, resume: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    paths, observed, known_models = _verify_inputs(config, root)
    train_data, tune_data, _ = _load_data(config, paths, known_models)
    preflight_path = _output(
        root, config["preflight_output_dir"], context="G20 preflight output"
    ) / "report.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("G20 preflight report is missing")
    preflight_report = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not (
        preflight_report.get("passed") is True
        and preflight_report.get("config_sha256") == file_sha256(config_path)
        and preflight_report.get("input_sha256") == observed
        and preflight_report.get("source_sha256") == _source_sha256()
    ):
        raise ValueError("G20 preflight report is stale")

    seed = int(config["train"]["seed"])
    set_seed(seed)
    device = resolve_device(str(config["train"]["device"]))
    output_dir = _output(root, config["output_dir"], context="G20 training output")
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.csv"
    summary_path = output_dir / "summary.json"
    if not resume and any(
        path.exists() for path in (latest_path, best_path, history_path, summary_path)
    ):
        raise ValueError("G20 artifacts already exist; use --resume if interrupted")
    if resume and summary_path.exists():
        raise ValueError("G20 training is already complete")

    classes = len(known_models)
    head = build_head(config, classes).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    epochs = int(config["train"]["epochs"])
    warmup = int(config["train"]["warmup_epochs"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda index: _scheduler_lambda(
            index, warmup=warmup, epochs=epochs
        ),
    )
    identity = {
        "protocol": PROTOCOL,
        "config_sha256": file_sha256(config_path),
        "input_sha256": observed,
        "source_sha256": _source_sha256(),
    }
    start_epoch = 1
    best_macro_f1 = -1.0
    best_minimum_recall = -1.0
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    if resume:
        if not latest_path.is_file():
            raise FileNotFoundError("G20 latest.pt is missing")
        saved = torch.load(latest_path, map_location="cpu", weights_only=True)
        if saved.get("identity") != identity:
            raise ValueError("G20 resume identity mismatch")
        head.load_state_dict(saved["head_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch = int(saved["epoch"]) + 1
        best_macro_f1 = float(saved["best_macro_f1"])
        best_minimum_recall = float(saved["best_minimum_recall"])
        best_epoch = int(saved["best_epoch"])
        bad_epochs = int(saved["bad_epochs"])
        history = list(saved["history"])
        _restore_rng_state(saved["rng_state"])

    settings = config["train"]
    per_class = int(settings["recordings_per_class"])
    label_smoothing = float(settings["label_smoothing"])
    temperature = float(settings["temperature"])
    maximum_entropy = float(settings["maximum_normalized_attention_entropy"])
    weights = {
        "supcon": float(settings["supcon_weight"]),
        "segment": float(settings["segment_ce_weight"]),
        "diversity": float(settings["attention_diversity_weight"]),
        "focus": float(settings["attention_focus_weight"]),
    }
    patience = int(settings["patience"])
    log_every = int(settings["log_every_steps"])
    batch_size = max(classes * per_class, 32)
    if resume and bad_epochs >= patience:
        start_epoch = epochs + 1

    for epoch in range(start_epoch, epochs + 1):
        head.train()
        totals = {name: 0.0 for name in ("loss", "ce", "supcon", "segment", "diversity", "focus")}
        processed = 0
        batches = balanced_recording_batches(
            train_data.examples,
            recordings_per_class=per_class,
            seed=seed,
            epoch=epoch,
        )
        for step, indices in enumerate(batches, start=1):
            features, mask, targets, _ = _collate(train_data, indices)
            features, mask, targets = (
                features.to(device),
                mask.to(device),
                targets.to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            output = head(features, mask)
            ce = F.cross_entropy(
                output.logits.float(), targets, label_smoothing=label_smoothing
            )
            supcon = supervised_contrastive_loss(
                output.embedding, targets, temperature=temperature
            )
            segment_ce = _segment_cross_entropy(
                output.segment_logits,
                mask,
                targets,
                label_smoothing=label_smoothing,
            )
            diversity, focus = attention_regularization(
                output.attention,
                mask,
                maximum_normalized_entropy=maximum_entropy,
            )
            loss = (
                ce
                + weights["supcon"] * supcon
                + weights["segment"] * segment_ce
                + weights["diversity"] * diversity
                + weights["focus"] * focus
            )
            if not torch.isfinite(loss):
                raise RuntimeError("G20 produced a non-finite loss")
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in head.parameters()
                if parameter.requires_grad
            ]
            if not all(
                gradient is not None and torch.isfinite(gradient).all()
                for gradient in gradients
            ):
                raise RuntimeError("G20 produced missing or non-finite gradients")
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), float(settings["gradient_clip_norm"])
            )
            optimizer.step()
            count = len(targets)
            for name, value in (
                ("loss", loss),
                ("ce", ce),
                ("supcon", supcon),
                ("segment", segment_ce),
                ("diversity", diversity),
                ("focus", focus),
            ):
                totals[name] += float(value.detach().cpu()) * count
            processed += count
            if log_every > 0 and step % log_every == 0:
                print(
                    f"G20 epoch {epoch} step {step}: "
                    f"loss={float(loss):.6f} ce={float(ce):.6f} "
                    f"segment={float(segment_ce):.6f}",
                    flush=True,
                )
        tune_loss, metrics = evaluate(
            head,
            tune_data,
            batch_size=batch_size,
            device=device,
            classes=classes,
        )
        improved = (
            metrics["macro_f1"] > best_macro_f1 + 1.0e-8
            or (
                abs(metrics["macro_f1"] - best_macro_f1) <= 1.0e-8
                and metrics["minimum_recall"] > best_minimum_recall + 1.0e-8
            )
        )
        if improved:
            best_macro_f1 = float(metrics["macro_f1"])
            best_minimum_recall = float(metrics["minimum_recall"])
            best_epoch = epoch
            bad_epochs = 0
            _atomic_torch_save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "protocol": PROTOCOL,
                    "epoch": epoch,
                    "head_state": {
                        name: value.detach().cpu()
                        for name, value in head.state_dict().items()
                    },
                    "architecture": dict(config["model"]),
                    "known_models": known_models,
                    "identity": identity,
                    "tune_recording_metrics": metrics,
                    "known_holdout_read": False,
                    "locked_datasets_read": [],
                },
                best_path,
            )
        else:
            bad_epochs += 1
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": totals["loss"] / processed,
            "train_recording_ce": totals["ce"] / processed,
            "train_supcon": totals["supcon"] / processed,
            "train_segment_ce": totals["segment"] / processed,
            "train_attention_diversity": totals["diversity"] / processed,
            "train_attention_focus": totals["focus"] / processed,
            "tune_loss": tune_loss,
            "tune_accuracy": metrics["accuracy"],
            "tune_macro_f1": metrics["macro_f1"],
            "tune_minimum_recall": metrics["minimum_recall"],
            "tune_attention_entropy": metrics["mean_normalized_attention_entropy"],
            "tune_attention_similarity": metrics["mean_attention_head_similarity"],
            "improved": improved,
            "bad_epochs": bad_epochs,
        }
        history.append(row)
        scheduler.step()
        _write_history(history, history_path)
        _atomic_torch_save(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "epoch": epoch,
                "head_state": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "identity": identity,
                "best_macro_f1": best_macro_f1,
                "best_minimum_recall": best_minimum_recall,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "history": history,
                "rng_state": _capture_rng_state(),
            },
            latest_path,
        )
        print(
            f"G20 epoch {epoch}: accuracy={metrics['accuracy']:.6f} "
            f"macro_f1={metrics['macro_f1']:.6f} "
            f"minimum_recall={metrics['minimum_recall']:.6f} "
            f"attention_entropy={metrics['mean_normalized_attention_entropy']:.6f}",
            flush=True,
        )
        if bad_epochs >= patience:
            break

    saved_best = torch.load(best_path, map_location="cpu", weights_only=True)
    best_metrics = saved_best["tune_recording_metrics"]
    gate_checks = {
        "accuracy": {
            "observed": float(best_metrics["accuracy"]),
            "minimum": float(config["gates"]["minimum_tune_recording_accuracy"]),
        },
        "macro_f1": {
            "observed": float(best_metrics["macro_f1"]),
            "minimum": float(config["gates"]["minimum_tune_recording_macro_f1"]),
        },
        "minimum_recall": {
            "observed": float(best_metrics["minimum_recall"]),
            "minimum": float(config["gates"]["minimum_tune_recording_recall"]),
        },
    }
    for value in gate_checks.values():
        value["passed"] = value["observed"] >= value["minimum"]
    passed = all(value["passed"] for value in gate_checks.values())
    summary = {
        "passed": passed,
        "execution_completed": True,
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": (
            "retain_g20_known_classifier_candidate"
            if passed
            else "revise_g20_known_classifier"
        ),
        "best_epoch": int(saved_best["epoch"]),
        "best_metrics": best_metrics,
        "gate_checks": gate_checks,
        "loss": {
            "recording_cross_entropy": 1.0,
            **{f"{name}_weight": value for name, value in weights.items()},
        },
        "identity": identity,
        "artifacts": {
            "best_checkpoint": best_path.relative_to(root).as_posix(),
            "history": history_path.relative_to(root).as_posix(),
        },
        "optimization_inputs": ["known_train"],
        "model_selection_inputs": ["known_tune"],
        "unseen_model_inputs": [],
        "known_holdout_read": False,
        "locked_datasets_read": [],
    }
    _atomic_json(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g20_closed_set_multiscale.yaml")
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    config_path = args.config
    if not config_path.is_absolute():
        config_path = root / config_path
    if args.preflight_only:
        print(json.dumps(preflight(config_path, root), ensure_ascii=False, indent=2))
    else:
        train(config_path, root, resume=args.resume)


if __name__ == "__main__":
    main()
