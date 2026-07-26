from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .config import load_config
from .data_firewall import file_sha256, reject_locked_path
from .train import _build_loaders, _training_criterion, resolve_device, set_seed
from .train_panns import _training_input_identity, build_model


PROTOCOL = "g14_panns_head_only_preflight_v1"
EXPECTED_TRAINABLE_NAMES = {
    "backbone.fc_audioset.weight",
    "backbone.fc_audioset.bias",
}


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    reject_locked_path(path, context="G14 head-only preflight")
    return path.resolve(strict=True)


def _bn_buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.named_buffers()
        if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }


def preflight(
    config_path: Path,
    manifest_path: Path,
    sampler_audit_path: Path,
    root: Path,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    sampler_audit_path = sampler_audit_path.resolve(strict=True)
    config = load_config(config_path)
    sampler_audit = json.loads(sampler_audit_path.read_text(encoding="utf-8"))
    if not (
        sampler_audit.get("passed") is True
        and sampler_audit.get("ready_for_g14_model_preflight") is True
        and sampler_audit.get("inputs", {}).get("segment_manifest_sha256")
        == file_sha256(manifest_path)
    ):
        raise ValueError("Sampler audit is not bound to the requested manifest")
    if config["model"].get("trainable_scope") != "binary_head_only":
        raise ValueError("G14-A preflight requires binary_head_only")
    if config["train"].get("augmentation"):
        raise ValueError("G14-A head-only preflight must not enable augmentation")

    seed = int(config["train"]["seeds"][0])
    set_seed(seed)
    device = resolve_device(str(config["train"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("G14 head-only preflight requires the server CUDA GPU")
    training_inputs = _training_input_identity(config, manifest_path)
    train_loader, tune_loader, _ = _build_loaders(config, manifest_path, seed)
    waveform, target = next(iter(train_loader))
    expected_per_class = int(config["train"]["batch_size"]) // 2
    labels, counts = torch.unique(target, return_counts=True)
    observed = {
        int(label.item()): int(count.item())
        for label, count in zip(labels, counts, strict=True)
    }
    if observed != {0: expected_per_class, 1: expected_per_class}:
        raise ValueError(f"Real training batch is not class balanced: {observed}")

    model = build_model(config).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(trainable) != EXPECTED_TRAINABLE_NAMES:
        raise ValueError(f"Unexpected trainable parameters: {sorted(trainable)}")
    trainable_parameters = sum(parameter.numel() for parameter in trainable.values())
    if trainable_parameters != 2049:
        raise ValueError(f"Unexpected binary-head parameter count: {trainable_parameters}")

    optimizer = torch.optim.Adam(
        trainable.values(), lr=float(config["train"]["learning_rate"])
    )
    amp = bool(config["train"].get("mixed_precision", True))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    criterion = _training_criterion(config, train_loader, device)
    torch.cuda.reset_peak_memory_stats(device)

    model.train()
    bn_before = _bn_buffers(model)
    head_before = {
        name: parameter.detach().clone() for name, parameter in trainable.items()
    }
    waveform = waveform.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device.type, enabled=amp):
        logits_first = model(waveform)
        logits_repeat = model(waveform)
        loss = criterion(logits_first.float(), target)
    if not torch.equal(logits_first, logits_repeat):
        raise ValueError("Frozen G7 representation is stochastic in head-only mode")
    if not torch.isfinite(logits_first).all() or not torch.isfinite(loss):
        raise ValueError("G14 head-only train logits/loss are non-finite")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients_finite = all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all().item())
        for parameter in trainable.values()
    )
    frozen_gradients_absent = all(
        parameter.grad is None
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    if not gradients_finite or not frozen_gradients_absent:
        raise ValueError("G14 head-only gradient boundary failed")
    scaler.step(optimizer)
    scaler.update()
    head_changed = all(
        not torch.equal(head_before[name], parameter.detach())
        for name, parameter in trainable.items()
    )
    if not head_changed:
        raise ValueError("Binary head did not update")
    bn_after = _bn_buffers(model)
    bn_buffers_unchanged = all(
        torch.equal(bn_before[name], bn_after[name]) for name in bn_before
    )
    if not bn_buffers_unchanged:
        raise ValueError("Frozen BatchNorm buffers changed")

    model.eval()
    with torch.no_grad():
        tune_waveform, tune_target = next(iter(tune_loader))
        tune_waveform = tune_waveform.to(device, non_blocking=True)
        tune_target = tune_target.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=amp):
            tune_logits = model(tune_waveform)
            tune_loss = criterion(tune_logits.float(), tune_target)
    if not torch.isfinite(tune_logits).all() or not torch.isfinite(tune_loss):
        raise ValueError("G14 head-only tune logits/loss are non-finite")

    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(device),
        "seed": seed,
        "batch_size": int(config["train"]["batch_size"]),
        "batch_label_counts": observed,
        "initialization": "g7_binary_checkpoint",
        "g7_checkpoint_sha256": config["model"]["binary_checkpoint_sha256"],
        "trainable_scope": config["model"]["trainable_scope"],
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_parameter_names": sorted(trainable),
        "frozen_parameters": total_parameters - trainable_parameters,
        "train_loss": float(loss.item()),
        "tune_loss": float(tune_loss.item()),
        "train_logits_finite": True,
        "tune_logits_finite": True,
        "trainable_gradients_finite": gradients_finite,
        "frozen_gradients_absent": frozen_gradients_absent,
        "binary_head_updated": head_changed,
        "batchnorm_buffers_unchanged": bn_buffers_unchanged,
        "frozen_representation_deterministic": True,
        "automatic_pos_weight": float(criterion.pos_weight.item()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "ready_for_stage1_training": True,
        "optimizer_steps_exercised": 1,
        "checkpoint_written": False,
        "formal_training_started": False,
        "model_inference_scope": "preflight_train_and_tune_batches_only",
        "locked_dataset_audio_read": False,
        "inputs": {
            "config_path": config_path.relative_to(root).as_posix(),
            "config_sha256": file_sha256(config_path),
            "manifest_path": manifest_path.relative_to(root).as_posix(),
            "manifest_sha256": file_sha256(manifest_path),
            "sampler_audit_path": sampler_audit_path.relative_to(root).as_posix(),
            "sampler_audit_sha256": file_sha256(sampler_audit_path),
            "training_inputs": training_inputs,
        },
    }
    output_dir = root / "artifacts/g14_domain_generalization/model_preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "g14_head_only_preflight.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the G14-A head-only GPU preflight.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g14_panns_head_only.yaml")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "artifacts/g14_domain_generalization/segment_cache/g14_segment_manifest.csv"
        ),
    )
    parser.add_argument(
        "--sampler-audit",
        type=Path,
        default=Path(
            "artifacts/g14_domain_generalization/sampler_preflight/sampler_preflight_audit.json"
        ),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    preflight(
        _resolve(root, args.config),
        _resolve(root, args.manifest),
        _resolve(root, args.sampler_audit),
        root,
    )


if __name__ == "__main__":
    main()
