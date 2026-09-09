from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import RandomSampler
from tqdm import tqdm

from .config import ensure_dirs, load_config
from .data_firewall import audit_csv_rows
from .metrics import binary_metrics, file_level_metrics
from .panns import PannsCnn14Binary, file_sha256
from .prepare_beats_probe import reject_locked_path
from .train import _build_loaders, _training_criterion, resolve_device, set_seed


HISTORY_FIELDS = [
    "epoch",
    "learning_rate",
    "train_loss",
    "val_loss",
    "val_accuracy",
    "val_f1",
    "val_auc",
]


def _manifest_row_count(path: Path) -> int:
    """Count records and reject locked paths in one streaming CSV scan."""
    try:
        return audit_csv_rows(path)
    except ValueError as error:
        if "Locked final-test value" in str(error):
            raise ValueError(f"Locked final-test row is forbidden in {path}") from error
        raise


def _training_input_identity(config: dict, manifest_path: Path) -> dict:
    """Bind training artifacts to the exact manifest and optional G9 audit."""
    reject_locked_path(manifest_path)
    manifest = manifest_path.resolve(strict=True)
    identity = {
        "manifest_path": manifest.as_posix(),
        "manifest_sha256": file_sha256(manifest),
        "manifest_rows": _manifest_row_count(manifest),
        "g9_audit_path": None,
        "g9_audit_sha256": None,
    }
    input_audits = config.get("data", {}).get("input_audits", [])
    if input_audits:
        if not isinstance(input_audits, list):
            raise ValueError("data.input_audits must be a list")
        bound_audits = []
        for value in input_audits:
            path = Path(str(value))
            reject_locked_path(path)
            path = path.resolve(strict=True)
            try:
                audit = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"Cannot read configured input audit: {path}") from error
            if not isinstance(audit, dict) or audit.get("passed") is not True:
                raise ValueError(f"Configured input audit is not passed: {path}")
            bound_audits.append(
                {
                    "path": path.as_posix(),
                    "sha256": file_sha256(path),
                    "protocol": audit.get("protocol"),
                }
            )
        identity["input_audits"] = bound_audits

    audit_value = config.get("data", {}).get("g9_audit_path")
    if audit_value in (None, ""):
        return identity

    audit_path = Path(str(audit_value))
    reject_locked_path(audit_path)
    audit_path = audit_path.resolve(strict=True)
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read the configured G9 data audit: {audit_path}") from error
    if not isinstance(audit, dict) or audit.get("passed") is not True:
        raise ValueError("G9 data audit must contain passed=true")
    if audit.get("locked_datasets_read") != []:
        raise ValueError("G9 data audit must report locked_datasets_read=[]")

    combined = audit.get("outputs", {}).get("combined_manifest")
    if not isinstance(combined, dict):
        raise ValueError("G9 data audit lacks outputs.combined_manifest")
    audited_path_value = combined.get("path")
    audited_sha256 = str(combined.get("sha256", "")).strip().lower()
    try:
        audited_rows = int(combined.get("rows"))
    except (TypeError, ValueError) as error:
        raise ValueError("G9 combined-manifest audit has an invalid row count") from error
    if not audited_path_value:
        raise ValueError("G9 combined-manifest audit lacks a path")
    audited_path = Path(str(audited_path_value)).resolve(strict=False)
    if audited_path != manifest:
        raise ValueError(
            f"G9 audit is bound to {audited_path}, not the requested manifest {manifest}"
        )
    if audited_sha256 != identity["manifest_sha256"]:
        raise ValueError("G9 training manifest SHA256 does not match the data audit")
    if audited_rows != identity["manifest_rows"]:
        raise ValueError("G9 training manifest row count does not match the data audit")

    identity["g9_audit_path"] = audit_path.as_posix()
    identity["g9_audit_sha256"] = file_sha256(audit_path)
    return identity


def _is_g9(config: dict) -> bool:
    return bool(config.get("data", {}).get("g9_audit_path"))


def _requires_training_input_identity(config: dict) -> bool:
    """Return whether legacy checkpoints without a bound manifest are forbidden."""
    data = config.get("data", {})
    return bool(_is_g9(config) or data.get("require_leakage_fixed_guard", False))


def _amp_init_scale(config: dict) -> float:
    """Read one finite, positive AMP scale for both preflight and training."""
    try:
        scale = float(config["train"].get("amp_init_scale", 65536.0))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("train.amp_init_scale must be a finite positive number") from error
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("train.amp_init_scale must be a finite positive number")
    return scale


def _amp_optimizer_step(
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    *,
    enabled: bool,
) -> tuple[bool, float, float]:
    """Take one scaler-managed step and report a non-finite-gradient skip."""
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    skipped_for_nonfinite_gradients = bool(enabled and scale_after < scale_before)
    return skipped_for_nonfinite_gradients, scale_before, scale_after


def _verify_checkpoint_inputs(
    saved: dict,
    expected: dict,
    *,
    require_identity: bool,
    artifact: str,
) -> None:
    observed = saved.get("training_inputs")
    if observed is None:
        # Preserve historical behavior only for protocols that do not require a
        # cryptographically bound manifest/audit identity.
        if require_identity:
            raise ValueError(f"{artifact} lacks the required training input identity")
        return
    if observed != expected:
        raise ValueError(
            f"{artifact} training manifest or input audit does not match this run"
        )


def build_model(config: dict) -> PannsCnn14Binary:
    model = config["model"]
    if model.get("type") != "panns_cnn14_16k":
        raise ValueError("train_panns only supports model.type=panns_cnn14_16k")
    return PannsCnn14Binary(
        initialization=str(model["initialization"]),
        vendor_dir=str(model["vendor_dir"]),
        checkpoint_path=str(model["checkpoint_path"]),
        checkpoint_sha256=str(model["checkpoint_sha256"]),
        spec_augment=bool(model.get("spec_augment", False)),
        frontend_precision=str(model.get("frontend_precision", "float32")),
        binary_checkpoint_path=model.get("binary_checkpoint_path"),
        binary_checkpoint_sha256=model.get("binary_checkpoint_sha256"),
        trainable_scope=str(model.get("trainable_scope", "full")),
        frequency_masking=model.get("frequency_masking"),
        frequency_mixstyle=model.get("frequency_mixstyle"),
    )


def predict(model: nn.Module, loader, device: torch.device, amp: bool) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    probabilities = []
    labels = []
    with torch.no_grad():
        for waveform, target in loader:
            waveform = waveform.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=amp):
                logits = model(waveform)
                loss = criterion(logits.float(), target)
            total_loss += float(loss.item()) * waveform.size(0)
            probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
            labels.append(target.cpu().numpy())
    return (
        np.concatenate(labels).astype(np.int64),
        np.concatenate(probabilities),
        total_loss / len(loader.dataset),
    )


def _g9_loader_protocol(config: dict, train_loader) -> tuple[set[int], bool]:
    """Validate the single-variable G9 data path before any optimization."""
    natural_sampling = isinstance(train_loader.sampler, RandomSampler) and not bool(
        train_loader.sampler.replacement
    )
    natural_sampling = bool(
        natural_sampling
        and int(train_loader.sampler.num_samples) == len(train_loader.dataset)
        and not config["train"].get("sampling")
    )
    if not _is_g9(config):
        return set(), natural_sampling
    if not natural_sampling:
        raise RuntimeError("G9 requires natural sampling without replacement")

    rows = train_loader.dataset.rows
    if "segment_kind" not in rows.columns:
        raise ValueError("G9 manifest lacks the segment_kind hard-negative marker")
    hard_mask = (
        rows["segment_kind"].fillna("").astype(str).str.strip().str.lower()
        == "hard_negative"
    )
    if not bool(hard_mask.any()):
        raise ValueError("G9 training split contains no hard-negative rows")
    if not bool((rows.loc[hard_mask, "label"] == 0).all()):
        raise ValueError("Every G9 hard-negative row must have label=0")
    if "background_mix_eligible" not in rows.columns:
        raise ValueError("G9 manifest lacks background_mix_eligible")
    if bool(rows.loc[hard_mask, "background_mix_eligible"].any()):
        raise ValueError("G9 hard negatives must have background_mix_eligible=false")
    hard_negative_indices = set(
        rows.index[hard_mask].to_numpy(dtype=np.int64).tolist()
    )
    eligible_backgrounds = set(train_loader.dataset.negative_indices.tolist())
    if hard_negative_indices & eligible_backgrounds:
        raise AssertionError("G9 hard negatives entered the positive background mixer")
    return hard_negative_indices, natural_sampling


def preflight(config: dict, manifest_path: Path, seed: int) -> dict:
    """Exercise the real loader, augmentation and AMP optimizer path without writing a run."""
    configured_amp_init_scale = _amp_init_scale(config)
    training_inputs = (
        _training_input_identity(config, manifest_path) if _is_g9(config) else None
    )
    set_seed(seed)
    device = resolve_device(str(config["train"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("PANNs preflight requires the server CUDA GPU")
    if training_inputs is None:
        training_inputs = _training_input_identity(config, manifest_path)
    train_loader, val_loader, _ = _build_loaders(config, manifest_path, seed)
    is_g9 = _is_g9(config)
    hard_negative_indices, natural_sampling = _g9_loader_protocol(config, train_loader)

    model = build_model(config).to(device)
    amp = bool(config["train"].get("mixed_precision", True))
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("PANNs configuration has no trainable parameters")
    optimizer = torch.optim.Adam(
        trainable_parameters, lr=float(config["train"]["learning_rate"])
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp,
        init_scale=configured_amp_init_scale,
    )
    criterion = _training_criterion(config, train_loader, device)
    if is_g9 and not isinstance(criterion, nn.BCEWithLogitsLoss):
        raise TypeError("G9 first stage requires BCEWithLogitsLoss")
    torch.cuda.reset_peak_memory_stats(device)

    model.train()
    forced_hard_negative = False
    if is_g9:
        # Exercise the real dataset and augmentation path with a natural batch,
        # but guarantee coverage of the new data path instead of relying on a
        # probabilistic first-batch draw.
        batch_indices = [int(value) for value in next(iter(train_loader.batch_sampler))]
        hard_negative_index = min(hard_negative_indices)
        if not any(value in hard_negative_indices for value in batch_indices):
            batch_indices[0] = hard_negative_index
            forced_hard_negative = True
        batch = [train_loader.dataset[index] for index in batch_indices]
        waveform = torch.stack([item[0] for item in batch])
        target = torch.stack([item[1] for item in batch])
        hard_negative_samples_in_batch = sum(
            index in hard_negative_indices for index in batch_indices
        )
        if hard_negative_samples_in_batch < 1:
            raise AssertionError("G9 preflight did not exercise a hard-negative sample")
    else:
        waveform, target = next(iter(train_loader))
        hard_negative_samples_in_batch = 0
    waveform = waveform.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device.type, enabled=amp):
        logits = model(waveform)
        train_loss = criterion(logits.float(), target)
    scaler.scale(train_loss).backward()
    scaler.unscale_(optimizer)
    nonfinite_gradient_parameters = []
    nonfinite_gradient_values = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        finite = torch.isfinite(parameter.grad)
        if bool(finite.all().item()):
            continue
        nonfinite_gradient_parameters.append(name)
        nonfinite_gradient_values[name] = int((~finite).sum().item())
    gradients_finite = not nonfinite_gradient_parameters
    (
        optimizer_step_skipped,
        initial_grad_scale,
        updated_grad_scale,
    ) = _amp_optimizer_step(scaler, optimizer, enabled=amp)

    model.eval()
    with torch.no_grad():
        val_waveform, val_target = next(iter(val_loader))
        val_waveform = val_waveform.to(device, non_blocking=True)
        val_target = val_target.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=amp):
            val_logits = model(val_waveform)
            val_loss = nn.functional.binary_cross_entropy_with_logits(
                val_logits.float(), val_target
            )
    result = {
        "passed": bool(
            torch.isfinite(train_loss).item()
            and torch.isfinite(val_loss).item()
            and torch.isfinite(logits).all().item()
            and torch.isfinite(val_logits).all().item()
            and gradients_finite
            and not optimizer_step_skipped
        ),
        "initialization": config["model"]["initialization"],
        "frontend_precision": config["model"].get("frontend_precision", "float32"),
        "seed": seed,
        "batch_size": int(waveform.size(0)),
        "train_loss": float(train_loss.item()),
        "val_loss": float(val_loss.item()),
        "train_logits_finite": bool(torch.isfinite(logits).all().item()),
        "val_logits_finite": bool(torch.isfinite(val_logits).all().item()),
        "train_gradients_finite": gradients_finite,
        "initial_grad_scale": initial_grad_scale,
        "updated_grad_scale": updated_grad_scale,
        "optimizer_step_skipped_for_nonfinite_gradients": optimizer_step_skipped,
        "nonfinite_gradient_parameters": nonfinite_gradient_parameters,
        "nonfinite_gradient_values": nonfinite_gradient_values,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "official_pretraining_sha256": config["model"]["checkpoint_sha256"],
        "training_inputs": training_inputs,
        "natural_sampling_without_replacement": natural_sampling,
        "hard_negative_rows": len(hard_negative_indices),
        "hard_negative_samples_in_batch": hard_negative_samples_in_batch,
        "forced_hard_negative_into_batch": forced_hard_negative,
        "locked_datasets_read": [],
    }
    if not result["passed"]:
        raise RuntimeError(f"PANNs preflight produced non-finite values: {result}")
    return result


def _capture_rng_state(train_loader) -> dict:
    numpy_state = np.random.get_state()
    dataset = train_loader.dataset
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "loader_generator": train_loader.generator.get_state(),
        "dataset_rng": dataset.rng.bit_generator.state,
        "augmenter_rng": (
            dataset.augmenter.rng.bit_generator.state if dataset.augmenter is not None else None
        ),
    }


def _restore_rng_state(state: dict, train_loader) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    # Resume payloads can be loaded with a CUDA map location for model state,
    # but PyTorch's CPU and DataLoader generators strictly require CPU byte
    # tensors. Force RNG tensors back to CPU independently of load policy.
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])
    train_loader.generator.set_state(state["loader_generator"].cpu())
    dataset = train_loader.dataset
    dataset.rng.bit_generator.state = state["dataset_rng"]
    if dataset.augmenter is not None:
        if state["augmenter_rng"] is None:
            raise ValueError("Resume checkpoint lacks the configured augmenter RNG state")
        dataset.augmenter.rng.bit_generator.state = state["augmenter_rng"]


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_history(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def train_one_seed(
    config: dict,
    manifest_path: Path,
    seed: int,
    *,
    resume: bool = False,
    training_inputs: dict | None = None,
) -> dict:
    configured_amp_init_scale = _amp_init_scale(config)
    if training_inputs is None and _is_g9(config):
        training_inputs = _training_input_identity(config, manifest_path)
    set_seed(seed)
    device = resolve_device(str(config["train"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("PANNs training requires the server CUDA GPU")
    if training_inputs is None:
        training_inputs = _training_input_identity(config, manifest_path)
    run_dir = Path(config["output_dir"]) / f"seed_{seed}"
    ensure_dirs(run_dir)
    train_loader, val_loader, test_loader = _build_loaders(config, manifest_path, seed)
    hard_negative_indices, _ = _g9_loader_protocol(config, train_loader)
    model = build_model(config).to(device)
    amp = bool(config["train"].get("mixed_precision", True))
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("PANNs configuration has no trainable parameters")
    optimizer = torch.optim.Adam(
        trainable_parameters, lr=float(config["train"]["learning_rate"])
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(config["train"].get("lr_decay", 1.0))
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp,
        init_scale=configured_amp_init_scale,
    )
    criterion = _training_criterion(config, train_loader, device)
    if _is_g9(config) and not isinstance(criterion, nn.BCEWithLogitsLoss):
        raise TypeError("G9 first stage requires BCEWithLogitsLoss")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"{'G9' if _is_g9(config) else 'G7'} {config['model']['initialization']} "
        f"parameters={parameter_count:,}, "
        f"batch={config['train']['batch_size']}, amp={amp}, "
        f"hard_negatives={len(hard_negative_indices)}"
    )

    best_f1 = -1.0
    best_epoch = 0
    bad_epochs = 0
    start_epoch = 1
    elapsed_offset = 0.0
    amp_nonfinite_gradient_skip_steps = 0
    amp_skip_count_complete = True
    history_rows: list[dict] = []
    history_path = run_dir / "history.csv"
    last_path = run_dir / "last.pt"
    if resume and last_path.exists():
        # Load the compound resume payload on CPU so RNG states retain their
        # required device. load_state_dict migrates model and Adam tensors to
        # their corresponding CUDA parameters.
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        if saved.get("config") != config or int(saved.get("seed", -1)) != seed:
            raise ValueError("Resume checkpoint config or seed does not match this run")
        _verify_checkpoint_inputs(
            saved,
            training_inputs,
            require_identity=_requires_training_input_identity(config),
            artifact="Resume checkpoint",
        )
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        best_f1 = float(saved["best_f1"])
        best_epoch = int(saved["best_epoch"])
        bad_epochs = int(saved["bad_epochs"])
        start_epoch = int(saved["next_epoch"])
        elapsed_offset = float(saved["elapsed_seconds"])
        if "amp_nonfinite_gradient_skip_steps" in saved:
            amp_nonfinite_gradient_skip_steps = int(
                saved["amp_nonfinite_gradient_skip_steps"]
            )
            amp_skip_count_complete = bool(
                saved.get("amp_skip_count_complete", True)
            )
        else:
            # A pre-counter checkpoint can still be resumed when its exact
            # training input identity matches, but its earlier AMP skips are
            # unknowable and must not be reported as a complete zero.
            amp_skip_count_complete = False
        history_rows = list(saved["history"])
        _restore_rng_state(saved["rng_state"], train_loader)
        _write_history(history_path, history_rows)
        print(
            f"Resuming {'G9' if _is_g9(config) else 'G7'} seed {seed} at epoch {start_epoch}; "
            f"best_epoch={best_epoch}, bad_epochs={bad_epochs}"
        )
    start_time = time.time()
    if not history_rows:
        _write_history(history_path, history_rows)
    experiment = "G9" if _is_g9(config) else "G7"
    for epoch in range(start_epoch, int(config["train"]["epochs"]) + 1):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch - 1)
        model.train()
        train_loss = 0.0
        train_examples = 0
        total_batches = len(train_loader)
        interactive_progress = bool(sys.stderr.isatty())
        progress = tqdm(
            train_loader,
            desc=f"{experiment} seed {seed} epoch {epoch}",
            leave=False,
            disable=not interactive_progress,
        )
        if not interactive_progress:
            print(
                f"{experiment} seed {seed} epoch {epoch} started: "
                f"total_steps={total_batches}",
                flush=True,
            )
        for batch_index, (waveform, target) in enumerate(progress, start=1):
            waveform = waveform.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=amp):
                logits = model(waveform)
                loss = criterion(logits.float(), target)
            scaler.scale(loss).backward()
            skipped_for_nonfinite_gradients, _, _ = _amp_optimizer_step(
                scaler, optimizer, enabled=amp
            )
            if skipped_for_nonfinite_gradients:
                amp_nonfinite_gradient_skip_steps += 1
            train_loss += float(loss.item()) * waveform.size(0)
            train_examples += int(waveform.size(0))
            if interactive_progress:
                progress.set_postfix(loss=float(loss.item()))
            elif batch_index % 100 == 0 or batch_index == total_batches:
                print(
                    f"{experiment} seed {seed} epoch {epoch} "
                    f"step {batch_index}/{total_batches} loss={float(loss.item()):.6f}",
                    flush=True,
                )

        val_true, val_probability, val_loss = predict(model, val_loader, device, amp)
        val_metrics = binary_metrics(val_true, val_probability, threshold=0.5)
        train_loss /= train_examples
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
        }
        history_rows.append(row)
        if not interactive_progress:
            print(
                f"{experiment} seed {seed} epoch {epoch} complete: "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"val_f1={val_metrics['f1']:.6f} val_auc={val_metrics['auc']:.6f}",
                flush=True,
            )
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            bad_epochs = 0
            _atomic_torch_save(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "config": config,
                    "training_inputs": training_inputs,
                    "amp_grad_scale": float(scaler.get_scale()),
                    "amp_nonfinite_gradient_skip_steps": (
                        amp_nonfinite_gradient_skip_steps
                    ),
                    "amp_skip_count_complete": amp_skip_count_complete,
                },
                run_dir / "best.pt",
            )
        else:
            bad_epochs += 1
        scheduler.step()
        elapsed_so_far = elapsed_offset + time.time() - start_time
        _atomic_torch_save(
            {
                "version": 2,
                "seed": seed,
                "config": config,
                "training_inputs": training_inputs,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "amp_nonfinite_gradient_skip_steps": (
                    amp_nonfinite_gradient_skip_steps
                ),
                "amp_skip_count_complete": amp_skip_count_complete,
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "next_epoch": epoch + 1,
                "elapsed_seconds": elapsed_so_far,
                "history": history_rows,
                "rng_state": _capture_rng_state(train_loader),
            },
            last_path,
        )
        _write_history(history_path, history_rows)
        if bad_epochs >= int(config["train"]["patience"]):
            break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    _verify_checkpoint_inputs(
        checkpoint,
        training_inputs,
        require_identity=_requires_training_input_identity(config),
        artifact="Best checkpoint",
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    val_true, val_probability, val_loss = predict(model, val_loader, device, amp)
    test_true, test_probability, test_loss = predict(model, test_loader, device, amp)
    thresholds = [float(value) for value in config["eval"]["thresholds"]]
    val_metrics = [binary_metrics(val_true, val_probability, value) for value in thresholds]
    test_metrics = [binary_metrics(test_true, test_probability, value) for value in thresholds]
    val_file_metrics = {
        aggregation: file_level_metrics(val_loader.dataset.rows, val_probability, thresholds, aggregation=aggregation)
        for aggregation in ("mean", "max")
    }
    test_file_metrics = {
        aggregation: file_level_metrics(test_loader.dataset.rows, test_probability, thresholds, aggregation=aggregation)
        for aggregation in ("mean", "max")
    }
    elapsed = elapsed_offset + time.time() - start_time
    result = {
        "seed": seed,
        "device": str(device),
        "model_type": "panns_cnn14_16k",
        "initialization": config["model"]["initialization"],
        "feature_type": "panns_log_mel",
        "parameter_count": parameter_count,
        "mixed_precision": amp,
        "amp_init_scale": configured_amp_init_scale,
        "amp_final_scale": float(scaler.get_scale()),
        "amp_nonfinite_gradient_skip_steps": amp_nonfinite_gradient_skip_steps,
        "amp_skip_count_complete": amp_skip_count_complete,
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "val_loss": val_loss,
        "val_threshold_metrics": val_metrics,
        "val_file_metrics": val_file_metrics,
        "test_loss": test_loss,
        "threshold_metrics": test_metrics,
        "test_file_metrics": test_file_metrics,
        "elapsed_seconds": elapsed,
        "checkpoint_sha256": file_sha256(run_dir / "best.pt"),
        "official_pretraining_sha256": config["model"]["checkpoint_sha256"],
        "training_inputs": training_inputs,
        "locked_datasets_read": [],
    }
    np.save(run_dir / "val_probabilities.npy", val_probability)
    np.save(run_dir / "val_labels.npy", val_true)
    np.save(run_dir / "test_probabilities.npy", test_probability)
    np.save(run_dir / "test_labels.npy", test_true)
    result["prediction_sha256"] = {
        name: file_sha256(run_dir / f"{name}.npy")
        for name in ("val_probabilities", "val_labels", "test_probabilities", "test_labels")
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"{'G9' if _is_g9(config) else 'G7'} {config['model']['initialization']} "
        f"seed {seed} complete: "
        f"best_epoch={best_epoch}, best_val_f1={best_f1:.5f}, minutes={elapsed / 60:.1f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train G7 PANNs Cnn14_16k")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run one real train step and one validation batch without writing run artifacts",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an atomically written last.pt epoch checkpoint when present",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    seeds = args.seeds or [int(value) for value in config["train"]["seeds"]]
    if args.preflight_only:
        if len(seeds) != 1:
            raise ValueError("--preflight-only accepts exactly one seed")
        print(json.dumps(preflight(config, args.manifest, seeds[0]), indent=2))
        return
    training_inputs = _training_input_identity(config, args.manifest)
    for seed in seeds:
        run_dir = Path(config["output_dir"]) / f"seed_{seed}"
        if run_dir.exists() and any(run_dir.iterdir()):
            if not args.resume:
                raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
            if not (run_dir / "last.pt").is_file():
                raise FileNotFoundError(f"Cannot resume without an epoch checkpoint: {run_dir / 'last.pt'}")
    results = [
        train_one_seed(
            config,
            args.manifest,
            seed,
            resume=args.resume,
            training_inputs=training_inputs,
        )
        for seed in seeds
    ]
    summary = Path(config["output_dir"]) / "summary.json"
    ensure_dirs(summary.parent)
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
