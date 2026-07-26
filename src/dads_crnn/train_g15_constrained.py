from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import ensure_dirs, load_config
from .data_firewall import file_sha256
from .evaluate_g14_counterfactual import synthesize_pair
from .metrics import binary_metrics
from .preflight_g15_constrained import (
    EXPECTED_STUDENT_TRAINABLE,
    RegistryAudioLoader,
    constrained_losses,
    select_source_rows,
)
from .train import resolve_device, set_seed
from .train_panns import build_model


PROTOCOL_V1 = "g15_p3_seed42_constrained_feasibility_v1"
PROTOCOL_V2 = "g15_p3b_seed42_safe_improvement_early_stopping_v2"
SUPPORTED_PROTOCOLS = {PROTOCOL_V1, PROTOCOL_V2}
HISTORY_FIELDS = [
    "epoch",
    "learning_rate",
    "train_total_loss",
    "train_supervised_loss",
    "train_distill_loss",
    "train_pair_loss",
    "dads_val_f1",
    "dads_val_specificity",
    "dads_val_auc",
    "g14_tune_f1",
    "g14_tune_specificity",
    "g14_tune_auc",
    "dads_floor_passed",
    "g14_gain_passed",
    "checkpoint_eligible",
    "safe_score_improved",
    "bad_epochs",
    "optimizer_steps",
    "amp_overflow_skipped_steps",
]

MAX_CONSECUTIVE_AMP_OVERFLOWS = 8


def update_safe_early_stopping(
    *,
    dads_safe: bool,
    g14_score: float,
    best_safe_score: float,
    bad_epochs: int,
    minimum_delta: float = 1.0e-8,
) -> tuple[float, int, bool]:
    """Track development improvement without weakening the promotion gate."""
    improved = bool(
        dads_safe and g14_score > best_safe_score + float(minimum_delta)
    )
    if improved:
        return float(g14_score), 0, True
    return float(best_safe_score), int(bad_epochs) + 1, False


def final_decision(
    *, protocol: str, best_checkpoint_exists: bool, history: list[dict[str, Any]]
) -> str:
    if best_checkpoint_exists:
        return "candidate_ready_for_development_regression"
    if protocol == PROTOCOL_V2 and any(
        bool(row.get("dads_floor_passed")) for row in history
    ):
        return "terminate_no_dual_gate_checkpoint"
    return "terminate_no_dads_safe_checkpoint"


def gradients_are_finite(parameters: dict[str, torch.nn.Parameter]) -> bool:
    return all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all().item())
        for parameter in parameters.values()
    )


def _weighted_indices(
    frame: pd.DataFrame,
    *,
    total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sizes = frame.groupby("source_group")["source_group"].transform("size")
    weights = np.power(sizes.to_numpy(dtype=np.float64), -0.5)
    weights /= weights.sum()
    return rng.choice(len(frame), size=total, replace=True, p=weights)


def build_epoch_plan(
    frames: dict[str, pd.DataFrame],
    *,
    batches: int,
    composition: dict[str, int],
    seed: int,
    epoch: int,
) -> dict[str, np.ndarray]:
    if sum(composition.values()) <= 0:
        raise ValueError("Empty G15 batch composition")
    rng = np.random.default_rng(seed + epoch * 1009)
    plan: dict[str, np.ndarray] = {}
    dads_count = int(composition["dads_replay"])
    if dads_count % 2:
        raise ValueError("DADS batch count must be even")
    dads_parts = []
    for label in (0, 1):
        subset = frames["dads_replay"][
            frames["dads_replay"]["label"].astype(int).eq(label)
        ].reset_index(drop=False)
        local = _weighted_indices(
            subset, total=batches * (dads_count // 2), rng=rng
        )
        dads_parts.append(
            subset.iloc[local]["index"].to_numpy(dtype=np.int64).reshape(batches, -1)
        )
    dads = np.concatenate(dads_parts, axis=1)
    for row in dads:
        rng.shuffle(row)
    plan["dads_replay"] = dads

    g9_count = int(composition["g9_mechanical_hard_negative"])
    classes = sorted(
        frames["g9_mechanical_hard_negative"]["hard_negative_class"].unique()
    )
    if g9_count != len(classes):
        raise ValueError("G15 requires one G9 example from each mechanical class")
    g9_parts = []
    for class_name in classes:
        subset = frames["g9_mechanical_hard_negative"][
            frames["g9_mechanical_hard_negative"]["hard_negative_class"].eq(
                class_name
            )
        ].reset_index(drop=False)
        local = _weighted_indices(subset, total=batches, rng=rng)
        g9_parts.append(subset.iloc[local]["index"].to_numpy(dtype=np.int64)[:, None])
    g9 = np.concatenate(g9_parts, axis=1)
    for row in g9:
        rng.shuffle(row)
    plan["g9_mechanical_hard_negative"] = g9

    for name in ("kielce_uav", "tau_background"):
        count = int(composition[name])
        indices = _weighted_indices(
            frames[name], total=batches * count, rng=rng
        )
        plan[name] = indices.reshape(batches, count)
    return plan


def _load_registry_set(
    specs: dict[str, dict[str, Any]]
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    frames = {}
    hashes = {}
    for name, spec in specs.items():
        path = Path(spec["path"])
        observed = file_sha256(path)
        if observed != str(spec["sha256"]):
            raise ValueError(f"G15 registry SHA256 mismatch: {name}")
        frame = pd.read_csv(path, low_memory=False)
        if len(frame) != int(spec["rows"]):
            raise ValueError(f"G15 registry row count changed: {name}")
        frames[name] = frame
        hashes[name] = observed
    return frames, hashes


def _fixed_validation_rows(
    frames: dict[str, pd.DataFrame],
    config: dict,
    seed: int,
) -> dict[str, pd.DataFrame]:
    count = int(config["validation"]["samples_per_domain"])
    if count % 2:
        raise ValueError("Validation samples_per_domain must be even")
    dads = select_source_rows(
        frames["dads_validation"],
        count=count,
        seed=seed + 500,
        balance_labels=True,
    )
    half = count // 2
    kielce = select_source_rows(
        frames["kielce_tune"],
        count=half,
        seed=seed + 501,
        balance_labels=False,
    )
    tau = select_source_rows(
        frames["tau_tune"],
        count=half,
        seed=seed + 502,
        balance_labels=False,
    )
    return {
        "dads_validation": dads,
        "g14_tune": pd.concat((kielce, tau), ignore_index=True),
    }


def _predict(
    model: torch.nn.Module,
    rows: pd.DataFrame,
    loader: RegistryAudioLoader,
    *,
    device: torch.device,
    batch_size: int,
    amp: bool,
    description: str,
) -> dict[str, Any]:
    model.eval()
    probabilities = []
    for start in range(0, len(rows), batch_size):
        waveforms = loader.batch(rows.iloc[start : start + batch_size]).to(device)
        with torch.no_grad(), torch.amp.autocast(device.type, enabled=amp):
            logits = model(waveforms)
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
        completed = min(start + batch_size, len(rows))
        if completed == len(rows) or completed % (batch_size * 8) == 0:
            print(
                f"G15 {description}: {completed}/{len(rows)}",
                flush=True,
            )
    values = np.concatenate(probabilities)
    labels = rows["label"].to_numpy(dtype=np.int64)
    return binary_metrics(labels, values, threshold=0.5)


def _atomic_torch_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _make_pair_batch(
    tau: torch.Tensor,
    kielce: torch.Tensor,
    pairing: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    negatives = []
    positives = []
    for background, uav in zip(tau.numpy(), kielce.numpy(), strict=True):
        negative, positive, _ = synthesize_pair(
            background,
            uav,
            float(pairing["target_snr_db"]),
            epsilon=float(pairing["epsilon"]),
            peak_limit=float(pairing["peak_limit"]),
        )
        negatives.append(negative)
        positives.append(positive)
    return (
        torch.from_numpy(np.stack(negatives)),
        torch.from_numpy(np.stack(positives)),
    )


def train(config_path: Path, *, preflight_only: bool, resume: bool) -> dict[str, Any]:
    config = load_config(config_path)
    protocol = str(config.get("protocol"))
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError("Unsupported G15 P3/P3b protocol")
    prerequisite = config["prerequisites"]
    prerequisite_hashes = {}
    for name, spec in prerequisite.items():
        observed = file_sha256(Path(spec["path"]))
        if observed != str(spec["sha256"]):
            raise ValueError(f"G15 prerequisite SHA256 mismatch: {name}")
        value = json.loads(Path(spec["path"]).read_text(encoding="utf-8"))
        if value.get("passed") is not True:
            raise ValueError(f"G15 prerequisite did not pass: {name}")
        prerequisite_hashes[name] = observed

    train_frames, train_hashes = _load_registry_set(config["registry"]["train"])
    dev_frames, dev_hashes = _load_registry_set(config["registry"]["development"])
    seed = int(config["train"]["seed"])
    set_seed(seed)
    device = resolve_device(str(config["train"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("G15 P3 requires the server CUDA GPU")
    amp = bool(config["train"]["mixed_precision"])
    composition = {
        name: int(value) for name, value in config["train"]["batch_composition"].items()
    }
    batch_size = sum(composition.values())
    if batch_size != int(config["train"]["batch_size"]):
        raise ValueError("G15 batch composition does not match batch_size")
    if composition["kielce_uav"] != composition["tau_background"]:
        raise ValueError("G15 paired Kielce/TAU batch counts must match")
    negatives = composition["dads_replay"] // 2 + composition[
        "g9_mechanical_hard_negative"
    ] + composition["tau_background"]
    positives = composition["dads_replay"] // 2 + composition["kielce_uav"]
    pos_weight = torch.tensor(negatives / positives, device=device)

    audio_loader = RegistryAudioLoader(
        int(config["data"]["sample_rate"] * config["data"]["clip_seconds"])
    )
    validation_rows = _fixed_validation_rows(dev_frames, config, seed)
    teacher = build_model(config).to(device)
    student = build_model(config).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    trainable = {
        name: parameter
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    }
    if set(trainable) != EXPECTED_STUDENT_TRAINABLE:
        raise ValueError("G15 student trainable boundary changed")
    optimizer = torch.optim.Adam(
        trainable.values(), lr=float(config["train"]["learning_rate"])
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    baseline = {
        name: _predict(
            teacher,
            rows,
            audio_loader,
            device=device,
            batch_size=int(config["validation"]["batch_size"]),
            amp=amp,
            description=f"G7 baseline {name}",
        )
        for name, rows in validation_rows.items()
    }
    run_dir = Path(config["output_dir"]) / "seed_42"
    if preflight_only:
        run_dir = Path(config["output_dir"]) / "preflight"
    ensure_dirs(run_dir)
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_score = float("-inf")
    best_safe_score = float("-inf")
    best_epoch = 0
    bad_epochs = 0
    amp_overflow_skipped_steps_total = 0
    if resume and last_path.is_file() and not preflight_only:
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if str(saved.get("protocol")) != protocol:
            raise ValueError("G15 resume checkpoint protocol mismatch")
        student.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        history = list(saved["history"])
        start_epoch = int(saved["epoch"]) + 1
        best_score = float(saved["best_score"])
        best_safe_score = float(
            saved.get(
                "best_safe_score",
                max(
                    (
                        float(row["g14_tune_f1"])
                        for row in history
                        if bool(row.get("dads_floor_passed"))
                    ),
                    default=float("-inf"),
                ),
            )
        )
        best_epoch = int(saved["best_epoch"])
        bad_epochs = int(saved["bad_epochs"])
        amp_overflow_skipped_steps_total = int(
            saved.get("amp_overflow_skipped_steps_total", 0)
        )

    epochs = 1 if preflight_only else int(config["train"]["epochs"])
    batches = 1 if preflight_only else int(config["train"]["batches_per_epoch"])
    for epoch in range(start_epoch, epochs + 1):
        plan = build_epoch_plan(
            train_frames,
            batches=batches,
            composition=composition,
            seed=seed,
            epoch=epoch,
        )
        totals = {"total": 0.0, "supervised": 0.0, "distill": 0.0, "pair": 0.0}
        optimizer_steps = 0
        amp_overflow_skipped_steps = 0
        consecutive_amp_overflows = 0
        student.train()
        for step in range(batches):
            source_rows = {
                name: train_frames[name].iloc[indices[step]]
                for name, indices in plan.items()
            }
            waveforms = {
                name: audio_loader.batch(rows) for name, rows in source_rows.items()
            }
            ordered = [
                "dads_replay",
                "g9_mechanical_hard_negative",
                "kielce_uav",
                "tau_background",
            ]
            batch = torch.cat([waveforms[name] for name in ordered]).to(device)
            labels = torch.cat(
                [
                    torch.from_numpy(
                        source_rows[name]["label"].to_numpy(dtype=np.float32)
                    )
                    for name in ordered
                ]
            )
            observed_negative = int((labels == 0).sum().item())
            observed_positive = int((labels == 1).sum().item())
            if (observed_negative, observed_positive) != (negatives, positives):
                raise RuntimeError(
                    "G15 runtime batch label composition changed: "
                    f"{observed_negative}/{observed_positive}"
                )
            labels = labels.to(device)
            pair_negative, pair_positive = _make_pair_batch(
                waveforms["tau_background"],
                waveforms["kielce_uav"],
                config["pairing"],
            )
            pair_waveforms = torch.cat((pair_positive, pair_negative)).to(device)
            old_count = composition["dads_replay"] + composition[
                "g9_mechanical_hard_negative"
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.amp.autocast(device.type, enabled=amp):
                teacher_logits = teacher(batch[:old_count])
            with torch.amp.autocast(device.type, enabled=amp):
                student_logits = student(batch)
                pair_logits = student(pair_waveforms)
                losses = constrained_losses(
                    student_logits,
                    labels,
                    student_logits[:old_count],
                    teacher_logits,
                    pair_logits[: composition["kielce_uav"]],
                    pair_logits[composition["kielce_uav"] :],
                    distill_weight=float(config["loss"]["distill_weight"]),
                    pair_weight=float(config["loss"]["pair_weight"]),
                    pair_margin=float(config["loss"]["pair_margin"]),
                    supervised_pos_weight=pos_weight,
                )
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            gradients_finite = gradients_are_finite(trainable)
            if not gradients_finite:
                if preflight_only or not amp:
                    raise RuntimeError("G15 produced non-finite head gradients")
                previous_scale = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                next_scale = float(scaler.get_scale())
                if next_scale >= previous_scale:
                    raise RuntimeError(
                        "G15 non-finite gradients were not handled by AMP scaler"
                    )
                amp_overflow_skipped_steps += 1
                amp_overflow_skipped_steps_total += 1
                consecutive_amp_overflows += 1
                print(
                    f"G15 seed 42 epoch {epoch} step {step + 1}/{batches} "
                    f"AMP overflow: optimizer step skipped, "
                    f"scale={previous_scale:.0f}->{next_scale:.0f}, "
                    f"consecutive={consecutive_amp_overflows}",
                    flush=True,
                )
                if consecutive_amp_overflows > MAX_CONSECUTIVE_AMP_OVERFLOWS:
                    raise RuntimeError(
                        "G15 exceeded consecutive AMP overflow safety limit"
                    )
                continue
            consecutive_amp_overflows = 0
            if preflight_only:
                break
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
            for name in totals:
                totals[name] += float(losses[name].item())
            if (step + 1) % int(config["train"]["log_every_steps"]) == 0:
                print(
                    f"G15 seed 42 epoch {epoch} step {step + 1}/{batches} "
                    f"loss={float(losses['total'].item()):.6f}",
                    flush=True,
                )
        if preflight_only:
            report = {
                "passed": True,
                "protocol": protocol,
                "mode": "training_path_preflight",
                "batch_composition": composition,
                "batch_label_counts": {"0": negatives, "1": positives},
                "supervised_pos_weight": float(pos_weight.item()),
                "losses": {
                    name: float(value.item()) for name, value in losses.items()
                },
                "gradients_finite": True,
                "optimizer_step_exercised": False,
                "checkpoint_written": False,
                "formal_training_started": False,
                "baseline_validation": baseline,
                "inputs": {
                    "config": file_sha256(config_path),
                    "prerequisites": prerequisite_hashes,
                    "train": train_hashes,
                    "development": dev_hashes,
                },
                "locked_datasets_read": [],
            }
            (run_dir / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return report

        if optimizer_steps <= 0:
            raise RuntimeError("G15 epoch completed without an optimizer step")

        metrics = {
            name: _predict(
                student,
                rows,
                audio_loader,
                device=device,
                batch_size=int(config["validation"]["batch_size"]),
                amp=amp,
                description=f"seed42 epoch {epoch} {name}",
            )
            for name, rows in validation_rows.items()
        }
        dads_floor = baseline["dads_validation"]["f1"] - float(
            config["selection"]["maximum_dads_f1_drop"]
        )
        g14_floor = baseline["g14_tune"]["f1"] + float(
            config["selection"]["minimum_g14_tune_f1_gain"]
        )
        dads_safe = metrics["dads_validation"]["f1"] >= dads_floor
        g14_gain = metrics["g14_tune"]["f1"] >= g14_floor
        eligible = dads_safe and g14_gain
        score = metrics["g14_tune"]["f1"] if eligible else float("-inf")
        if protocol == PROTOCOL_V2:
            best_safe_score, next_bad_epochs, safe_score_improved = (
                update_safe_early_stopping(
                    dads_safe=dads_safe,
                    g14_score=float(metrics["g14_tune"]["f1"]),
                    best_safe_score=best_safe_score,
                    bad_epochs=bad_epochs,
                )
            )
        else:
            next_bad_epochs = bad_epochs
            safe_score_improved = False
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{
                f"train_{name}_loss": totals[name] / optimizer_steps
                for name in ("total", "supervised", "distill", "pair")
            },
            "dads_val_f1": metrics["dads_validation"]["f1"],
            "dads_val_specificity": metrics["dads_validation"]["specificity"],
            "dads_val_auc": metrics["dads_validation"]["auc"],
            "g14_tune_f1": metrics["g14_tune"]["f1"],
            "g14_tune_specificity": metrics["g14_tune"]["specificity"],
            "g14_tune_auc": metrics["g14_tune"]["auc"],
            "dads_floor_passed": dads_safe,
            "g14_gain_passed": g14_gain,
            "checkpoint_eligible": eligible,
            "safe_score_improved": safe_score_improved,
            "bad_epochs": next_bad_epochs,
            "optimizer_steps": optimizer_steps,
            "amp_overflow_skipped_steps": amp_overflow_skipped_steps,
        }
        history.append(row)
        improved = eligible and score > best_score + 1e-8
        if improved:
            best_score = score
            best_epoch = epoch
        if protocol == PROTOCOL_V2:
            bad_epochs = next_bad_epochs
        elif improved:
            bad_epochs = 0
        else:
            bad_epochs += 1
        row["bad_epochs"] = bad_epochs
        state = {
            "protocol": protocol,
            "config": config,
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "best_safe_score": best_safe_score,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "amp_overflow_skipped_steps_total": amp_overflow_skipped_steps_total,
            "history": history,
            "baseline_validation": baseline,
            "input_hashes": {
                "config": file_sha256(config_path),
                "prerequisites": prerequisite_hashes,
                "train": train_hashes,
                "development": dev_hashes,
            },
        }
        _atomic_torch_save(state, last_path)
        if improved:
            _atomic_torch_save(state, best_path)
        _write_history(run_dir / "history.csv", history)
        print(
            f"G15 seed 42 epoch {epoch} complete: "
            f"dads_f1={metrics['dads_validation']['f1']:.6f} "
            f"g14_f1={metrics['g14_tune']['f1']:.6f} "
            f"safe_improved={safe_score_improved} "
            f"bad_epochs={bad_epochs} eligible={eligible} "
            f"optimizer_steps={optimizer_steps} "
            f"amp_skips={amp_overflow_skipped_steps}",
            flush=True,
        )
        if bad_epochs >= int(config["train"]["patience"]):
            break

    report = {
        "passed": True,
        "protocol": protocol,
        "seed": seed,
        "decision": final_decision(
            protocol=protocol,
            best_checkpoint_exists=best_path.is_file(),
            history=history,
        ),
        "best_epoch": best_epoch,
        "best_score": best_score if best_path.is_file() else None,
        "epochs_completed": len(history),
        "baseline_validation": baseline,
        "selection_gates": {
            "maximum_dads_f1_drop": float(
                config["selection"]["maximum_dads_f1_drop"]
            ),
            "minimum_g14_tune_f1_gain": float(
                config["selection"]["minimum_g14_tune_f1_gain"]
            ),
        },
        "early_stopping": {
            "policy": (
                "dads_safe_g14_improvement"
                if protocol == PROTOCOL_V2
                else "dual_gate_checkpoint_improvement"
            ),
            "patience": int(config["train"]["patience"]),
            "best_safe_g14_f1": (
                best_safe_score
                if protocol == PROTOCOL_V2 and np.isfinite(best_safe_score)
                else None
            ),
        },
        "amp_safety": {
            "overflow_skipped_steps_total": amp_overflow_skipped_steps_total,
            "maximum_consecutive_overflows": MAX_CONSECUTIVE_AMP_OVERFLOWS,
            "nonfinite_model_parameters_written": False,
        },
        "history": history,
        "formal_training_started": True,
        "locked_datasets_read": [],
        "test_guard_dev_holdout_read": False,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train G15 constrained seed42")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g15_seed42_feasibility.yaml"),
    )
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = train(
        args.config,
        preflight_only=args.mode == "preflight",
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
