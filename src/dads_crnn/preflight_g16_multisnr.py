from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ensure_dirs, load_config
from .data_firewall import file_sha256
from .evaluate_g14_counterfactual import synthesize_pair
from .preflight_g15_constrained import (
    EXPECTED_STUDENT_TRAINABLE,
    RegistryAudioLoader,
    constrained_losses,
)
from .train import resolve_device, set_seed
from .train_g15_constrained import (
    _load_registry_set,
    build_epoch_plan,
    gradients_are_finite,
)
from .train_panns import build_model


PROTOCOL = "g16_p1_multisnr_shared_gain_head_preflight_v1"


def make_multisnr_pair_batch(
    tau: torch.Tensor,
    kielce: torch.Tensor,
    *,
    target_snr_db: list[float],
    epsilon: float,
    peak_limit: float,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    if tau.shape != kielce.shape or tau.ndim != 2:
        raise ValueError("G16 paired source batches must be aligned 2-D tensors")
    snrs = [float(value) for value in target_snr_db]
    if len(snrs) < 2 or len(set(snrs)) != len(snrs) or snrs != sorted(snrs):
        raise ValueError("G16 target SNR grid must be unique and increasing")
    negatives: list[np.ndarray] = []
    positives: list[np.ndarray] = []
    diagnostics: list[dict[str, float]] = []
    for pair_index, (background_value, uav_value) in enumerate(zip(tau, kielce)):
        background = background_value.numpy()
        uav = uav_value.numpy()
        required_gains = [
            synthesize_pair(
                background,
                uav,
                snr,
                epsilon=float(epsilon),
                peak_limit=float(peak_limit),
            )[2]["required_common_gain"]
            for snr in snrs
        ]
        shared_gain = float(min(required_gains))
        pair_negatives = []
        for snr in snrs:
            negative, positive, diagnostic = synthesize_pair(
                background,
                uav,
                snr,
                epsilon=float(epsilon),
                peak_limit=float(peak_limit),
                common_gain_override=shared_gain,
            )
            pair_negatives.append(negative)
            negatives.append(negative)
            positives.append(positive)
            diagnostics.append(
                {
                    **diagnostic,
                    "pair_index": float(pair_index),
                    "target_snr_db": snr,
                    "shared_gain": shared_gain,
                }
            )
        if not all(
            np.array_equal(pair_negatives[0], value)
            for value in pair_negatives[1:]
        ):
            raise RuntimeError("G16 shared background identity failed")
    return (
        torch.from_numpy(np.stack(negatives)),
        torch.from_numpy(np.stack(positives)),
        diagnostics,
    )


def preflight(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported G16 P1 protocol")

    base_spec = config["base_training_config"]
    base_path = Path(base_spec["path"])
    base_hash = file_sha256(base_path)
    if base_hash != str(base_spec["sha256"]):
        raise ValueError("G16 base training config SHA256 mismatch")
    base = load_config(base_path)
    if base.get("protocol") != "g15_p3b_seed42_safe_improvement_early_stopping_v2":
        raise ValueError("G16 base config is not the locked G15 P3b config")

    p4_spec = config["p4_analysis"]
    p4_path = Path(p4_spec["path"])
    p4_hash = file_sha256(p4_path)
    if p4_hash != str(p4_spec["sha256"]):
        raise ValueError("G16 P4 analysis SHA256 mismatch")
    p4 = json.loads(p4_path.read_text(encoding="utf-8"))
    if (
        p4.get("passed") is not True
        or p4.get("mechanism_improvement_supported") is not True
        or p4.get("recommendation")
        != "preregister_multisnr_constrained_head_training"
        or p4.get("candidate_remains_ineligible_for_promotion") is not True
        or p4.get("locked_datasets_read") != []
        or p4.get("dev_holdout_read") is not False
    ):
        raise ValueError("G16 P4 prerequisite is not valid")

    train_frames, train_hashes = _load_registry_set(base["registry"]["train"])
    seed = int(config["preflight"]["seed"])
    set_seed(seed)
    device = resolve_device(str(config["preflight"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("G16 P1 requires the server CUDA GPU")
    composition = {
        name: int(value)
        for name, value in base["train"]["batch_composition"].items()
    }
    plan = build_epoch_plan(
        train_frames,
        batches=1,
        composition=composition,
        seed=seed,
        epoch=1,
    )
    source_rows = {
        name: train_frames[name].iloc[indices[0]]
        for name, indices in plan.items()
    }
    loader = RegistryAudioLoader(
        int(base["data"]["sample_rate"] * base["data"]["clip_seconds"])
    )
    waveforms = {name: loader.batch(rows) for name, rows in source_rows.items()}
    ordered = [
        "dads_replay",
        "g9_mechanical_hard_negative",
        "kielce_uav",
        "tau_background",
    ]
    batch = torch.cat([waveforms[name] for name in ordered])
    labels = torch.cat(
        [
            torch.from_numpy(
                source_rows[name]["label"].to_numpy(dtype=np.float32)
            )
            for name in ordered
        ]
    )
    negatives = int((labels == 0).sum().item())
    positives = int((labels == 1).sum().item())
    pos_weight = torch.tensor(negatives / positives, device=device)

    pairing = config["pairing"]
    pair_negative, pair_positive, diagnostics = make_multisnr_pair_batch(
        waveforms["tau_background"],
        waveforms["kielce_uav"],
        target_snr_db=list(pairing["target_snr_db"]),
        epsilon=float(pairing["epsilon"]),
        peak_limit=float(pairing["peak_limit"]),
    )
    snrs = [float(value) for value in pairing["target_snr_db"]]
    pair_count = composition["kielce_uav"]
    expected_pair_conditions = pair_count * len(snrs)
    if pair_negative.shape[0] != expected_pair_conditions:
        raise RuntimeError("G16 multi-SNR pair count changed")
    maximum_rms_error = max(
        abs(item["positive_rms"] - item["negative_rms"])
        / item["negative_rms"]
        for item in diagnostics
    )
    maximum_snr_error = max(
        abs(item["achieved_snr_db"] - item["target_snr_db"])
        for item in diagnostics
    )
    maximum_peak = max(item["joint_peak"] for item in diagnostics)
    if maximum_rms_error > float(pairing["rms_relative_tolerance"]):
        raise RuntimeError("G16 paired RMS control failed")
    if maximum_snr_error > float(pairing["snr_absolute_tolerance_db"]):
        raise RuntimeError("G16 paired SNR control failed")
    if maximum_peak > float(pairing["peak_limit"]) + 1.0e-6:
        raise RuntimeError("G16 paired peak control failed")

    teacher = build_model(base).to(device)
    student = build_model(base).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    student.eval()
    trainable = {
        name: parameter
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    }
    if set(trainable) != EXPECTED_STUDENT_TRAINABLE:
        raise RuntimeError("G16 trainable parameter boundary changed")
    old_count = composition["dads_replay"] + composition[
        "g9_mechanical_hard_negative"
    ]
    amp = bool(config["preflight"]["mixed_precision"])
    batch = batch.to(device)
    labels = labels.to(device)
    pair_negative = pair_negative.to(device)
    pair_positive = pair_positive.to(device)
    with torch.no_grad(), torch.amp.autocast(device.type, enabled=amp):
        teacher_initial = teacher(batch[:old_count])
        student_initial = student(batch[:old_count])
    maximum_initial_error = float(
        torch.max(torch.abs(teacher_initial.float() - student_initial.float())).item()
    )
    if maximum_initial_error > float(
        config["preflight"]["initial_logit_tolerance"]
    ):
        raise RuntimeError("G16 student initialization differs from G7 teacher")

    optimizer = torch.optim.Adam(
        trainable.values(), lr=float(base["train"]["learning_rate"])
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    student.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad(), torch.amp.autocast(device.type, enabled=amp):
        teacher_logits = teacher(batch[:old_count])
    with torch.amp.autocast(device.type, enabled=amp):
        student_logits = student(batch)
        pair_logits = student(torch.cat((pair_positive, pair_negative), dim=0))
        pair_positive_logits = pair_logits[:expected_pair_conditions]
        pair_background_logits = pair_logits[expected_pair_conditions:]
        losses = constrained_losses(
            student_logits,
            labels,
            student_logits[:old_count],
            teacher_logits,
            pair_positive_logits,
            pair_background_logits,
            distill_weight=float(base["loss"]["distill_weight"]),
            pair_weight=float(base["loss"]["pair_weight"]),
            pair_margin=float(base["loss"]["pair_margin"]),
            supervised_pos_weight=pos_weight,
        )
    scaler.scale(losses["total"]).backward()
    scaler.unscale_(optimizer)
    if not gradients_are_finite(trainable):
        raise RuntimeError("G16 produced non-finite head gradients")
    pair_margin = float(base["loss"]["pair_margin"])
    pair_penalty = torch.relu(
        pair_margin
        - (pair_positive_logits.float() - pair_background_logits.float())
    ).detach().cpu().numpy().reshape(pair_count, len(snrs))

    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "device": torch.cuda.get_device_name(device),
        "seed": seed,
        "single_variable_change": "paired_target_snr_grid",
        "base_training_protocol": base["protocol"],
        "batch_composition": composition,
        "batch_label_counts": {"0": negatives, "1": positives},
        "supervised_pos_weight": float(pos_weight.item()),
        "pairing": {
            "base_pairs": pair_count,
            "target_snr_db": snrs,
            "conditions_per_snr": {
                str(snr): pair_count for snr in snrs
            },
            "positive_conditions": expected_pair_conditions,
            "negative_conditions": expected_pair_conditions,
            "shared_negative_identity": True,
            "maximum_relative_rms_error": maximum_rms_error,
            "maximum_absolute_snr_error_db": maximum_snr_error,
            "maximum_peak": maximum_peak,
            "minimum_shared_gain": min(
                item["shared_gain"] for item in diagnostics
            ),
            "pair_loss_by_snr": {
                str(snr): float(pair_penalty[:, index].mean())
                for index, snr in enumerate(snrs)
            },
        },
        "losses": {
            name: float(value.item()) for name, value in losses.items()
        },
        "teacher_frozen": True,
        "student_trainable_parameters": sum(
            parameter.numel() for parameter in trainable.values()
        ),
        "student_trainable_parameter_names": sorted(trainable),
        "maximum_initial_teacher_logit_error": maximum_initial_error,
        "gradients_finite": True,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "optimizer_step_exercised": False,
        "checkpoint_written": False,
        "formal_training_started": False,
        "inputs": {
            "config": file_sha256(config_path),
            "base_training_config": base_hash,
            "p4_analysis": p4_hash,
            "train": train_hashes,
        },
        "locked_datasets_read": [],
        "dev_holdout_read": False,
        "ready_for_g16_training_implementation": True,
    }
    output = Path(config["output"])
    ensure_dirs(output.parent)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight G16 multi-SNR training")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g16_p1_multisnr_preflight.yaml"),
    )
    args = parser.parse_args()
    print(json.dumps(preflight(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
