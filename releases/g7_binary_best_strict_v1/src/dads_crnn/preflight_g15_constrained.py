from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .audio import peak_normalize, to_fixed_length
from .config import ensure_dirs, load_config
from .data_firewall import file_sha256
from .evaluate_g14_counterfactual import synthesize_pair
from .train import resolve_device, set_seed
from .train_panns import build_model


PROTOCOL = "g15_p1_constrained_adaptation_preflight_v1"
EXPECTED_STUDENT_TRAINABLE = {
    "backbone.fc_audioset.weight",
    "backbone.fc_audioset.bias",
}


def _batchnorm_buffers(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.named_buffers()
        if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }


def select_source_rows(
    frame: pd.DataFrame,
    *,
    count: int,
    seed: int,
    balance_labels: bool,
) -> pd.DataFrame:
    if count <= 0:
        raise ValueError("Source batch count must be positive")
    required = {"source_group", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Source registry lacks columns: {sorted(missing)}")

    def draw(rows: pd.DataFrame, size: int, random_state: int) -> pd.DataFrame:
        if len(rows) < size:
            raise ValueError(f"Source registry has {len(rows)} rows, needs {size}")
        group_sizes = rows.groupby("source_group")["source_group"].transform("size")
        weights = np.power(group_sizes.to_numpy(dtype=np.float64), -0.5)
        return rows.sample(
            n=size,
            replace=False,
            weights=weights,
            random_state=random_state,
        )

    if balance_labels:
        if count % 2:
            raise ValueError("Label-balanced source count must be even")
        if set(frame["label"].astype(int)) != {0, 1}:
            raise ValueError("Label-balanced source must contain labels 0 and 1")
        half = count // 2
        result = pd.concat(
            [
                draw(
                    frame[frame["label"].astype(int).eq(label)],
                    half,
                    seed + label,
                )
                for label in (0, 1)
            ],
            ignore_index=False,
        )
    else:
        result = draw(frame, count, seed)
    return result.sample(frac=1.0, random_state=seed + 100).reset_index(drop=True)


class RegistryAudioLoader:
    def __init__(self, target_samples: int) -> None:
        self.target_samples = int(target_samples)
        self._memmaps: OrderedDict[str, np.ndarray] = OrderedDict()

    def _memmap(self, path: str) -> np.ndarray:
        if path not in self._memmaps:
            value = np.load(path, mmap_mode="r")
            if value.ndim != 2:
                raise ValueError(f"Expected 2-D memmap cache: {path}")
            self._memmaps[path] = value
        return self._memmaps[path]

    def load(self, row: pd.Series) -> np.ndarray:
        cache_format = str(row["cache_format"])
        path = str(row["cache_path"])
        if cache_format == "individual_npy":
            audio = np.load(path)
        elif cache_format == "memmap_npy":
            audio = self._memmap(path)[int(row["cache_index"])]
        else:
            raise ValueError(f"Unsupported registry cache format: {cache_format}")
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        audio = to_fixed_length(audio, self.target_samples, random_crop=False)
        return peak_normalize(audio).astype(np.float32, copy=False)

    def batch(self, rows: pd.DataFrame) -> torch.Tensor:
        values = [self.load(row) for _, row in rows.iterrows()]
        return torch.from_numpy(np.stack(values))


def constrained_losses(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    pair_positive_logits: torch.Tensor,
    pair_background_logits: torch.Tensor,
    *,
    distill_weight: float,
    pair_weight: float,
    pair_margin: float,
    supervised_pos_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    supervised = nn.functional.binary_cross_entropy_with_logits(
        student_logits.float(),
        labels.float(),
        pos_weight=supervised_pos_weight,
    )
    distill = nn.functional.smooth_l1_loss(
        teacher_student_logits.float(), teacher_logits.detach().float()
    )
    pair = torch.relu(
        float(pair_margin)
        - (pair_positive_logits.float() - pair_background_logits.float())
    ).mean()
    total = supervised + float(distill_weight) * distill + float(pair_weight) * pair
    return {
        "supervised": supervised,
        "distill": distill,
        "pair": pair,
        "total": total,
    }


def _verify_registry_inputs(
    config: dict,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    registry = config["registry"]
    audit_path = Path(registry["audit"]["path"])
    audit_hash = file_sha256(audit_path)
    if audit_hash != str(registry["audit"]["sha256"]):
        raise ValueError("G15 P0 audit SHA256 mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("passed") is not True
        or audit.get("audio_payload_read") is not False
        or audit.get("training_started") is not False
    ):
        raise ValueError("G15 P0 audit is not valid")
    frames = {}
    verified = {"p0_audit": audit_hash}
    for name, spec in registry["sources"].items():
        path = Path(spec["path"])
        observed = file_sha256(path)
        if observed != str(spec["sha256"]):
            raise ValueError(f"G15 source-registry SHA256 mismatch: {name}")
        frame = pd.read_csv(path, low_memory=False)
        if len(frame) != int(spec["rows"]):
            raise ValueError(f"G15 source-registry row count changed: {name}")
        if set(frame["training_source"].astype(str)) != {name}:
            raise ValueError(f"G15 source identity changed: {name}")
        if set(frame["split"].astype(str)) != {"train"}:
            raise ValueError(f"Nontraining split entered G15 P1: {name}")
        frames[name] = frame
        verified[name] = observed
    return frames, verified


def preflight(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported G15 P1 protocol")
    frames, verified = _verify_registry_inputs(config)
    seed = int(config["preflight"]["seed"])
    set_seed(seed)
    count = int(config["preflight"]["batch_per_source"])
    selected = {
        "dads_replay": select_source_rows(
            frames["dads_replay"], count=count, seed=seed, balance_labels=True
        ),
        "g9_mechanical_hard_negative": select_source_rows(
            frames["g9_mechanical_hard_negative"],
            count=count,
            seed=seed + 10,
            balance_labels=False,
        ),
        "kielce_uav": select_source_rows(
            frames["kielce_uav"],
            count=count,
            seed=seed + 20,
            balance_labels=False,
        ),
        "tau_background": select_source_rows(
            frames["tau_background"],
            count=count,
            seed=seed + 30,
            balance_labels=False,
        ),
    }
    expected_labels = {
        "dads_replay": {0: count // 2, 1: count // 2},
        "g9_mechanical_hard_negative": {0: count},
        "kielce_uav": {1: count},
        "tau_background": {0: count},
    }
    observed_labels = {
        name: {
            int(key): int(value)
            for key, value in rows["label"].value_counts().sort_index().items()
        }
        for name, rows in selected.items()
    }
    if observed_labels != expected_labels:
        raise ValueError(f"G15 four-source batch composition failed: {observed_labels}")
    if selected["g9_mechanical_hard_negative"]["background_mix_eligible"].any():
        raise ValueError("G9 mechanical negatives entered the background mixer")

    device = resolve_device(str(config["preflight"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("G15 P1 preflight requires the server CUDA GPU")
    loader = RegistryAudioLoader(
        int(config["data"]["sample_rate"] * config["data"]["clip_seconds"])
    )
    waveforms = {name: loader.batch(rows) for name, rows in selected.items()}
    ordered_names = [
        "dads_replay",
        "g9_mechanical_hard_negative",
        "kielce_uav",
        "tau_background",
    ]
    batch = torch.cat([waveforms[name] for name in ordered_names], dim=0)
    labels = torch.cat(
        [
            torch.from_numpy(selected[name]["label"].to_numpy(dtype=np.float32))
            for name in ordered_names
        ]
    )
    total_labels = {
        int(key): int(value)
        for key, value in pd.Series(labels.numpy()).value_counts().sort_index().items()
    }
    expected_total_labels = {
        0: count * 2 + count // 2,
        1: count + count // 2,
    }
    if total_labels != expected_total_labels:
        raise ValueError(
            f"G15 complete batch label composition changed: {total_labels}"
        )
    if config["loss"].get("supervised_pos_weight") != "auto_from_fixed_batch":
        raise ValueError("G15 P1 must derive pos_weight from the fixed batch")
    supervised_pos_weight = torch.tensor(
        total_labels[0] / total_labels[1],
        dtype=torch.float32,
        device=device,
    )

    mixing = config["pairing"]
    paired_negative = []
    paired_positive = []
    pair_diagnostics = []
    for background, uav in zip(
        waveforms["tau_background"].numpy(),
        waveforms["kielce_uav"].numpy(),
        strict=True,
    ):
        negative, positive, diagnostic = synthesize_pair(
            background,
            uav,
            float(mixing["target_snr_db"]),
            epsilon=float(mixing["epsilon"]),
            peak_limit=float(mixing["peak_limit"]),
        )
        paired_negative.append(negative)
        paired_positive.append(positive)
        pair_diagnostics.append(diagnostic)
    paired_positive_tensor = torch.from_numpy(np.stack(paired_positive))
    paired_negative_tensor = torch.from_numpy(np.stack(paired_negative))
    maximum_relative_rms_error = max(
        abs(value["positive_rms"] - value["negative_rms"])
        / value["negative_rms"]
        for value in pair_diagnostics
    )
    maximum_absolute_snr_error = max(
        abs(value["achieved_snr_db"] - float(mixing["target_snr_db"]))
        for value in pair_diagnostics
    )
    maximum_peak = max(value["joint_peak"] for value in pair_diagnostics)
    if maximum_relative_rms_error > float(mixing["rms_relative_tolerance"]):
        raise ValueError("G15 P1 paired RMS control failed")
    if maximum_absolute_snr_error > float(mixing["snr_absolute_tolerance_db"]):
        raise ValueError("G15 P1 paired SNR control failed")
    if maximum_peak > float(mixing["peak_limit"]) + 1e-6:
        raise ValueError("G15 P1 paired peak control failed")

    teacher = build_model(config).to(device)
    student = build_model(config).to(device)
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
        raise ValueError(f"Unexpected G15 student trainable parameters: {sorted(trainable)}")

    amp = bool(config["preflight"]["mixed_precision"])
    batch = batch.to(device)
    labels = labels.to(device)
    distill_count = count * 2
    paired_positive_tensor = paired_positive_tensor.to(device)
    paired_negative_tensor = paired_negative_tensor.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        with torch.amp.autocast(device.type, enabled=amp):
            teacher_initial = teacher(batch[:distill_count])
            student_initial = student(batch[:distill_count])
    maximum_initial_logit_error = float(
        torch.max(torch.abs(teacher_initial.float() - student_initial.float())).item()
    )
    if maximum_initial_logit_error > float(
        config["preflight"]["initial_logit_tolerance"]
    ):
        raise ValueError("G15 student does not exactly initialize from the G7 teacher")

    student.train()
    batchnorm_before = _batchnorm_buffers(student)
    for parameter in student.parameters():
        parameter.grad = None
    with torch.no_grad():
        with torch.amp.autocast(device.type, enabled=amp):
            teacher_logits = teacher(batch[:distill_count])
    with torch.amp.autocast(device.type, enabled=amp):
        student_logits = student(batch)
        pair_logits = student(
            torch.cat((paired_positive_tensor, paired_negative_tensor), dim=0)
        )
        pair_positive_logits = pair_logits[:count]
        pair_background_logits = pair_logits[count:]
        losses = constrained_losses(
            student_logits,
            labels,
            student_logits[:distill_count],
            teacher_logits,
            pair_positive_logits,
            pair_background_logits,
            distill_weight=float(config["loss"]["distill_weight"]),
            pair_weight=float(config["loss"]["pair_weight"]),
            pair_margin=float(config["loss"]["pair_margin"]),
            supervised_pos_weight=supervised_pos_weight,
        )
    losses["total"].backward()
    gradients_finite = all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all().item())
        for parameter in trainable.values()
    )
    frozen_student_gradients_absent = all(
        parameter.grad is None
        for parameter in student.parameters()
        if not parameter.requires_grad
    )
    teacher_gradients_absent = all(
        parameter.grad is None for parameter in teacher.parameters()
    )
    batchnorm_after = _batchnorm_buffers(student)
    batchnorm_buffers_unchanged = all(
        torch.equal(batchnorm_before[name], batchnorm_after[name])
        for name in batchnorm_before
    )
    all_losses_finite = all(
        bool(torch.isfinite(value).item()) for value in losses.values()
    )
    if not (
        gradients_finite
        and frozen_student_gradients_absent
        and teacher_gradients_absent
        and batchnorm_buffers_unchanged
        and all_losses_finite
    ):
        raise ValueError("G15 P1 loss or gradient boundary failed")

    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(device),
        "seed": seed,
        "batch_size": int(len(batch)),
        "batch_source_counts": {name: count for name in ordered_names},
        "batch_source_label_counts": observed_labels,
        "batch_label_counts": total_labels,
        "g9_background_mix_eligible": False,
        "teacher": {
            "checkpoint_sha256": config["model"]["binary_checkpoint_sha256"],
            "frozen": True,
            "gradient_absent": teacher_gradients_absent,
        },
        "student": {
            "initialization": "g7_binary_checkpoint",
            "checkpoint_sha256": config["model"]["binary_checkpoint_sha256"],
            "trainable_scope": config["model"]["trainable_scope"],
            "trainable_parameter_names": sorted(trainable),
            "trainable_parameters": sum(value.numel() for value in trainable.values()),
            "maximum_initial_teacher_logit_error": maximum_initial_logit_error,
            "trainable_gradients_finite": gradients_finite,
            "frozen_gradients_absent": frozen_student_gradients_absent,
            "batchnorm_buffers_unchanged": batchnorm_buffers_unchanged,
        },
        "losses": {name: float(value.item()) for name, value in losses.items()},
        "loss_weights": {
            "supervised": 1.0,
            "supervised_pos_weight": float(supervised_pos_weight.item()),
            "distill": float(config["loss"]["distill_weight"]),
            "pair": float(config["loss"]["pair_weight"]),
            "pair_margin": float(config["loss"]["pair_margin"]),
        },
        "pairing": {
            "pairs": count,
            "target_snr_db": float(mixing["target_snr_db"]),
            "maximum_relative_rms_error": maximum_relative_rms_error,
            "maximum_absolute_snr_error_db": maximum_absolute_snr_error,
            "maximum_peak": maximum_peak,
        },
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "optimizer_step_exercised": False,
        "checkpoint_written": False,
        "formal_training_started": False,
        "audio_payload_scope": "one_fixed_training_batch_only",
        "locked_datasets_read": [],
        "inputs": {"config_sha256": file_sha256(config_path), **verified},
        "ready_for_seed42_feasibility_implementation": True,
    }
    output_dir = Path(config["output_dir"])
    ensure_dirs(output_dir)
    (output_dir / "preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G15 constrained-adaptation P1")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g15_constrained_preflight.yaml"),
    )
    args = parser.parse_args()
    report = preflight(args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
