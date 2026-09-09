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
from torch.utils.data import DataLoader

from .config import ensure_dirs, load_config
from .data_firewall import file_sha256
from .evaluate_g14_counterfactual import CounterfactualPairDataset
from .preflight_g15_constrained import (
    EXPECTED_STUDENT_TRAINABLE,
    RegistryAudioLoader,
    constrained_losses,
)
from .preflight_g16_multisnr import make_multisnr_pair_batch
from .train import resolve_device, set_seed
from .train_g15_constrained import (
    MAX_CONSECUTIVE_AMP_OVERFLOWS,
    _fixed_validation_rows,
    _load_registry_set,
    _predict,
    build_epoch_plan,
    gradients_are_finite,
    update_safe_early_stopping,
)
from .train_panns import build_model


PROTOCOL = "g16_p2_seed42_multisnr_constrained_head_v1"
HISTORY_FIELDS = [
    "epoch",
    "train_total_loss",
    "train_supervised_loss",
    "train_distill_loss",
    "train_pair_loss",
    "dads_f1",
    "raw_g14_f1",
    "minimum_snr_lift_gain",
    "minimum_snr_ordering_gain",
    "maximum_snr_background_fpr_increase",
    "monotonicity_gain",
    "minimum_low_snr_tpr_gain",
    "dads_safe",
    "raw_g14_safe",
    "checkpoint_eligible",
    "safe_score_improved",
    "bad_epochs",
    "optimizer_steps",
    "amp_overflow_skipped_steps",
]


def counterfactual_metrics(
    model: torch.nn.Module,
    dataset: CounterfactualPairDataset,
    *,
    device: torch.device,
    batch_size: int,
    amp: bool,
    threshold: float,
    description: str,
) -> dict[str, Any]:
    model.eval()
    negative_probability = np.empty(len(dataset), dtype=np.float32)
    positive_probability = np.empty(len(dataset), dtype=np.float32)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for negative, positive, indices in loader:
            waveforms = torch.cat((negative, positive), dim=0).to(device)
            with torch.amp.autocast(device.type, enabled=amp):
                # Match the frozen G14-D V2/P4 inference semantics exactly.
                probabilities = torch.sigmoid(model(waveforms)).cpu().numpy()
            count = len(indices)
            index_values = indices.numpy()
            negative_probability[index_values] = probabilities[:count]
            positive_probability[index_values] = probabilities[count:]
            completed = min(int(index_values[-1]) + 1, len(dataset))
            if completed == len(dataset) or completed % 1024 == 0:
                print(f"G16 {description}: {completed}/{len(dataset)}", flush=True)
    rows = dataset.rows
    by_snr = {}
    for snr, group in rows.groupby("target_snr_db", sort=True):
        indices = group.index.to_numpy(dtype=np.int64)
        negative = negative_probability[indices]
        positive = positive_probability[indices]
        lift = positive - negative
        by_snr[str(float(snr))] = {
            "mean_lift": float(lift.mean()),
            "ordering_accuracy": float((lift > 0).mean()),
            "negative_fpr": float((negative >= threshold).mean()),
            "positive_tpr": float((positive >= threshold).mean()),
        }
    pivot = pd.DataFrame(
        {
            "base_pair_id": rows["base_pair_id"],
            "target_snr_db": rows["target_snr_db"],
            "positive_probability": positive_probability,
        }
    ).pivot(
        index="base_pair_id",
        columns="target_snr_db",
        values="positive_probability",
    ).sort_index(axis=1)
    monotonic = np.all(np.diff(pivot.to_numpy(), axis=1) >= -1.0e-6, axis=1)
    return {
        "by_snr": by_snr,
        "monotonicity": float(monotonic.mean()),
    }


def selection_status(
    *,
    baseline_raw: dict[str, dict[str, float]],
    candidate_raw: dict[str, dict[str, float]],
    baseline_pair: dict[str, Any],
    candidate_pair: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    snrs = sorted(baseline_pair["by_snr"], key=float)
    if snrs != sorted(candidate_pair["by_snr"], key=float):
        raise ValueError("G16 candidate SNR grid differs from baseline")
    deltas = {}
    for snr in snrs:
        baseline = baseline_pair["by_snr"][snr]
        candidate = candidate_pair["by_snr"][snr]
        deltas[snr] = {
            name: float(candidate[name] - baseline[name])
            for name in (
                "mean_lift",
                "ordering_accuracy",
                "negative_fpr",
                "positive_tpr",
            )
        }
    minimum_lift = min(item["mean_lift"] for item in deltas.values())
    minimum_ordering = min(
        item["ordering_accuracy"] for item in deltas.values()
    )
    maximum_background_fpr = max(
        item["negative_fpr"] for item in deltas.values()
    )
    low_snrs = [str(float(value)) for value in gates["low_snr_db"]]
    minimum_low_tpr = min(deltas[snr]["positive_tpr"] for snr in low_snrs)
    monotonicity_gain = float(
        candidate_pair["monotonicity"] - baseline_pair["monotonicity"]
    )
    checks = {
        "dads_safe": candidate_raw["dads_validation"]["f1"]
        >= baseline_raw["dads_validation"]["f1"]
        - float(gates["maximum_dads_f1_drop"]),
        "raw_g14_safe": candidate_raw["g14_tune"]["f1"]
        >= baseline_raw["g14_tune"]["f1"]
        - float(gates["maximum_raw_g14_f1_drop"]),
        "each_snr_lift_gain": minimum_lift
        >= float(gates["minimum_each_snr_mean_lift_gain"]),
        "each_snr_ordering_gain": minimum_ordering
        >= float(gates["minimum_each_snr_ordering_gain"]),
        "background_fpr_safe": maximum_background_fpr
        <= float(gates["maximum_each_snr_background_fpr_increase"]),
        "monotonicity_gain": monotonicity_gain
        >= float(gates["minimum_monotonicity_gain"]),
        "low_snr_tpr_safe": minimum_low_tpr
        >= -float(gates["maximum_low_snr_tpr_drop"]),
    }
    return {
        "checks": checks,
        "eligible": all(checks.values()),
        "score": minimum_lift,
        "deltas_by_snr": deltas,
        "minimum_snr_lift_gain": minimum_lift,
        "minimum_snr_ordering_gain": minimum_ordering,
        "maximum_snr_background_fpr_increase": maximum_background_fpr,
        "monotonicity_gain": monotonicity_gain,
        "minimum_low_snr_tpr_gain": minimum_low_tpr,
    }


def _atomic_save(value: dict[str, Any], path: Path) -> None:
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


def _verify_inputs(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    verified = {}
    for name, spec in config["prerequisites"].items():
        path = Path(spec["path"])
        observed = file_sha256(path)
        if observed != str(spec["sha256"]):
            raise ValueError(f"G16 prerequisite SHA256 mismatch: {name}")
        verified[name] = observed
    p1 = json.loads(
        Path(config["prerequisites"]["p1_preflight"]["path"]).read_text()
    )
    if (
        p1.get("passed") is not True
        or p1.get("ready_for_g16_training_implementation") is not True
        or p1.get("formal_training_started") is not False
        or p1.get("locked_datasets_read") != []
    ):
        raise ValueError("G16 P1 prerequisite is not valid")
    base = load_config(Path(config["prerequisites"]["base_training_config"]["path"]))
    tune = config["counterfactual_tune"]
    manifest = Path(tune["manifest"])
    observed = file_sha256(manifest)
    if observed != str(tune["sha256"]):
        raise ValueError("G16 counterfactual tune SHA256 mismatch")
    frame = pd.read_csv(manifest, low_memory=False)
    if (
        len(frame) != int(tune["rows"])
        or frame["base_pair_id"].nunique() != int(tune["base_pairs"])
        or set(frame["split"].astype(str)) != {"tune"}
    ):
        raise ValueError("G16 counterfactual tune identity changed")
    verified["counterfactual_tune"] = observed
    return base, verified


def run(config_path: Path, *, preflight_only: bool, resume: bool) -> dict[str, Any]:
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported G16 P2 protocol")
    base, verified = _verify_inputs(config)
    train_frames, train_hashes = _load_registry_set(base["registry"]["train"])
    dev_frames, dev_hashes = _load_registry_set(base["registry"]["development"])
    seed = int(base["train"]["seed"])
    set_seed(seed)
    device = resolve_device(str(base["train"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("G16 P2 requires the server CUDA GPU")
    amp = bool(base["train"]["mixed_precision"])
    composition = {
        name: int(value)
        for name, value in base["train"]["batch_composition"].items()
    }
    negatives = composition["dads_replay"] // 2 + composition[
        "g9_mechanical_hard_negative"
    ] + composition["tau_background"]
    positives = composition["dads_replay"] // 2 + composition["kielce_uav"]
    pos_weight = torch.tensor(negatives / positives, device=device)
    audio_loader = RegistryAudioLoader(
        int(base["data"]["sample_rate"] * base["data"]["clip_seconds"])
    )
    raw_validation = _fixed_validation_rows(dev_frames, base, seed)
    pairing = config["pairing"]
    pair_dataset = CounterfactualPairDataset(
        Path(config["counterfactual_tune"]["manifest"]),
        epsilon=float(pairing["epsilon"]),
        peak_limit=float(pairing["peak_limit"]),
        share_gain_across_snr=True,
    )

    teacher = build_model(base).to(device)
    student = build_model(base).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    trainable = {
        name: parameter
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    }
    if set(trainable) != EXPECTED_STUDENT_TRAINABLE:
        raise RuntimeError("G16 trainable parameter boundary changed")
    optimizer = torch.optim.Adam(
        trainable.values(), lr=float(base["train"]["learning_rate"])
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    baseline_raw = {
        name: _predict(
            teacher,
            rows,
            audio_loader,
            device=device,
            batch_size=int(base["validation"]["batch_size"]),
            amp=amp,
            description=f"G16 G7 baseline {name}",
        )
        for name, rows in raw_validation.items()
    }
    baseline_pair = counterfactual_metrics(
        teacher,
        pair_dataset,
        device=device,
        batch_size=int(config["counterfactual_validation"]["batch_size_pairs"]),
        amp=amp,
        threshold=float(config["counterfactual_validation"]["threshold"]),
        description="G7 counterfactual tune",
    )

    run_dir = Path(config["output_dir"]) / (
        "preflight" if preflight_only else "seed_42"
    )
    ensure_dirs(run_dir)
    if preflight_only:
        report = {
            "passed": True,
            "protocol": PROTOCOL,
            "mode": "selection_preflight",
            "baseline_raw": baseline_raw,
            "baseline_counterfactual": baseline_pair,
            "selection_gates": config["selection"],
            "student_initialization": "g7_seed42",
            "student_trainable_parameters": sum(
                parameter.numel() for parameter in trainable.values()
            ),
            "optimizer_step_exercised": False,
            "checkpoint_written": False,
            "formal_training_started": False,
            "inputs": {
                "config": file_sha256(config_path),
                "prerequisites": verified,
                "train": train_hashes,
                "development": dev_hashes,
            },
            "locked_datasets_read": [],
            "dev_holdout_read": False,
            "ready_for_formal_seed42": True,
        }
        (run_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_score = float("-inf")
    best_epoch = 0
    best_safe_score = float("-inf")
    bad_epochs = 0
    amp_skips_total = 0
    if resume and last_path.is_file():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if saved.get("protocol") != PROTOCOL:
            raise ValueError("G16 resume protocol mismatch")
        student.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        history = list(saved["history"])
        start_epoch = int(saved["epoch"]) + 1
        best_score = float(saved["best_score"])
        best_epoch = int(saved["best_epoch"])
        best_safe_score = float(saved["best_safe_score"])
        bad_epochs = int(saved["bad_epochs"])
        amp_skips_total = int(saved.get("amp_skips_total", 0))

    epochs = int(base["train"]["epochs"])
    batches = int(base["train"]["batches_per_epoch"])
    snrs = [float(value) for value in pairing["target_snr_db"]]
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
        amp_skips = 0
        consecutive_overflows = 0
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
            ).to(device)
            pair_negative, pair_positive, _ = make_multisnr_pair_batch(
                waveforms["tau_background"],
                waveforms["kielce_uav"],
                target_snr_db=snrs,
                epsilon=float(pairing["epsilon"]),
                peak_limit=float(pairing["peak_limit"]),
            )
            pair_positive = pair_positive.to(device)
            pair_negative = pair_negative.to(device)
            old_count = composition["dads_replay"] + composition[
                "g9_mechanical_hard_negative"
            ]
            pair_conditions = len(pair_positive)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.amp.autocast(device.type, enabled=amp):
                teacher_logits = teacher(batch[:old_count])
            with torch.amp.autocast(device.type, enabled=amp):
                student_logits = student(batch)
                pair_logits = student(torch.cat((pair_positive, pair_negative)))
                losses = constrained_losses(
                    student_logits,
                    labels,
                    student_logits[:old_count],
                    teacher_logits,
                    pair_logits[:pair_conditions],
                    pair_logits[pair_conditions:],
                    distill_weight=float(base["loss"]["distill_weight"]),
                    pair_weight=float(base["loss"]["pair_weight"]),
                    pair_margin=float(base["loss"]["pair_margin"]),
                    supervised_pos_weight=pos_weight,
                )
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            if not gradients_are_finite(trainable):
                previous_scale = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                next_scale = float(scaler.get_scale())
                if not amp or next_scale >= previous_scale:
                    raise RuntimeError("G16 non-finite gradients were not AMP overflow")
                amp_skips += 1
                amp_skips_total += 1
                consecutive_overflows += 1
                print(
                    f"G16 epoch {epoch} step {step + 1}/{batches} AMP overflow: "
                    f"scale={previous_scale:.0f}->{next_scale:.0f}",
                    flush=True,
                )
                if consecutive_overflows > MAX_CONSECUTIVE_AMP_OVERFLOWS:
                    raise RuntimeError("G16 exceeded AMP overflow safety limit")
                continue
            consecutive_overflows = 0
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
            for name in totals:
                totals[name] += float(losses[name].item())
            if (step + 1) % int(base["train"]["log_every_steps"]) == 0:
                print(
                    f"G16 epoch {epoch} step {step + 1}/{batches} "
                    f"loss={float(losses['total'].item()):.6f}",
                    flush=True,
                )
        if optimizer_steps <= 0:
            raise RuntimeError("G16 epoch had no optimizer steps")

        candidate_raw = {
            name: _predict(
                student,
                rows,
                audio_loader,
                device=device,
                batch_size=int(base["validation"]["batch_size"]),
                amp=amp,
                description=f"epoch {epoch} {name}",
            )
            for name, rows in raw_validation.items()
        }
        candidate_pair = counterfactual_metrics(
            student,
            pair_dataset,
            device=device,
            batch_size=int(
                config["counterfactual_validation"]["batch_size_pairs"]
            ),
            amp=amp,
            threshold=float(config["counterfactual_validation"]["threshold"]),
            description=f"epoch {epoch} counterfactual tune",
        )
        status = selection_status(
            baseline_raw=baseline_raw,
            candidate_raw=candidate_raw,
            baseline_pair=baseline_pair,
            candidate_pair=candidate_pair,
            gates=config["selection"],
        )
        safe = bool(
            status["checks"]["dads_safe"] and status["checks"]["raw_g14_safe"]
        )
        best_safe_score, bad_epochs, safe_improved = update_safe_early_stopping(
            dads_safe=safe,
            g14_score=float(status["score"]),
            best_safe_score=best_safe_score,
            bad_epochs=bad_epochs,
        )
        eligible = bool(status["eligible"])
        improved = eligible and status["score"] > best_score + 1.0e-8
        if improved:
            best_score = float(status["score"])
            best_epoch = epoch
        row = {
            "epoch": epoch,
            **{
                f"train_{name}_loss": totals[name] / optimizer_steps
                for name in ("total", "supervised", "distill", "pair")
            },
            "dads_f1": candidate_raw["dads_validation"]["f1"],
            "raw_g14_f1": candidate_raw["g14_tune"]["f1"],
            **{
                name: status[name]
                for name in (
                    "minimum_snr_lift_gain",
                    "minimum_snr_ordering_gain",
                    "maximum_snr_background_fpr_increase",
                    "monotonicity_gain",
                    "minimum_low_snr_tpr_gain",
                )
            },
            "dads_safe": status["checks"]["dads_safe"],
            "raw_g14_safe": status["checks"]["raw_g14_safe"],
            "checkpoint_eligible": eligible,
            "safe_score_improved": safe_improved,
            "bad_epochs": bad_epochs,
            "optimizer_steps": optimizer_steps,
            "amp_overflow_skipped_steps": amp_skips,
        }
        history.append(row)
        state = {
            "protocol": PROTOCOL,
            "config": base,
            "g16_config": config,
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "best_safe_score": best_safe_score,
            "bad_epochs": bad_epochs,
            "amp_skips_total": amp_skips_total,
            "history": history,
            "baseline_raw": baseline_raw,
            "baseline_counterfactual": baseline_pair,
            "candidate_raw": candidate_raw,
            "candidate_counterfactual": candidate_pair,
            "selection_status": status,
            "input_hashes": {
                "config": file_sha256(config_path),
                "prerequisites": verified,
                "train": train_hashes,
                "development": dev_hashes,
            },
        }
        _atomic_save(state, last_path)
        if improved:
            _atomic_save(state, best_path)
        _write_history(run_dir / "history.csv", history)
        print(
            f"G16 epoch {epoch} complete: dads_f1={row['dads_f1']:.6f} "
            f"raw_g14_f1={row['raw_g14_f1']:.6f} "
            f"min_lift_gain={row['minimum_snr_lift_gain']:.6f} "
            f"low_tpr_gain={row['minimum_low_snr_tpr_gain']:.6f} "
            f"eligible={eligible} safe_improved={safe_improved} "
            f"bad_epochs={bad_epochs}",
            flush=True,
        )
        if bad_epochs >= int(base["train"]["patience"]):
            break

    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "seed": seed,
        "decision": (
            "candidate_ready_for_development_regression"
            if best_path.is_file()
            else "terminate_no_multimetric_checkpoint"
        ),
        "best_epoch": best_epoch,
        "best_score": best_score if best_path.is_file() else None,
        "epochs_completed": len(history),
        "selection_gates": config["selection"],
        "baseline_raw": baseline_raw,
        "baseline_counterfactual": baseline_pair,
        "history": history,
        "amp_overflow_skipped_steps_total": amp_skips_total,
        "formal_training_started": True,
        "locked_datasets_read": [],
        "dev_holdout_read": False,
        "test_guard_final_read": False,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train G16 multi-SNR head")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g16_seed42_multisnr.yaml"),
    )
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(
        args.config,
        preflight_only=args.mode == "preflight",
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
