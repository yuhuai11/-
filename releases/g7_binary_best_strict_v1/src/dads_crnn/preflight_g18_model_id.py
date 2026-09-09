from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .config import load_config
from .data_firewall import audit_csv_rows, file_sha256, reject_locked_path
from .model_identification import G7ModelIdentifier
from .panns import PannsCnn14Binary
from .train import resolve_device, set_seed


PROTOCOL = "g18_p1_frozen_g7_model_id_preflight_v1"


def select_balanced_models(
    frame: pd.DataFrame, *, samples_per_model: int, seed: int
) -> pd.DataFrame:
    if samples_per_model <= 0:
        raise ValueError("samples_per_model must be positive")
    required = {"model_id", "target_index", "audio_sha256"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"G18 registry lacks columns: {sorted(missing)}")
    selected = []
    for offset, (_, rows) in enumerate(frame.groupby("model_id", sort=True)):
        if len(rows) < samples_per_model:
            raise ValueError("G18 model class is too small for the preflight batch")
        weights = np.power(
            rows.groupby("audio_sha256")["audio_sha256"].transform("size").to_numpy(
                dtype=np.float64
            ),
            -1.0,
        )
        selected.append(
            rows.sample(
                n=samples_per_model,
                replace=False,
                weights=weights,
                random_state=seed + offset,
            )
        )
    return pd.concat(selected, ignore_index=True).sample(
        frac=1.0, random_state=seed + 100
    ).reset_index(drop=True)


class MemmapAudioLoader:
    def __init__(self, target_samples: int) -> None:
        self.target_samples = target_samples
        self.maps: OrderedDict[str, np.ndarray] = OrderedDict()

    def load(self, row: pd.Series) -> np.ndarray:
        path = str(row["cache_path"])
        reject_locked_path(Path(path), context="G18 cached development audio")
        if path not in self.maps:
            self.maps[path] = np.load(path, mmap_mode="r")
        waveform = np.asarray(
            self.maps[path][int(row["cache_index"])], dtype=np.float32
        ).reshape(-1)
        if waveform.size != self.target_samples or not np.isfinite(waveform).all():
            raise ValueError("Invalid G18 cached waveform")
        return waveform

    def batch(self, frame: pd.DataFrame) -> torch.Tensor:
        return torch.from_numpy(
            np.stack([self.load(row) for _, row in frame.iterrows()])
        )


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    reject_locked_path(path, context="G18 P1 preflight input")
    return path.resolve(strict=True)


def preflight(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected G18 P1 protocol")

    registry_path = _resolve(root, config["inputs"]["known_train"]["path"])
    registry_audit_path = _resolve(root, config["inputs"]["registry_audit"]["path"])
    g7_path = _resolve(root, config["model"]["binary_checkpoint_path"])
    official_path = _resolve(root, config["model"]["checkpoint_path"])
    for name, path, expected in (
        ("known_train", registry_path, config["inputs"]["known_train"]["sha256"]),
        (
            "registry_audit",
            registry_audit_path,
            config["inputs"]["registry_audit"]["sha256"],
        ),
        ("g7_checkpoint", g7_path, config["model"]["binary_checkpoint_sha256"]),
        ("official_checkpoint", official_path, config["model"]["checkpoint_sha256"]),
    ):
        if file_sha256(path) != str(expected):
            raise ValueError(f"G18 P1 input SHA256 mismatch: {name}")

    registry_audit = json.loads(registry_audit_path.read_text(encoding="utf-8"))
    if not (
        registry_audit.get("passed") is True
        and registry_audit.get("protocol") == "g18_p0_open_set_model_id_registry_v1"
        and registry_audit.get("locked_datasets_read") == []
    ):
        raise ValueError("G18 P0 registry audit is not valid")
    rows_audited = audit_csv_rows(
        registry_path,
        required_columns=(
            "model_id",
            "target_index",
            "audio_sha256",
            "cache_path",
            "cache_index",
        ),
    )
    frame = pd.read_csv(registry_path)
    known_models = list(registry_audit["known_models"])
    if sorted(frame["model_id"].unique()) != sorted(known_models):
        raise ValueError("G18 known-model registry class mismatch")

    seed = int(config["preflight"]["seed"])
    set_seed(seed)
    batch_rows = select_balanced_models(
        frame,
        samples_per_model=int(config["preflight"]["samples_per_model"]),
        seed=seed,
    )
    loader = MemmapAudioLoader(
        int(config["data"]["sample_rate"] * config["data"]["clip_seconds"])
    )
    waveforms = loader.batch(batch_rows)
    targets = torch.from_numpy(batch_rows["target_index"].to_numpy(dtype=np.int64))
    device = resolve_device(str(config["preflight"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("G18 P1 preflight requires the server CUDA GPU")

    detector = PannsCnn14Binary(
        initialization=str(config["model"]["initialization"]),
        vendor_dir=str(config["model"]["vendor_dir"]),
        checkpoint_path=official_path.as_posix(),
        checkpoint_sha256=str(config["model"]["checkpoint_sha256"]),
        spec_augment=False,
        frontend_precision="float32",
        binary_checkpoint_path=g7_path.as_posix(),
        binary_checkpoint_sha256=str(config["model"]["binary_checkpoint_sha256"]),
        trainable_scope="binary_head_only",
    )
    model = G7ModelIdentifier(
        detector,
        embedding_dim=int(config["model_id_head"]["embedding_dim"]),
        classes=len(known_models),
    ).to(device)
    model.train()
    waveforms = waveforms.to(device)
    targets = targets.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    drone_logits, model_logits = model(waveforms)
    loss = F.cross_entropy(model_logits.float(), targets)
    loss.backward()

    trainable_gradients_finite = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for name, parameter in model.named_parameters()
        if name.startswith(("embedding_norm.", "classifier."))
    )
    detector_gradients_absent = all(
        parameter.grad is None for parameter in model.detector.parameters()
    )
    if not (
        torch.isfinite(drone_logits).all()
        and torch.isfinite(model_logits).all()
        and torch.isfinite(loss)
        and trainable_gradients_finite
        and detector_gradients_absent
    ):
        raise RuntimeError("G18 P1 produced invalid logits or gradients")

    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(device),
        "seed": seed,
        "batch_size": len(batch_rows),
        "batch_model_counts": dict(
            sorted(Counter(batch_rows["model_id"]).items())
        ),
        "known_models": known_models,
        "loss": float(loss.detach().cpu()),
        "drone_logits_finite": True,
        "model_logits_finite": True,
        "trainable_gradients_finite": trainable_gradients_finite,
        "frozen_g7_gradients_absent": detector_gradients_absent,
        "trainable_parameter_names": model.trainable_parameter_names,
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "frozen_g7_parameter_count": sum(
            parameter.numel() for parameter in model.detector.parameters()
        ),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "optimizer_step_exercised": False,
        "checkpoint_written": False,
        "formal_training_started": False,
        "unknown_threshold_calibrated": False,
        "locked_datasets_read": [],
        "registry_rows_firewall_audited": rows_audited,
        "inputs": {
            "config_sha256": file_sha256(config_path),
            "known_train_sha256": file_sha256(registry_path),
            "registry_audit_sha256": file_sha256(registry_audit_path),
            "g7_checkpoint_sha256": file_sha256(g7_path),
            "official_checkpoint_sha256": file_sha256(official_path),
        },
        "ready_for_seed42_feasibility_training": True,
    }
    output_dir = root / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight the frozen-G7 model-ID head.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g18_model_id_preflight.yaml")
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    preflight(args.config, args.root)


if __name__ == "__main__":
    main()
