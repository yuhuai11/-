from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from numpy._core.multiarray import _reconstruct


EXPECTED_CHECKPOINT_SHA256 = "e2ee543a27919542c2ea03eabaa70b24dcd4e6c8e05621de6b67a94e4c5058e6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_checkpoint(path: Path) -> dict:
    if sha256(path) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("PANNs checkpoint SHA256 mismatch")
    original_module = _reconstruct.__module__
    try:
        # The checkpoint was produced with NumPy 1.x, while this environment
        # exposes the same function under numpy._core.  Preserve the legacy
        # qualified name only inside the restricted weights-only load.
        _reconstruct.__module__ = "numpy.core.multiarray"
        safe_types = [_reconstruct, np.ndarray, np.dtype, np.dtypes.Int64DType]
        with torch.serialization.safe_globals(safe_types):
            return torch.load(path, map_location="cpu", weights_only=True)
    finally:
        _reconstruct.__module__ = original_module


def build_binary_model(checkpoint: dict, vendor_dir: Path) -> torch.nn.Module:
    sys.path.insert(0, str(vendor_dir.resolve()))
    try:
        from models import Cnn14_16k
    finally:
        sys.path.pop(0)
    model = Cnn14_16k(
        sample_rate=16000,
        window_size=512,
        hop_size=160,
        mel_bins=64,
        fmin=50,
        fmax=8000,
        classes_num=527,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.fc_audioset = torch.nn.Linear(2048, 1)
    torch.nn.init.xavier_uniform_(model.fc_audioset.weight)
    torch.nn.init.zeros_(model.fc_audioset.bias)
    return model


def probe_batch(
    checkpoint: dict,
    vendor_dir: Path,
    waveform: np.ndarray,
    batch_size: int,
) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = optimizer = inputs = labels = None
    try:
        model = build_binary_model(checkpoint, vendor_dir).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scaler = torch.amp.GradScaler("cuda")
        inputs = torch.from_numpy(np.repeat(waveform[None, :], batch_size, axis=0)).cuda()
        labels = (
            torch.arange(batch_size, device="cuda", dtype=torch.float32)
            .remainder(2)
            .unsqueeze(1)
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            probabilities = model(inputs)["clipwise_output"]
        # The official PANNs model returns post-sigmoid probabilities. PyTorch
        # deliberately rejects probability-space BCE *inside* autocast; run
        # this small reduction in float32 outside autocast while retaining the
        # mixed-precision model forward and scaled backward pass.
        loss = torch.nn.functional.binary_cross_entropy(probabilities.float(), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        return {
            "batch_size": batch_size,
            "passed": True,
            "loss": float(loss.detach().cpu()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    except torch.cuda.OutOfMemoryError as error:
        return {"batch_size": batch_size, "passed": False, "error": str(error)}
    finally:
        del labels, inputs, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe PANNs Cnn14_16k training memory")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/g7_panns/checkpoints/Cnn14_16k_mAP=0.438.pth"),
    )
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=Path("artifacts/g7_panns/vendor/audioset_tagging_cnn/pytorch"),
    )
    parser.add_argument(
        "--waveform",
        type=Path,
        default=Path(
            "artifacts_full/audio_cache/test/0/"
            "train-00000-of-00039_rg0000_row0002_seg0000.npy"
        ),
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/g7_panns/gpu_memory_audit.json")
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script in the server GPU terminal")
    waveform = np.load(args.waveform).astype(np.float32)
    waveform = np.pad(waveform[:16000], (0, max(0, 16000 - waveform.size)))
    checkpoint = safe_checkpoint(args.checkpoint)
    results = []
    for batch_size in args.batch_sizes:
        result = probe_batch(checkpoint, args.vendor_dir, waveform, batch_size)
        results.append(result)
        print(json.dumps(result))
        if not result["passed"]:
            break
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    passed = [item["batch_size"] for item in results if item["passed"]]
    safe = [
        item["batch_size"]
        for item in results
        if item["passed"] and item["peak_reserved_bytes"] <= 0.75 * total_memory
    ]
    report = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "total_device_memory_bytes": total_memory,
        "recommendation_memory_fraction_limit": 0.75,
        "mixed_precision": True,
        "optimizer": "AdamW",
        "results": results,
        "maximum_tested_passing_batch_size": max(passed) if passed else None,
        "recommended_batch_size": max(safe) if safe else None,
        "locked_datasets_read": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not passed:
        raise RuntimeError("PANNs failed the batch-size-1 GPU training probe")


if __name__ == "__main__":
    main()
