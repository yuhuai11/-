from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dads_crnn.audio import decode_wav_bytes, ensure_sample_rate, peak_normalize
from dads_crnn.train import resolve_device
from dads_crnn.train_panns import build_model


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 8000


def load_model(seed: int, device: torch.device) -> torch.nn.Module:
    checkpoint = ROOT / f"artifacts/g7_strict_retrain_v1/runs/seed_{seed}/best.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(saved["config"])
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval()


def windows(audio: np.ndarray) -> tuple[np.ndarray, list[float]]:
    if audio.size == 0:
        raise ValueError("Input WAV contains no audio samples")
    count = max(1, int(np.ceil(audio.size / WINDOW_SAMPLES)))
    output = np.zeros((count, WINDOW_SAMPLES), dtype=np.float32)
    durations = []
    for index in range(count):
        start = index * WINDOW_SAMPLES
        values = audio[start : start + WINDOW_SAMPLES]
        output[index, : values.size] = values
        output[index] = peak_normalize(output[index])
        durations.append(values.size / SAMPLE_RATE)
    return output, durations


def predict(
    wav_path: Path,
    *,
    seed: int,
    threshold: float,
    device_name: str,
    batch_size: int,
) -> dict:
    audio, original_rate = decode_wav_bytes(wav_path.read_bytes())
    audio = ensure_sample_rate(audio, original_rate, SAMPLE_RATE)
    values, durations = windows(audio)
    device = resolve_device(device_name)
    model = load_model(seed, device)
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            waveform = torch.from_numpy(values[start : start + batch_size]).to(device)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(waveform)
            probabilities.extend(torch.sigmoid(logits.float()).cpu().tolist())

    half_second = [
        {
            "window_index": index,
            "start_seconds": index * 0.5,
            "valid_duration_seconds": durations[index],
            "probability": float(probability),
            "prediction": "drone" if probability >= threshold else "background",
        }
        for index, probability in enumerate(probabilities)
    ]
    one_second = []
    for decision_index, start in enumerate(range(0, len(probabilities), 2)):
        group = probabilities[start : start + 2]
        probability = float(np.mean(group))
        one_second.append(
            {
                "decision_index": decision_index,
                "start_seconds": start * 0.5,
                "views": len(group),
                "probability": probability,
                "prediction": "drone" if probability >= threshold else "background",
            }
        )
    return {
        "task": "binary_drone_detection",
        "input": str(wav_path.resolve()),
        "original_sample_rate": original_rate,
        "model_sample_rate": SAMPLE_RATE,
        "seed": seed,
        "threshold": threshold,
        "tail_policy": "zero_pad_final_partial_half_second_window",
        "half_second_windows": half_second,
        "nonoverlap_one_second_mean": one_second,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G7 binary inference on one WAV")
    parser.add_argument("wav", type=Path)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.wav.is_file():
        raise FileNotFoundError(args.wav)
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    report = predict(
        args.wav,
        seed=args.seed,
        threshold=args.threshold,
        device_name=args.device,
        batch_size=args.batch_size,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()

