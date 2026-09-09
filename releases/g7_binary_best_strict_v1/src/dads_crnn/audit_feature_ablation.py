from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .audit_experiment_delta import config_differences, sha256_file
from .config import ensure_dirs, load_config
from .train import _build_feature_extractor, _build_model


ALLOWED_CHANGES = {
    "output_dir",
    "features.type",
    "features.n_mfcc",
    "features.preemphasis",
    "features.window_type",
}
EXPECTED_MFCC_SETTINGS = {
    "type": "mfcc",
    "n_mfcc": 64,
    "n_mels": 64,
    "n_fft": 512,
    "win_length": 400,
    "hop_length": 160,
    "f_min": 20,
    "f_max": 8000,
    "preemphasis": 0.0,
    "window_type": "hann",
}


def _protected_paths(guardrails: dict, manifest: Path) -> dict[str, Path]:
    baseline = guardrails["baseline"]
    return {
        "baseline_config": Path(baseline["config"]),
        "baseline_checkpoint": Path(baseline["checkpoint"]),
        "baseline_metrics": Path(baseline["metrics"]),
        "baseline_calibration": Path(baseline["calibration"]),
        "baseline_tune_predictions": Path(baseline["predictions"]["tune"]),
        "baseline_holdout_predictions": Path(baseline["predictions"]["holdout"]),
        "dads_manifest": manifest,
        "val_ood_tune_manifest": Path("artifacts/val_ood/manifests/val_ood_tune_manifest.csv"),
        "val_ood_holdout_manifest": Path("artifacts/val_ood/manifests/val_ood_holdout_manifest.csv"),
        "features_code": Path("src/dads_crnn/features.py"),
        "audio_code": Path("src/dads_crnn/audio.py"),
        "config_code": Path("src/dads_crnn/config.py"),
        "train_code": Path("src/dads_crnn/train.py"),
        "model_code": Path("src/dads_crnn/model.py"),
        "metrics_code": Path("src/dads_crnn/metrics.py"),
        "dataset_code": Path("src/dads_crnn/dataset.py"),
        "augmentation_code": Path("src/dads_crnn/augmentation.py"),
        "external_data_code": Path("src/dads_crnn/external_data.py"),
        "calibration_code": Path("src/dads_crnn/calibrate_ood.py"),
        "gate_code": Path("src/dads_crnn/gate_candidate.py"),
        "feature_audit_code": Path("src/dads_crnn/audit_feature_ablation.py"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the strictly controlled G5 MFCC-64 ablation")
    parser.add_argument("--baseline", default="configs/crnn_dads_full_augmented_g2.yaml")
    parser.add_argument("--candidate", default="configs/crnn_dads_full_augmented_g5_mfcc64.yaml")
    parser.add_argument("--guardrails", default="configs/g5_mfcc64_guardrails.yaml")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts_full/manifests/dads_all_seed42.csv"))
    parser.add_argument("--mode", choices=("preflight", "verify"), default="preflight")
    parser.add_argument("--inventory", type=Path, default=Path("artifacts/g5_mfcc64/baseline_inventory.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/g5_mfcc64/delta_audit.json"))
    args = parser.parse_args()

    baseline = load_config(args.baseline)
    candidate = load_config(args.candidate)
    guardrails = load_config(args.guardrails)
    differences = config_differences(baseline, candidate)
    protected_paths = _protected_paths(guardrails, args.manifest)
    missing = [name for name, path in protected_paths.items() if not path.is_file()]
    inventory = (
        {name: {"path": path.as_posix(), "sha256": sha256_file(path)} for name, path in protected_paths.items()}
        if not missing
        else {}
    )
    inventory_match = True
    if args.inventory.exists():
        inventory_match = json.loads(args.inventory.read_text(encoding="utf-8")) == inventory
    elif args.mode == "verify":
        inventory_match = False

    torch.manual_seed(20260717)
    time = torch.arange(16000, dtype=torch.float32) / 16000.0
    waveform = torch.stack(
        (
            torch.sin(2.0 * torch.pi * 400.0 * time),
            torch.sin(2.0 * torch.pi * 900.0 * time) + 0.05 * torch.randn_like(time),
        )
    )
    baseline_feature = _build_feature_extractor(baseline)
    candidate_feature = _build_feature_extractor(candidate)
    baseline_output = baseline_feature(waveform)
    candidate_output = candidate_feature(waveform)
    baseline_model = _build_model(baseline)
    candidate_model = _build_model(candidate)
    baseline_parameters = sum(parameter.numel() for parameter in baseline_model.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate_model.parameters())
    logits = candidate_model(candidate_output)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0.0, 1.0]))
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in candidate_model.parameters()
    )
    dct = candidate_feature.dct.detach()
    identity = torch.eye(dct.shape[0], dtype=dct.dtype)
    dct_orthogonality_error = float(torch.max(torch.abs(dct @ dct.T - identity)))

    output_dir = Path(candidate["output_dir"])
    output_empty = not output_dir.exists() or not any(output_dir.rglob("*"))
    candidate_checkpoint = Path(guardrails["candidate"]["checkpoint"])
    candidate_metrics = Path(guardrails["candidate"]["metrics"])
    candidate_artifacts_exist = candidate_checkpoint.is_file() and candidate_metrics.is_file()
    checkpoint_config_match = args.mode == "preflight"
    if args.mode == "verify" and candidate_artifacts_exist:
        checkpoint = torch.load(candidate_checkpoint, map_location="cpu", weights_only=False)
        checkpoint_config_match = checkpoint.get("config") == candidate

    checks = {
        "only_approved_config_changes": set(differences) == ALLOWED_CHANGES,
        "baseline_is_g2_log_mel": baseline["features"].get("type") == "log_mel",
        "candidate_mfcc_settings_exact": candidate["features"] == EXPECTED_MFCC_SETTINGS,
        "g2_augmentation_unchanged": candidate["train"]["augmentation"]
        == baseline["train"]["augmentation"],
        "natural_sampling_preserved": "sampling" not in candidate["train"],
        "training_settings_unchanged": {
            key: value for key, value in candidate["train"].items() if key != "augmentation"
        }
        == {key: value for key, value in baseline["train"].items() if key != "augmentation"},
        "model_config_unchanged": candidate["model"] == baseline["model"],
        "feature_shape_equal": tuple(baseline_output.shape) == tuple(candidate_output.shape)
        == (2, 1, 64, 101),
        "feature_outputs_finite": bool(torch.isfinite(baseline_output).all())
        and bool(torch.isfinite(candidate_output).all()),
        "hann_window_equal": bool(torch.equal(baseline_feature.window, candidate_feature.window)),
        "model_parameter_count_equal": baseline_parameters == candidate_parameters == 1_012_193,
        "candidate_forward_backward_finite": bool(torch.isfinite(logits).all())
        and math_is_finite(float(loss))
        and finite_gradients,
        "dct_is_orthonormal": dct_orthogonality_error < 1e-5,
        "independent_output_directory": candidate["output_dir"] != baseline["output_dir"],
        "candidate_run_state_valid": output_empty if args.mode == "preflight" else candidate_artifacts_exist,
        "candidate_checkpoint_config_match": checkpoint_config_match,
        "protected_artifacts_exist": not missing,
        "protected_artifacts_unchanged": inventory_match,
    }
    report = {
        "passed": all(checks.values()),
        "mode": args.mode,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "guardrails": args.guardrails,
        "baseline_config_sha256": sha256_file(Path(args.baseline)),
        "candidate_config_sha256": sha256_file(Path(args.candidate)),
        "guardrails_sha256": sha256_file(Path(args.guardrails)),
        "differences": differences,
        "allowed_changes": sorted(ALLOWED_CHANGES),
        "feature_shapes": {
            "baseline": list(baseline_output.shape),
            "candidate": list(candidate_output.shape),
        },
        "model_parameters": {
            "baseline": baseline_parameters,
            "candidate": candidate_parameters,
        },
        "dct_orthogonality_max_error": dct_orthogonality_error,
        "protected_inventory": inventory,
        "checks": checks,
    }
    ensure_dirs(args.output.parent, args.inventory.parent)
    if args.mode == "preflight" and not args.inventory.exists() and not missing:
        args.inventory.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
        checks["protected_artifacts_unchanged"] = True
        report["passed"] = all(checks.values())
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("G5 MFCC-64 feature ablation audit failed closed")


def math_is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


if __name__ == "__main__":
    main()
