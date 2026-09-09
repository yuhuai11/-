from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .audit_experiment_delta import config_differences, sha256_file
from .config import ensure_dirs, load_config
from .train import _build_feature_extractor, _build_model


ALLOWED_CHANGES = {"model.temporal_pooling", "output_dir"}
EXPECTED_BASELINE_PARAMETERS = 1_012_193
EXPECTED_CANDIDATE_PARAMETERS = 1_012_449
EXPECTED_NEW_STATE_KEYS = {"temporal_attention.weight"}
HISTORICAL_ANCHOR = Path("artifacts/g5_mfcc64/baseline_inventory.json")
HISTORICAL_ANCHOR_KEYS = {
    "baseline_config",
    "baseline_checkpoint",
    "baseline_metrics",
    "baseline_calibration",
    "baseline_tune_predictions",
    "baseline_holdout_predictions",
    "dads_manifest",
    "val_ood_tune_manifest",
    "val_ood_holdout_manifest",
    "features_code",
    "audio_code",
    "config_code",
    "metrics_code",
    "dataset_code",
    "augmentation_code",
    "external_data_code",
    "feature_audit_code",
}
HISTORICAL_INTENTIONAL_CODE_CHANGES = {
    "train_code",
    "model_code",
    "calibration_code",
    "gate_code",
}


def _protected_paths(guardrails: dict[str, Any], manifest: Path) -> dict[str, Path]:
    baseline = guardrails["baseline"]
    return {
        "baseline_config": Path(baseline["config"]),
        "baseline_checkpoint": Path(baseline["checkpoint"]),
        "baseline_metrics": Path(baseline["metrics"]),
        "baseline_val_probabilities": Path(baseline["val_probabilities"]),
        "baseline_val_labels": Path(baseline["val_labels"]),
        "baseline_test_probabilities": Path(baseline["test_probabilities"]),
        "baseline_test_labels": Path(baseline["test_labels"]),
        "baseline_calibration": Path(baseline["calibration"]),
        "baseline_tune_predictions": Path(baseline["predictions"]["tune"]),
        "baseline_holdout_predictions": Path(baseline["predictions"]["holdout"]),
        "dads_manifest": manifest,
        "val_ood_manifest": Path("artifacts/val_ood/manifests/val_ood_manifest.csv"),
        "val_ood_tune_manifest": Path(
            "artifacts/val_ood/manifests/val_ood_tune_manifest.csv"
        ),
        "val_ood_holdout_manifest": Path(
            "artifacts/val_ood/manifests/val_ood_holdout_manifest.csv"
        ),
        "val_ood_audit": Path("artifacts/val_ood/manifests/val_ood_audit.json"),
        "features_code": Path("src/dads_crnn/features.py"),
        "audio_code": Path("src/dads_crnn/audio.py"),
        "config_code": Path("src/dads_crnn/config.py"),
        "train_code": Path("src/dads_crnn/train.py"),
        "model_code": Path("src/dads_crnn/model.py"),
        "metrics_code": Path("src/dads_crnn/metrics.py"),
        "dataset_code": Path("src/dads_crnn/dataset.py"),
        "augmentation_code": Path("src/dads_crnn/augmentation.py"),
        "sampling_code": Path("src/dads_crnn/sampling.py"),
        "external_data_code": Path("src/dads_crnn/external_data.py"),
        "calibration_code": Path("src/dads_crnn/calibrate_ood.py"),
        "external_evaluation_code": Path("src/dads_crnn/evaluate_external.py"),
        "gate_code": Path("src/dads_crnn/gate_candidate.py"),
        "feature_audit_code": Path("src/dads_crnn/audit_feature_ablation.py"),
        "temporal_pooling_audit_code": Path(
            "src/dads_crnn/audit_temporal_pooling_ablation.py"
        ),
    }


def _inventory(paths: dict[str, Path]) -> tuple[dict[str, dict[str, str]], list[str]]:
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return {}, missing
    return (
        {
            name: {"path": path.as_posix(), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        [],
    )


def _historical_anchor_matches(inventory: dict[str, dict[str, str]]) -> bool:
    if not HISTORICAL_ANCHOR.is_file():
        return False
    saved = json.loads(HISTORICAL_ANCHOR.read_text(encoding="utf-8"))
    if set(saved) != HISTORICAL_ANCHOR_KEYS | HISTORICAL_INTENTIONAL_CODE_CHANGES:
        return False
    for key in HISTORICAL_ANCHOR_KEYS:
        if saved.get(key) != inventory.get(key):
            return False
    return True


def _manual_mean_forward(model: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    encoded = model.cnn(features)
    batch, channels, mel_bins, frames = encoded.shape
    sequence = encoded.permute(0, 3, 1, 2).reshape(
        batch, frames, channels * mel_bins
    )
    sequence, _ = model.rnn(sequence)
    return model.classifier(sequence.mean(dim=1)).squeeze(1)


def _candidate_artifacts(
    guardrails: dict[str, Any], mode: str
) -> tuple[dict[str, dict[str, str]], list[str]]:
    required = guardrails["audit"].get("require_candidate_artifact_hashes", [])
    if required is True:
        required = ["checkpoint", "metrics"]
    paths = {name: Path(guardrails["candidate"][name]) for name in required}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if mode == "preflight" or missing:
        return {}, missing
    return (
        {
            name: {"path": path.as_posix(), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        [],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the strictly controlled G6 temporal-attention ablation"
    )
    parser.add_argument("--baseline", default="configs/crnn_dads_full_augmented_g2.yaml")
    parser.add_argument(
        "--candidate",
        default="configs/crnn_dads_full_augmented_g6_temporal_attention.yaml",
    )
    parser.add_argument(
        "--guardrails", default="configs/g6_temporal_attention_guardrails.yaml"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts_full/manifests/dads_all_seed42.csv"),
    )
    parser.add_argument("--mode", choices=("preflight", "verify"), default="preflight")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("artifacts/g6_temporal_attention/baseline_inventory.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/g6_temporal_attention/delta_audit.json"),
    )
    args = parser.parse_args()

    baseline = load_config(args.baseline)
    candidate = load_config(args.candidate)
    guardrails = load_config(args.guardrails)
    differences = config_differences(baseline, candidate)
    protected_inventory, missing_protected = _inventory(
        _protected_paths(guardrails, args.manifest)
    )
    inventory_match = True
    if args.inventory.exists():
        inventory_match = (
            json.loads(args.inventory.read_text(encoding="utf-8"))
            == protected_inventory
        )
    elif args.mode == "verify":
        inventory_match = False

    torch.manual_seed(20260717)
    baseline_model = _build_model(baseline)
    baseline_rng_state = torch.get_rng_state().clone()
    torch.manual_seed(20260717)
    candidate_model = _build_model(candidate)
    candidate_rng_state = torch.get_rng_state().clone()
    baseline_parameters = sum(parameter.numel() for parameter in baseline_model.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate_model.parameters())
    baseline_keys = set(baseline_model.state_dict())
    candidate_keys = set(candidate_model.state_dict())
    new_state_keys = candidate_keys - baseline_keys

    baseline_checkpoint = torch.load(
        guardrails["baseline"]["checkpoint"], map_location="cpu", weights_only=False
    )
    baseline_checkpoint_config_match = baseline_checkpoint.get("config") == baseline
    baseline_checkpoint_seed_match = int(baseline_checkpoint.get("seed", -1)) == 42
    baseline_checkpoint_finite = all(
        not torch.is_tensor(value) or bool(torch.isfinite(value).all())
        for value in baseline_checkpoint["model"].values()
    )
    baseline_load = baseline_model.load_state_dict(baseline_checkpoint["model"], strict=True)
    candidate_load = candidate_model.load_state_dict(
        baseline_checkpoint["model"], strict=False
    )

    torch.manual_seed(20260717)
    time = torch.arange(16000, dtype=torch.float32) / 16000.0
    waveform = torch.stack(
        (
            torch.sin(2.0 * torch.pi * 400.0 * time),
            torch.sin(2.0 * torch.pi * 900.0 * time)
            + 0.05 * torch.randn_like(time),
        )
    )
    baseline_feature = _build_feature_extractor(baseline)
    candidate_feature = _build_feature_extractor(candidate)
    baseline_features = baseline_feature(waveform)
    candidate_features = candidate_feature(waveform)
    baseline_model.eval()
    candidate_model.eval()
    with torch.no_grad():
        baseline_logits = baseline_model(baseline_features)
        old_path_logits = _manual_mean_forward(baseline_model, baseline_features)
        candidate_logits = candidate_model(candidate_features)
        encoded = candidate_model.cnn(candidate_features)
        batch, channels, mel_bins, frames = encoded.shape
        sequence = encoded.permute(0, 3, 1, 2).reshape(
            batch, frames, channels * mel_bins
        )
        sequence, _ = candidate_model.rnn(sequence)
        scores = candidate_model.temporal_attention(sequence).squeeze(-1)
        attention_weights = torch.softmax(scores, dim=1)

    candidate_model.zero_grad(set_to_none=True)
    probe = torch.randn(2, frames, sequence.shape[-1])
    probe_target = torch.linspace(-1.0, 1.0, sequence.shape[-1])
    probe_loss = (candidate_model.temporal_pool(probe) * probe_target).sum()
    probe_loss.backward()
    attention_gradient = candidate_model.temporal_attention.weight.grad

    output_dir = Path(candidate["output_dir"])
    output_empty = not output_dir.exists() or not any(output_dir.rglob("*"))
    unapproved_seed_directories = (
        [
            path.as_posix()
            for path in output_dir.glob("seed_*")
            if path.is_dir() and path.name != "seed_42"
        ]
        if output_dir.exists()
        else []
    )
    artifacts, missing_candidate = _candidate_artifacts(guardrails, args.mode)
    candidate_artifacts_exist = not missing_candidate and bool(artifacts)
    checkpoint_config_match = args.mode == "preflight"
    checkpoint_state_loads = args.mode == "preflight"
    checkpoint_seed_match = args.mode == "preflight"
    checkpoint_epoch_match = args.mode == "preflight"
    checkpoint_weights_finite = args.mode == "preflight"
    metrics_metadata_match = args.mode == "preflight"
    prediction_arrays_valid = args.mode == "preflight"
    if args.mode == "verify" and candidate_artifacts_exist:
        checkpoint = torch.load(
            guardrails["candidate"]["checkpoint"],
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_config_match = checkpoint.get("config") == candidate
        checkpoint_seed_match = int(checkpoint.get("seed", -1)) == 42
        checkpoint_weights_finite = all(
            not torch.is_tensor(value) or bool(torch.isfinite(value).all())
            for value in checkpoint["model"].values()
        )
        trained_model = _build_model(candidate)
        try:
            trained_model.load_state_dict(checkpoint["model"], strict=True)
            checkpoint_state_loads = True
        except RuntimeError:
            checkpoint_state_loads = False
        metrics = json.loads(
            Path(guardrails["candidate"]["metrics"]).read_text(encoding="utf-8")
        )
        metrics_metadata_match = bool(
            int(metrics.get("seed", -1)) == 42
            and metrics.get("model_type") == "crnn"
            and metrics.get("feature_type") == "log_mel"
            and metrics.get("temporal_pooling") == "attention"
            and int(metrics.get("parameter_count", -1))
            == EXPECTED_CANDIDATE_PARAMETERS
            and metrics.get("checkpoint_sha256")
            == artifacts["checkpoint"]["sha256"]
            and isinstance(metrics.get("prediction_sha256"), dict)
            and all(
                metrics["prediction_sha256"].get(name)
                == artifacts[name]["sha256"]
                for name in (
                    "val_probabilities",
                    "val_labels",
                    "test_probabilities",
                    "test_labels",
                )
            )
        )
        checkpoint_epoch_match = int(checkpoint.get("epoch", -1)) == int(
            metrics.get("best_epoch", -2)
        )
        prediction_arrays_valid = True
        for split in ("val", "test"):
            probabilities = np.load(guardrails["candidate"][f"{split}_probabilities"])
            labels = np.load(guardrails["candidate"][f"{split}_labels"])
            baseline_labels = np.load(
                guardrails["baseline"][f"{split}_labels"]
            )
            prediction_arrays_valid = bool(
                prediction_arrays_valid
                and probabilities.shape == labels.shape
                and probabilities.ndim == 1
                and probabilities.size > 0
                and np.isfinite(probabilities).all()
                and ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
                and np.isin(labels, [0, 1]).all()
                and np.array_equal(labels, baseline_labels)
            )

    shared_model = {
        key: value for key, value in candidate["model"].items() if key != "temporal_pooling"
    }
    attention = candidate_model.temporal_attention
    checks = {
        "only_approved_config_changes": set(differences) == ALLOWED_CHANGES,
        "baseline_defaults_to_mean_pooling": "temporal_pooling" not in baseline["model"]
        and baseline_model.temporal_pooling == "mean",
        "candidate_uses_attention_pooling": candidate["model"].get("temporal_pooling")
        == "attention"
        and candidate_model.temporal_pooling == "attention",
        "data_config_unchanged": candidate["data"] == baseline["data"],
        "feature_config_unchanged": candidate["features"] == baseline["features"],
        "training_config_unchanged": candidate["train"] == baseline["train"],
        "evaluation_config_unchanged": candidate["eval"] == baseline["eval"],
        "shared_model_config_unchanged": shared_model == baseline["model"],
        "natural_sampling_preserved": "sampling" not in candidate["train"],
        "seed42_only": candidate["train"]["seeds"] == [42],
        "single_worker_reproducibility": int(candidate["train"]["num_workers"]) == 0,
        "feature_outputs_identical": torch.equal(baseline_features, candidate_features)
        and tuple(baseline_features.shape) == (2, 1, 64, 101),
        "parameter_count_exact": baseline_parameters == EXPECTED_BASELINE_PARAMETERS
        and candidate_parameters == EXPECTED_CANDIDATE_PARAMETERS,
        "only_attention_state_added": new_state_keys == EXPECTED_NEW_STATE_KEYS
        and not (baseline_keys - candidate_keys),
        "attention_shape_and_bias_exact": attention is not None
        and tuple(attention.weight.shape) == (1, 256)
        and attention.bias is None,
        "attention_zero_initialized": attention is not None
        and bool(torch.equal(attention.weight, torch.zeros_like(attention.weight))),
        "model_construction_rng_identical": bool(
            torch.equal(baseline_rng_state, candidate_rng_state)
        ),
        "g2_checkpoint_config_match": baseline_checkpoint_config_match,
        "g2_checkpoint_seed_match": baseline_checkpoint_seed_match,
        "g2_checkpoint_weights_finite": baseline_checkpoint_finite,
        "g2_checkpoint_strict_load": not baseline_load.missing_keys
        and not baseline_load.unexpected_keys,
        "g2_old_forward_path_unchanged": bool(
            torch.allclose(baseline_logits, old_path_logits, rtol=0.0, atol=1e-7)
        ),
        "candidate_loads_only_new_attention_key": set(candidate_load.missing_keys)
        == EXPECTED_NEW_STATE_KEYS
        and not candidate_load.unexpected_keys,
        "initial_attention_equals_mean": bool(
            torch.allclose(candidate_logits, baseline_logits, rtol=1e-6, atol=1e-7)
        ),
        "attention_weights_uniform": bool(
            torch.allclose(
                attention_weights,
                torch.full_like(attention_weights, 1.0 / frames),
                rtol=0.0,
                atol=1e-7,
            )
        ),
        "attention_weights_normalized": bool(
            (attention_weights >= 0).all()
            and torch.allclose(
                attention_weights.sum(dim=1),
                torch.ones(attention_weights.shape[0]),
                rtol=0.0,
                atol=1e-6,
            )
        ),
        "attention_gradient_finite_nonzero": attention_gradient is not None
        and bool(torch.isfinite(attention_gradient).all())
        and float(attention_gradient.norm()) > 0.0,
        "forward_outputs_finite": bool(torch.isfinite(candidate_logits).all())
        and tuple(candidate_logits.shape) == (2,),
        "independent_output_directory": candidate["output_dir"]
        != baseline["output_dir"],
        "candidate_run_state_valid": output_empty
        if args.mode == "preflight"
        else candidate_artifacts_exist,
        "no_unapproved_seed_directories": not unapproved_seed_directories,
        "candidate_checkpoint_config_match": checkpoint_config_match,
        "candidate_checkpoint_state_loads": checkpoint_state_loads,
        "candidate_checkpoint_seed_match": checkpoint_seed_match,
        "candidate_checkpoint_epoch_match": checkpoint_epoch_match,
        "candidate_checkpoint_weights_finite": checkpoint_weights_finite,
        "candidate_metrics_metadata_match": metrics_metadata_match,
        "candidate_prediction_arrays_valid": prediction_arrays_valid,
        "protected_artifacts_exist": not missing_protected,
        "protected_artifacts_unchanged": inventory_match,
        "historical_g5_anchor_unchanged": _historical_anchor_matches(
            protected_inventory
        ),
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
        "feature_shape": list(candidate_features.shape),
        "temporal_frames": int(frames),
        "model_parameters": {
            "baseline": baseline_parameters,
            "candidate": candidate_parameters,
            "delta": candidate_parameters - baseline_parameters,
        },
        "initial_logit_max_abs_difference": float(
            torch.max(torch.abs(candidate_logits - baseline_logits))
        ),
        "attention_gradient_norm": float(attention_gradient.norm())
        if attention_gradient is not None
        else None,
        "protected_inventory": protected_inventory,
        "candidate_artifacts": artifacts,
        "unapproved_seed_directories": unapproved_seed_directories,
        "historical_anchor_intentional_code_changes": sorted(
            HISTORICAL_INTENTIONAL_CODE_CHANGES
        ),
        "checks": checks,
    }
    ensure_dirs(args.output.parent, args.inventory.parent)
    if (
        args.mode == "preflight"
        and not args.inventory.exists()
        and not missing_protected
        and all(checks.values())
    ):
        args.inventory.write_text(
            json.dumps(protected_inventory, indent=2), encoding="utf-8"
        )
        checks["protected_artifacts_unchanged"] = True
        report["passed"] = all(checks.values())
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("G6 temporal-attention ablation audit failed closed")


if __name__ == "__main__":
    main()
