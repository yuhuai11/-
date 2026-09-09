from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import load_config
from .data_firewall import file_sha256
from .panns import FrequencyMixStyle


PROTOCOL = "g7_r7_freq_mixstyle_single_variable_audit_v1"
BASELINE = Path("configs/g7_strict_retrain_v1.yaml")
CANDIDATE = Path("configs/g7_r7_freq_mixstyle.yaml")


def audit() -> dict:
    baseline = load_config(BASELINE)
    candidate = load_config(CANDIDATE)
    expected = json.loads(json.dumps(baseline))
    expected["protocol"] = candidate["protocol"]
    expected["model"]["frequency_mixstyle"] = candidate["model"]["frequency_mixstyle"]
    expected["train"]["seeds"] = [42]
    expected["output_dir"] = candidate["output_dir"]
    if candidate != expected:
        raise ValueError("G7-R7 config changes fields beyond Frequency MixStyle and outputs")

    module = FrequencyMixStyle(probability=0.5, beta_alpha=0.6)
    features = torch.arange(4 * 2 * 3 * 8, dtype=torch.float32).reshape(4, 2, 3, 8)
    module.eval()
    evaluation = module(features)
    module.train()
    mixed = module(features, force=True)
    mixed.sum().backward() if mixed.requires_grad else None
    result = {
        "passed": bool(
            torch.equal(evaluation, features)
            and mixed.shape == features.shape
            and torch.isfinite(mixed).all()
            and not torch.equal(mixed, features)
            and module.last_applied
        ),
        "protocol": PROTOCOL,
        "baseline_config": {"path": str(BASELINE), "sha256": file_sha256(BASELINE)},
        "candidate_config": {"path": str(CANDIDATE), "sha256": file_sha256(CANDIDATE)},
        "single_variable": "model.frequency_mixstyle",
        "training_only": True,
        "evaluation_identity": bool(torch.equal(evaluation, features)),
        "forced_training_changed_features": bool(not torch.equal(mixed, features)),
        "finite": bool(torch.isfinite(mixed).all()),
        "shape": list(mixed.shape),
        "parameters": candidate["model"]["frequency_mixstyle"],
        "locked_datasets_read": [],
        "model_inference_run": False,
        "optimizer_step_run": False,
    }
    if not result["passed"]:
        raise RuntimeError("G7-R7 Frequency MixStyle audit failed")
    output = Path("artifacts/g7_r7_freq_mixstyle/preflight/config_delta_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    print(json.dumps(audit(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
