from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .data_firewall import file_sha256
from .train_g18_model_id import MULTISEED_PROTOCOL, _verify_inputs


PROTOCOL = "g18_p4_multiseed_preflight_v1"
SEEDS = (43, 44)


def comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["train"].pop("seed", None)
    return result


def _config_paths(root: Path) -> dict[int, Path]:
    return {
        seed: (root / f"configs/g18_model_id_seed{seed}.yaml").resolve(strict=True)
        for seed in SEEDS
    }


def _validate_configs(root: Path) -> tuple[dict[int, Path], dict[int, dict[str, Any]]]:
    paths = _config_paths(root)
    configs = {seed: load_config(path) for seed, path in paths.items()}
    for seed, config in configs.items():
        if config.get("protocol") != MULTISEED_PROTOCOL:
            raise ValueError(f"G18 seed {seed} has an invalid replication protocol")
        if int(config["train"]["seed"]) != seed:
            raise ValueError(f"G18 replication config seed mismatch: {seed}")
        if str(config["output_dir"]) != "artifacts/g18_model_identification/p4_multiseed":
            raise ValueError("G18 multiseed output directory is not frozen")
        if any(
            name in config["inputs"]
            for name in ("unknown_tune", "known_holdout", "unknown_holdout")
        ):
            raise ValueError("G18 multiseed config binds forbidden development inputs")
        _verify_inputs(config, root)
    if comparable_config(configs[43]) != comparable_config(configs[44]):
        raise ValueError("G18 seed configs differ outside the random seed")
    return paths, configs


def preflight(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_paths, configs = _validate_configs(root)
    outputs = {
        seed: root
        / str(configs[seed]["output_dir"])
        / f"seed_{seed}"
        for seed in SEEDS
    }
    existing = {
        seed: sorted(path.name for path in output.iterdir()) if output.is_dir() else []
        for seed, output in outputs.items()
    }
    if any(existing.values()):
        raise FileExistsError(f"G18 multiseed output is not empty: {existing}")
    training_module = Path(__file__).with_name("train_g18_model_id.py")
    report = {
        "passed": True,
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ready_for_sequential_training": True,
        "seeds": list(SEEDS),
        "audio_payload_read": False,
        "optimizer_step_exercised": False,
        "formal_training_started": False,
        "unknown_tune_read": False,
        "known_holdout_read": False,
        "unknown_holdout_read": False,
        "locked_datasets_read": [],
        "output_directories_empty": True,
        "inputs": {
            "preflight_implementation_sha256": file_sha256(Path(__file__)),
            "training_implementation_sha256": file_sha256(training_module),
            "seed_configs": {
                str(seed): {
                    "path": config_paths[seed].relative_to(root).as_posix(),
                    "sha256": file_sha256(config_paths[seed]),
                }
                for seed in SEEDS
            },
            "replication_authorization_sha256": str(
                configs[43]["inputs"]["replication_authorization"]["sha256"]
            ),
        },
    }
    output_dir = root / "artifacts/g18_model_identification/p4_preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def validate_existing(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    report_path = (
        root / "artifacts/g18_model_identification/p4_preflight/report.json"
    )
    if not report_path.is_file():
        raise FileNotFoundError("G18 P4 preflight report is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config_paths, configs = _validate_configs(root)
    training_module = Path(__file__).with_name("train_g18_model_id.py")
    if not (
        report.get("passed") is True
        and report.get("ready_for_sequential_training") is True
        and report.get("audio_payload_read") is False
        and report.get("formal_training_started") is False
        and report.get("unknown_tune_read") is False
        and report.get("known_holdout_read") is False
        and report.get("unknown_holdout_read") is False
        and report.get("inputs", {}).get("preflight_implementation_sha256")
        == file_sha256(Path(__file__))
        and report.get("inputs", {}).get("training_implementation_sha256")
        == file_sha256(training_module)
        and all(
            report["inputs"]["seed_configs"][str(seed)]["sha256"]
            == file_sha256(config_paths[seed])
            for seed in SEEDS
        )
        and report["inputs"]["replication_authorization_sha256"]
        == str(configs[43]["inputs"]["replication_authorization"]["sha256"])
    ):
        raise ValueError("G18 P4 preflight report is invalid or stale")
    print(json.dumps({"passed": True, "preflight_valid": True, "seeds": list(SEEDS)}))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight G18 seeds 43 and 44.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()
    if args.validate_existing:
        validate_existing(args.root)
    else:
        preflight(args.root)


if __name__ == "__main__":
    main()
