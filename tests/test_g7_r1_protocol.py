from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dads_crnn.audit_g7_r1 import (
    EXPECTED_ALLOWED_CONFIG_DIFFERENCES,
    EXPECTED_PROMOTION_GATES,
    EXPECTED_SCORE_PROTOCOL,
    EXPECTED_STRUCTURED_RANGES,
    EXPECTED_TRAINING_MANIFEST_PATH,
    EXPECTED_TRAINING_MANIFEST_SHA256,
    _audit_r1_manifest_paths,
    _validate_score_protocol,
    _validate_single_variable_contract,
    _validate_training_manifest_identity,
    config_differences,
)
from dads_crnn.config import load_config


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/g7_r1_protocol.yaml"


def _configs() -> tuple[dict, dict, dict]:
    protocol = load_config(PROTOCOL_PATH)
    baseline = load_config(ROOT / protocol["inputs"]["baseline_config"])
    candidate = load_config(ROOT / protocol["inputs"]["candidate_config"])
    return protocol, baseline, candidate


class G7R1ProtocolTests(unittest.TestCase):
    def test_observed_delta_exactly_matches_the_fixed_allowlist(self) -> None:
        protocol, baseline, candidate = _configs()
        observed = config_differences(baseline, candidate)
        allowed = sorted(protocol["contract"]["allowed_config_differences"])

        self.assertEqual(tuple(allowed), EXPECTED_ALLOWED_CONFIG_DIFFERENCES)
        self.assertEqual(observed, allowed)

        audit = _validate_single_variable_contract(
            protocol, baseline, candidate
        )
        self.assertEqual(audit["observed_differences"], allowed)

    def test_probability_seed_ranges_and_promotion_gates_are_frozen(self) -> None:
        protocol, baseline, candidate = _configs()
        audit = _validate_single_variable_contract(
            protocol, baseline, candidate
        )

        self.assertEqual(audit["frequency_response_probability"], 0.50)
        self.assertEqual(
            baseline["train"]["augmentation"]["frequency_response_probability"],
            0.50,
        )
        self.assertEqual(
            candidate["train"]["augmentation"]["frequency_response_probability"],
            0.50,
        )
        self.assertEqual(audit["training_seeds"], [42])
        self.assertEqual(
            audit["structured_ranges"], EXPECTED_STRUCTURED_RANGES
        )
        self.assertEqual(audit["promotion_gates"], EXPECTED_PROMOTION_GATES)

    def test_per_model_tune_calibration_and_g7_diagnostic_mapping_are_frozen(
        self,
    ) -> None:
        protocol, _, _ = _configs()
        low_fpr_path = ROOT / protocol["inputs"]["baseline_low_fpr_metrics"]
        r0_metrics_path = ROOT / protocol["inputs"]["r0_idmt_metrics"]
        low_fpr = json.loads(low_fpr_path.read_text(encoding="utf-8"))
        r0_metrics = json.loads(r0_metrics_path.read_text(encoding="utf-8"))

        observed = _validate_score_protocol(protocol, low_fpr, r0_metrics)

        self.assertEqual(observed, EXPECTED_SCORE_PROTOCOL)
        self.assertFalse(
            observed["holdout_used"]
        )
        self.assertFalse(observed["idmt_used"])
        self.assertEqual(
            observed["primary_comparison"],
            "per_model_same_tune_calibration",
        )
        self.assertEqual(
            observed["fixed_g7_mapping_role"],
            "deployment_compatibility_diagnostic_only",
        )

    def test_probability_seed_range_gate_or_score_drift_fails_closed(self) -> None:
        protocol, baseline, candidate = _configs()
        mutations = []

        probability = copy.deepcopy(candidate)
        probability["train"]["augmentation"]["frequency_response_probability"] = 0.49
        mutations.append(("probability", protocol, probability))

        seed = copy.deepcopy(candidate)
        seed["train"]["seeds"] = [42, 43]
        mutations.append(("seed", protocol, seed))

        frequency_range = copy.deepcopy(candidate)
        frequency_range["train"]["augmentation"]["microphone_highpass_hz"] = [
            50.0,
            120.0,
        ]
        mutations.append(("range", protocol, frequency_range))

        missing_gate_protocol = copy.deepcopy(protocol)
        del missing_gate_protocol["promotion_gates"][
            "g9_mechanical_guard_at_probability_0_5"
        ]
        mutations.append(("promotion_gate", missing_gate_protocol, candidate))

        for name, mutated_protocol, mutated_candidate in mutations:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    _validate_single_variable_contract(
                        mutated_protocol, baseline, mutated_candidate
                    )

        refit_protocol = copy.deepcopy(protocol)
        refit_protocol["contract"]["score_protocol"]["idmt_used"] = True
        with self.assertRaisesRegex(ValueError, "preregistered policy"):
            _validate_score_protocol(refit_protocol, {}, {})

    def test_manifest_firewall_rejects_locked_tokens_and_symlink_targets(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            safe = root / "safe"
            safe.mkdir()
            manifest = safe / "manifest.csv"

            for locked_value in (
                "Hohenwarte/audio.wav",
                "final_holdout/audio.wav",
                "FINAL-HOLDOUT/audio.wav",
            ):
                with self.subTest(locked_value=locked_value):
                    manifest.write_text(
                        "split,label,path\n"
                        f"train,0,{locked_value}\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        ValueError, "Locked final-test value"
                    ):
                        _audit_r1_manifest_paths(root, manifest)

            locked_root = root / "Hohenwarte"
            locked_root.mkdir()
            (locked_root / "audio.wav").write_bytes(b"not-opened")
            alias = root / "apparently_safe"
            alias.symlink_to(locked_root, target_is_directory=True)
            manifest.write_text(
                "split,label,path\n"
                "train,0,apparently_safe/audio.wav\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Locked final-test path"):
                _audit_r1_manifest_paths(root, manifest)

    def test_manifest_path_itself_cannot_resolve_into_hohenwarte(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            locked_root = root / "Hohenwarte"
            locked_root.mkdir()
            real_manifest = locked_root / "manifest.csv"
            real_manifest.write_text(
                "split,label,path\ntrain,0,safe.wav\n",
                encoding="utf-8",
            )
            alias = root / "safe_manifest.csv"
            alias.symlink_to(real_manifest)

            with self.assertRaisesRegex(ValueError, "Locked final-test path"):
                _audit_r1_manifest_paths(root, alias)

    def test_training_manifest_path_and_sha256_are_exactly_frozen(self) -> None:
        protocol, _, _ = _configs()
        manifest = ROOT / protocol["inputs"]["training_manifest"]

        observed = _validate_training_manifest_identity(ROOT, manifest)

        self.assertEqual(
            manifest.relative_to(ROOT).as_posix(),
            EXPECTED_TRAINING_MANIFEST_PATH,
        )
        self.assertEqual(observed, EXPECTED_TRAINING_MANIFEST_SHA256)

        with TemporaryDirectory(dir=ROOT) as directory:
            wrong_manifest = Path(directory) / "dads_all_seed42.csv"
            wrong_manifest.write_text(
                "split,label,path\ntrain,0,safe.wav\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "path changed"):
                _validate_training_manifest_identity(ROOT, wrong_manifest)


if __name__ == "__main__":
    unittest.main()
