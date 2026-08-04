from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import RandomSampler

from dads_crnn.dataset import DADSDataset
from dads_crnn import prepare_g9_hard_negatives as g9_prepare
from dads_crnn import train_panns
from dads_crnn.prepare_g9_hard_negatives import (
    CandidateClip,
    FIXED_CLASS_MAP,
    _candidate_inventory,
    deterministic_group_split,
    parse_esc50_filename,
    prepare,
    reject_locked_path,
)
from dads_crnn.train import _build_loaders, _training_criterion


ROOT = Path(__file__).resolve().parents[1]


def _write_pcm16(path: Path, value: int = 1000, samples: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.full(samples, value, dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(audio.tobytes())


def _cache(path: Path, value: float) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.full(16, value, dtype=np.float32), allow_pickle=False)
    return path.as_posix()


class G9FilenameAndSplitTests(unittest.TestCase):
    def test_parser_decodes_class_segment_and_merges_ab_provenance(self) -> None:
        first = parse_esc50_filename(
            Path("1-100210-A-360.wav"), FIXED_CLASS_MAP
        )
        second = parse_esc50_filename(
            Path("1-100210-B-364.wav"), FIXED_CLASS_MAP
        )
        self.assertEqual(first.fold, 1)
        self.assertEqual(first.clip_id, "100210")
        self.assertEqual(first.class_id, 36)
        self.assertEqual(first.class_name, "vacuum_cleaner")
        self.assertEqual(first.segment_index, 0)
        self.assertEqual(first.recording_group, "1-100210-A-36")
        self.assertEqual(second.recording_group, "1-100210-B-36")
        self.assertEqual(first.source_group, "1-100210")
        self.assertEqual(second.source_group, first.source_group)
        self.assertEqual(second.segment_index, 4)

    def test_parser_rejects_malformed_or_unregistered_class(self) -> None:
        with self.assertRaisesRegex(ValueError, "Malformed"):
            parse_esc50_filename(Path("1-100210-A-365.wav"), FIXED_CLASS_MAP)
        with self.assertRaisesRegex(ValueError, "not in the frozen"):
            parse_esc50_filename(Path("1-100210-A-420.wav"), FIXED_CLASS_MAP)

    def test_official_fold_split_is_fixed_and_order_independent(self) -> None:
        groups = {
            "1-101": 1,
            "2-202": 2,
            "3-303": 3,
            "4-404": 4,
            "5-505": 5,
        }
        expected = deterministic_group_split(
            groups,
            class_name="helicopter",
            train_folds=[1, 2, 3, 4],
            guard_folds=[5],
        )
        observed = deterministic_group_split(
            dict(reversed(list(groups.items()))),
            class_name="helicopter",
            train_folds=[4, 3, 2, 1],
            guard_folds=[5],
        )
        self.assertEqual(expected, observed)
        self.assertEqual(expected["5-505"], "guard")
        self.assertTrue(all(expected[key] == "train" for key in groups if key != "5-505"))

    def test_nonofficial_fold_policy_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "official ESC-50 folds"):
            deterministic_group_split(
                {"1-a": 1, "5-b": 5},
                class_name="engine",
                train_folds=[1, 2, 3, 5],
                guard_folds=[4],
            )


class G9IsolationTests(unittest.TestCase):
    def test_locked_names_and_symlink_targets_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "UNSEEN.csv",
                "real_world.csv",
                "real-world.csv",
                "raw_recorded_audios.csv",
                "external_confirmation_v2.csv",
                "g13_external_confirmation.csv",
            ):
                with self.assertRaises(ValueError):
                    reject_locked_path(root / name)
            locked = root / "real_world_manifest.csv"
            locked.write_text("x\n", encoding="utf-8")
            alias = root / "safe.csv"
            alias.symlink_to(locked)
            with self.assertRaises(ValueError):
                reject_locked_path(alias)

    def test_duplicate_selected_wav_hashes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "candidate_audio"
            # One frozen class has eight original ESC recordings in each of
            # five official folds. Each recording is exported as five clips.
            # Identical payloads deliberately trigger the SHA duplicate gate.
            for fold in range(1, 6):
                for recording in range(8):
                    clip_id = fold * 1000 + recording
                    for segment in range(5):
                        _write_pcm16(
                            source / f"{fold}-{clip_id}-A-40{segment}.wav",
                            value=1000,
                        )
            config = {
                "source": {
                    "sample_rate": 16000,
                    "minimum_rms_dbfs_exclusive": -65.0,
                    "expected_groups_per_class": 40,
                    "expected_segments_per_group": 5,
                }
            }
            with self.assertRaisesRegex(ValueError, "Duplicate selected"):
                _candidate_inventory(source, config, {40: "helicopter"})

    def test_candidate_dads_hash_overlap_aborts_before_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "1-123-A-400.wav"
            _write_pcm16(source_path)
            parsed = parse_esc50_filename(source_path, FIXED_CLASS_MAP)
            clip = CandidateClip(
                path=source_path.resolve(),
                parsed=parsed,
                sha256="a" * 64,
                sample_rate=16000,
                samples=64,
                rms_dbfs=-20.0,
                audio=np.ones(64, dtype=np.float32),
            )
            config = yaml.safe_load(
                (ROOT / "configs/g9_hard_negative_data.yaml").read_text(
                    encoding="utf-8"
                )
            )
            config["source"]["root"] = (root / "source").as_posix()
            config["dads"]["parquet_dir"] = (root / "parquet").as_posix()
            config["val_ood"]["tune_manifest"] = (root / "tune.csv").as_posix()
            config["val_ood"]["holdout_manifest"] = (root / "diagnostic.csv").as_posix()
            config["val_ood"]["external_repo_root"] = (root / "external").as_posix()
            config["output_dir"] = (root / "output").as_posix()
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            dads_manifest = root / "dads.csv"
            dads_manifest.write_text("split,label\ntrain,0\n", encoding="utf-8")
            val_inventory = {
                "canonical_paths": set(),
                "source_values": set(),
                "source_groups": set(),
                "sample_hashes": set(),
                "raw_source_hashes": set(),
                "audit": {},
            }
            dads_inventory = {
                "frame": pd.DataFrame(),
                "hashes": {clip.sha256},
                "source_values": set(),
                "audit": {},
            }
            with (
                patch.object(
                    g9_prepare,
                    "_candidate_inventory",
                    return_value=([clip], {}),
                ),
                patch.object(
                    g9_prepare, "_val_ood_inventory", return_value=val_inventory
                ),
                patch.object(
                    g9_prepare, "_dads_inventory", return_value=dads_inventory
                ),
                patch.object(g9_prepare, "_build_outputs") as build_outputs,
            ):
                with self.assertRaisesRegex(ValueError, "isolation failed"):
                    prepare(config_path, dads_manifest)
            build_outputs.assert_not_called()
            self.assertFalse((root / "output").exists())

    def test_same_upstream_clip_id_cannot_cross_official_folds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "1-777-A-400.wav"
            guard_path = root / "5-777-A-400.wav"
            _write_pcm16(train_path, value=900)
            _write_pcm16(guard_path, value=1100)
            train_parsed = parse_esc50_filename(train_path, FIXED_CLASS_MAP)
            guard_parsed = parse_esc50_filename(guard_path, FIXED_CLASS_MAP)
            clips = [
                CandidateClip(
                    path=train_path,
                    parsed=train_parsed,
                    sha256="1" * 64,
                    sample_rate=16000,
                    samples=64,
                    rms_dbfs=-20.0,
                    audio=np.full(64, 0.1, dtype=np.float32),
                ),
                CandidateClip(
                    path=guard_path,
                    parsed=guard_parsed,
                    sha256="2" * 64,
                    sample_rate=16000,
                    samples=64,
                    rms_dbfs=-20.0,
                    audio=np.full(64, 0.2, dtype=np.float32),
                ),
            ]
            assignment = {
                ("helicopter", train_parsed.source_group): "train",
                ("helicopter", guard_parsed.source_group): "guard",
            }
            with self.assertRaisesRegex(ValueError, "clip ID leaked"):
                g9_prepare._build_outputs(
                    stage=root / "stage",
                    output_dir=root / "final",
                    clips=clips,
                    group_splits=assignment,
                    dads_frame=pd.DataFrame(),
                    target_samples=64,
                )


class G9DatasetRegressionTests(unittest.TestCase):
    def _dataset(self, root: Path, rows: list[dict]) -> DADSDataset:
        manifest = root / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        return DADSDataset(
            manifest,
            "train",
            sample_rate=16,
            clip_seconds=1.0,
            training=False,
            seed=42,
        )

    def test_legacy_manifest_keeps_all_dads_negatives_in_background_mixer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {
                    "split": "train",
                    "label": label,
                    "source_path": f"source-{index}.wav",
                    "cache_path": _cache(root / f"{index}.npy", index + 1),
                }
                for index, label in enumerate((0, 0, 1))
            ]
            dataset = self._dataset(root, rows)
            self.assertEqual(dataset.negative_indices.tolist(), [0, 1])

    def test_hard_negatives_train_normally_but_never_enter_background_mixer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {
                    "split": "train",
                    "label": 0,
                    "source_path": "dads-negative.wav",
                    "cache_path": _cache(root / "dads-negative.npy", 1.0),
                    "dataset_origin": "dads",
                    "segment_kind": "full",
                    "background_mix_eligible": True,
                },
                {
                    "split": "train",
                    "label": 0,
                    "source_path": "hn-negative.wav",
                    "cache_path": _cache(root / "hn-negative.npy", 2.0),
                    "dataset_origin": "g9_hard_negative",
                    "segment_kind": "hard_negative",
                    "background_mix_eligible": False,
                },
                {
                    "split": "train",
                    "label": 1,
                    "source_path": "positive.wav",
                    "cache_path": _cache(root / "positive.npy", 3.0),
                    "dataset_origin": "dads",
                    "segment_kind": "full",
                    "background_mix_eligible": True,
                },
            ]
            dataset = self._dataset(root, rows)
            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset.negative_indices.tolist(), [0])
            _, label = dataset[1]
            self.assertEqual(float(label), 0.0)

    def test_malformed_background_mix_flag_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {
                    "split": "train",
                    "label": 0,
                    "source_path": "hn-negative.wav",
                    "cache_path": _cache(root / "hn-negative.npy", 1.0),
                    "background_mix_eligible": "yes",
                }
            ]
            with self.assertRaisesRegex(ValueError, "true/false"):
                self._dataset(root, rows)


class G9TrainingContractTests(unittest.TestCase):
    def test_g9_config_delta_is_only_data_audit_and_output(self) -> None:
        g7 = yaml.safe_load(
            (ROOT / "configs/g7_panns_cnn14_16k_pt.yaml").read_text(encoding="utf-8")
        )
        g9 = yaml.safe_load(
            (ROOT / "configs/g9_panns_cnn14_16k_hn_bce.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(g9["train"]["pos_weight"], "auto")
        self.assertNotIn("sampling", g9["train"])
        self.assertEqual(g9["train"], g7["train"])
        self.assertEqual(g9["model"], g7["model"])
        self.assertEqual(g9["features"], g7["features"])
        self.assertEqual(g9["eval"], g7["eval"])
        self.assertNotEqual(g9["output_dir"], g7["output_dir"])
        audit_path = g9["data"].pop("g9_audit_path")
        self.assertEqual(audit_path, "artifacts/g9_hard_negatives/audit.json")
        self.assertEqual(g9["data"], g7["data"])

    def test_loader_uses_natural_shuffle_and_plain_bce(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for split_index, split in enumerate(("train", "val", "test")):
                for label in (0, 1):
                    rows.append(
                        {
                            "split": split,
                            "label": label,
                            "source_path": f"{split}-{label}.wav",
                            "cache_path": _cache(
                                root / f"{split}-{label}.npy",
                                split_index + label + 1.0,
                            ),
                        }
                    )
            manifest = root / "manifest.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            config = {
                "data": {"sample_rate": 16, "clip_seconds": 1.0},
                "train": {
                    "batch_size": 2,
                    "num_workers": 0,
                    "pos_weight": "auto",
                },
            }
            train_loader, _, _ = _build_loaders(config, manifest, 42)
            self.assertIsInstance(train_loader.sampler, RandomSampler)
            self.assertFalse(train_loader.sampler.replacement)
            self.assertEqual(train_loader.sampler.num_samples, len(train_loader.dataset))
            criterion = _training_criterion(config, train_loader, torch.device("cpu"))
            self.assertIsInstance(criterion, nn.BCEWithLogitsLoss)

    def test_manifest_and_audit_hash_binding_and_resume_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "combined.csv"
            manifest.write_text("split,label\ntrain,0\n", encoding="utf-8")
            audit = root / "audit.json"
            audit_payload = {
                "passed": True,
                "locked_datasets_read": [],
                "outputs": {
                    "combined_manifest": {
                        "path": manifest.resolve().as_posix(),
                        "sha256": train_panns.file_sha256(manifest),
                        "rows": 1,
                    }
                },
            }
            audit.write_text(json.dumps(audit_payload), encoding="utf-8")
            config = {"data": {"g9_audit_path": audit.as_posix()}}
            identity = train_panns._training_input_identity(config, manifest)
            self.assertEqual(identity["manifest_rows"], 1)
            self.assertEqual(identity["g9_audit_path"], audit.resolve().as_posix())
            self.assertEqual(identity["g9_audit_sha256"], train_panns.file_sha256(audit))

            train_panns._verify_checkpoint_inputs(
                {"training_inputs": identity},
                identity,
                require_identity=True,
                artifact="test checkpoint",
            )
            with self.assertRaisesRegex(
                ValueError, "lacks the required training input identity"
            ):
                train_panns._verify_checkpoint_inputs(
                    {}, identity, require_identity=True, artifact="old checkpoint"
                )
            changed = dict(identity)
            changed["manifest_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "does not match"):
                train_panns._verify_checkpoint_inputs(
                    {"training_inputs": identity},
                    changed,
                    require_identity=True,
                    artifact="resume checkpoint",
                )

            manifest.write_text("split,label\ntrain,0\ntrain,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 does not match"):
                train_panns._training_input_identity(config, manifest)

    def test_stage1_script_has_no_external_evaluation_or_implicit_resume(self) -> None:
        script = (ROOT / "scripts/run_g9_hn_stage1.sh").read_text(encoding="utf-8")
        self.assertIn("prepare_g9_hard_negatives", script)
        self.assertIn("--preflight-only", script)
        self.assertIn("dads_crnn.train_panns", script)
        self.assertNotIn("dads_crnn.calibrate_ood", script)
        self.assertNotIn("dads_crnn.evaluate", script)
        self.assertNotIn("      --resume", script)


if __name__ == "__main__":
    unittest.main()
