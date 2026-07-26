from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml

from dads_crnn import train_panns
from dads_crnn.config import load_config


class G7PannsTests(unittest.TestCase):
    def test_history_is_written_as_atomic_csv(self) -> None:
        rows = [
            {
                "epoch": 1,
                "learning_rate": 0.0001,
                "train_loss": 0.2,
                "val_loss": 0.3,
                "val_accuracy": 0.8,
                "val_f1": 0.75,
                "val_auc": 0.9,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            train_panns._write_history(path, rows)
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_suffix(".csv.tmp").exists())
            contents = path.read_text(encoding="utf-8")
        self.assertIn("epoch,learning_rate,train_loss", contents)
        self.assertIn("1,0.0001,0.2,0.3,0.8,0.75,0.9", contents)

    def test_pt_and_scratch_configs_differ_only_as_registered(self) -> None:
        root = Path(__file__).resolve().parents[1]
        pt = yaml.safe_load((root / "configs/g7_panns_cnn14_16k_pt.yaml").read_text())
        scratch = yaml.safe_load(
            (root / "configs/g7_panns_cnn14_16k_scratch.yaml").read_text()
        )
        self.assertEqual(pt["model"]["initialization"], "audioset")
        self.assertEqual(scratch["model"]["initialization"], "scratch")
        self.assertNotEqual(pt["output_dir"], scratch["output_dir"])
        pt["model"]["initialization"] = "same"
        scratch["model"]["initialization"] = "same"
        pt["output_dir"] = "same"
        scratch["output_dir"] = "same"
        self.assertEqual(pt, scratch)

    def test_registered_batch_matches_paper_and_gpu_probe(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / "configs/g7_panns_cnn14_16k_pt.yaml").read_text())
        self.assertEqual(config["train"]["batch_size"], 128)
        self.assertTrue(config["train"]["mixed_precision"])
        self.assertFalse(config["model"]["spec_augment"])
        self.assertEqual(config["model"]["frontend_precision"], "float32")

    def test_preflight_rejects_cpu_before_loading_data(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs/g7_panns_cnn14_16k_pt.yaml")
        config["train"]["device"] = "cpu"
        with patch.object(train_panns, "_build_loaders") as build_loaders:
            with self.assertRaisesRegex(RuntimeError, "requires the server CUDA GPU"):
                train_panns.preflight(config, Path("unused.csv"), 42)
        build_loaders.assert_not_called()

    def test_epoch_resume_rng_state_is_weights_only_safe_and_restorable(self) -> None:
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        loader = SimpleNamespace(
            generator=torch.Generator().manual_seed(17),
            dataset=SimpleNamespace(
                rng=np.random.default_rng(17),
                augmenter=SimpleNamespace(rng=np.random.default_rng(18)),
            ),
        )
        state = train_panns._capture_rng_state(loader)
        expected = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
            float(loader.dataset.rng.random()),
            float(loader.dataset.augmenter.rng.random()),
            float(torch.rand((), generator=loader.generator)),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            train_panns._atomic_torch_save({"rng_state": state}, path)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(loaded["rng_state"]["torch_cpu"].device.type, "cpu")
        self.assertEqual(loaded["rng_state"]["loader_generator"].device.type, "cpu")
        train_panns._restore_rng_state(loaded["rng_state"], loader)
        observed = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
            float(loader.dataset.rng.random()),
            float(loader.dataset.augmenter.rng.random()),
            float(torch.rand((), generator=loader.generator)),
        )
        self.assertEqual(expected, observed)

    def test_training_identity_rejects_locked_tokens_inside_manifest_rows(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs/g7_panns_cnn14_16k_pt.yaml")
        for column, value in (
            ("source_path", "/datasets/Unseen/example.wav"),
            ("cache_path", "/datasets/real-world/example.npy"),
            ("background_source", "sources/REAL WORLD/recording.wav"),
            ("source_path", "/datasets/test_augmented/example.wav"),
            ("cache_path", "/datasets/raw_recorded_audios/example.npy"),
            ("source_path", "/datasets/external_confirmation_v2/UAV/example.wav"),
            ("cache_path", "/artifacts/g13_external_confirmation/example.npy"),
        ):
            with self.subTest(column=column, value=value):
                with tempfile.TemporaryDirectory() as directory:
                    manifest = Path(directory) / "manifest.csv"
                    manifest.write_text(
                        f"split,label,{column}\ntrain,0,{value}\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, "Locked final-test row"):
                        train_panns._training_input_identity(config, manifest)


if __name__ == "__main__":
    unittest.main()
