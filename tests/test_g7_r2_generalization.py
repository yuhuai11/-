from __future__ import annotations

import copy
import unittest

import numpy as np
import torch

from dads_crnn.augmentation import WaveformAugmenter
from dads_crnn.config import load_config
from dads_crnn.panns import RecordingMeanFrequencyMasking


def augmentation_config(**updates):
    config = {
        "positive_mix_probability": 1.0,
        "mix_snr_uniform_db": [10.0, 20.0],
        "frequency_response_probability": 0.0,
        "frequency_response_anchors": 8,
        "frequency_response_std_db": 3.0,
        "frequency_response_limit_db": 6.0,
        "reverb_probability": 0.0,
        "reverb_delay_ms": [12.0, 80.0],
        "reverb_gain": [0.05, 0.25],
        "colored_noise_probability": 0.0,
        "colored_noise_snr_db": [10.0, 30.0],
        "time_shift_probability": 0.0,
        "max_time_shift_ms": 0.0,
    }
    config.update(updates)
    return config


class G7R2WaveformAugmentationTests(unittest.TestCase):
    def test_uniform_snr_sampling_is_bounded_and_hits_requested_snr(self) -> None:
        augmenter = WaveformAugmenter(augmentation_config(), 16000, seed=42)
        time = np.arange(8000, dtype=np.float32) / 16000.0
        signal = np.sin(2.0 * np.pi * 220.0 * time).astype(np.float32)
        background = np.sin(2.0 * np.pi * 900.0 * time).astype(np.float32)
        observed = []
        for _ in range(32):
            _, metadata = augmenter.apply(signal, 1, lambda: background)
            self.assertTrue(metadata["background_mix"])
            self.assertAlmostEqual(
                metadata["achieved_snr_db"], metadata["target_snr_db"], places=5
            )
            observed.append(metadata["target_snr_db"])
        self.assertGreaterEqual(min(observed), 10.0)
        self.assertLessEqual(max(observed), 20.0)
        self.assertGreater(max(observed) - min(observed), 5.0)

    def test_uniform_and_discrete_snr_configuration_are_mutually_exclusive(self) -> None:
        config = augmentation_config(
            mix_snr_db=[10.0],
            mix_snr_weights=[1.0],
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            WaveformAugmenter(config, 16000, seed=42)


class G7R2FrequencyMaskingTests(unittest.TestCase):
    def test_frequency_masking_is_train_only_and_uses_recording_mean(self) -> None:
        module = RecordingMeanFrequencyMasking(
            probability=1.0,
            maximum_masks=3,
            maximum_width_fraction=0.15,
        )
        features = torch.arange(2 * 1 * 8 * 64, dtype=torch.float32).reshape(2, 1, 8, 64)
        module.eval()
        self.assertTrue(torch.equal(module(features), features))

        torch.manual_seed(42)
        module.train()
        masked = module(features)
        self.assertFalse(torch.equal(masked, features))
        for index in range(features.size(0)):
            fill = features[index].mean()
            changed = masked[index] != features[index]
            self.assertTrue(bool(changed.any()))
            self.assertTrue(torch.all(masked[index][changed] == fill))
            changed_bins = changed.any(dim=0).any(dim=0)
            self.assertLessEqual(int(changed_bins.sum()), 3 * int(np.floor(64 * 0.15)))

    def test_frequency_masking_rejects_invalid_ranges(self) -> None:
        with self.assertRaises(ValueError):
            RecordingMeanFrequencyMasking(
                probability=1.1,
                maximum_masks=3,
                maximum_width_fraction=0.15,
            )


class G7R2ConfigurationTests(unittest.TestCase):
    def test_ablation_sequence_changes_only_registered_components(self) -> None:
        control = load_config("configs/g7_r2_pt_control.yaml")
        mic = load_config("configs/g7_r2_pt_mic.yaml")
        mic_bg = load_config("configs/g7_r2_pt_mic_bg.yaml")
        full = load_config("configs/g7_r2_pt_mic_bg_freq.yaml")

        self.assertNotIn("augmentation", control["train"])
        self.assertEqual(mic["train"]["augmentation"]["positive_mix_probability"], 0.0)
        self.assertEqual(mic_bg["train"]["augmentation"]["positive_mix_probability"], 0.33)
        self.assertNotIn("frequency_masking", mic_bg["model"])
        self.assertTrue(full["model"]["frequency_masking"]["enabled"])

        for candidate in (mic, mic_bg, full):
            normalized = copy.deepcopy(candidate)
            normalized["output_dir"] = control["output_dir"]
            normalized["train"].pop("augmentation", None)
            normalized["model"].pop("frequency_masking", None)
            self.assertEqual(normalized, control)


if __name__ == "__main__":
    unittest.main()
