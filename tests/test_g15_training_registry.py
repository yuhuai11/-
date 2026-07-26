from __future__ import annotations

import unittest

import pandas as pd

from dads_crnn.prepare_g15_training_registry import build_registry


def dads_frame(split: str = "train", audio_hash: str = "a" * 64) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": [split],
            "label": [0],
            "cache_path": ["dads.npy"],
            "raw_audio_sha256": [audio_hash],
            "recording_group": ["dads-group"],
        }
    )


def g9_frame(split: str = "train") -> pd.DataFrame:
    classes = ["chainsaw", "engine", "vacuum_cleaner", "washing_machine"]
    return pd.DataFrame(
        {
            "split": [split] * 4,
            "label": [0] * 4,
            "cache_path": [f"g9-{index}.npy" for index in range(4)],
            "sha256": [f"{index + 10:064x}" for index in range(4)],
            "cache_sha256": [f"{index + 20:064x}" for index in range(4)],
            "recording_group": [f"recording-{index}" for index in range(4)],
            "source_group": [f"source-{index}" for index in range(4)],
            "hard_negative_class": classes,
            "background_mix_eligible": [False] * 4,
        }
    )


def g14_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["kielce_17_uav", "tau_urban_2022"],
            "split": ["train", "train"],
            "label": [1, 0],
            "source_group": ["kielce:one", "tau:one"],
            "cache_path": ["train.npy", "train.npy"],
            "cache_index": [0, 1],
            "audio_sha256": ["b" * 64, "c" * 64],
            "segment_sha256": ["d" * 64, "e" * 64],
        }
    )


class G15TrainingRegistryTests(unittest.TestCase):
    def test_builds_four_sources_with_fixed_permissions(self) -> None:
        combined, sources, overlaps = build_registry(
            dads_frame(),
            g9_frame(),
            g14_frame(),
            g9_classes={
                "chainsaw",
                "engine",
                "vacuum_cleaner",
                "washing_machine",
            },
            forbidden_hashes=frozenset(),
        )
        self.assertEqual(
            set(sources),
            {
                "dads_replay",
                "g9_mechanical_hard_negative",
                "kielce_uav",
                "tau_background",
            },
        )
        self.assertEqual(len(combined), 7)
        g9 = sources["g9_mechanical_hard_negative"]
        self.assertFalse(g9["background_mix_eligible"].any())
        self.assertTrue(g9["teacher_distillation_eligible"].all())
        self.assertTrue(sources["kielce_uav"]["paired_ranking_eligible"].all())
        self.assertTrue(all(value == 0 for value in overlaps.values()))

    def test_rejects_nontraining_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "only the train split"):
            build_registry(
                dads_frame(split="test"),
                g9_frame(),
                g14_frame(),
                g9_classes={
                    "chainsaw",
                    "engine",
                    "vacuum_cleaner",
                    "washing_machine",
                },
                forbidden_hashes=frozenset(),
            )

    def test_rejects_consumed_hash_after_rename(self) -> None:
        consumed = "a" * 64
        with self.assertRaisesRegex(ValueError, "consumed G13"):
            build_registry(
                dads_frame(audio_hash=consumed),
                g9_frame(),
                g14_frame(),
                g9_classes={
                    "chainsaw",
                    "engine",
                    "vacuum_cleaner",
                    "washing_machine",
                },
                forbidden_hashes=frozenset({consumed}),
            )

    def test_rejects_cross_source_audio_overlap(self) -> None:
        g14 = g14_frame()
        g14.loc[0, "audio_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "Cross-source audio overlap"):
            build_registry(
                dads_frame(),
                g9_frame(),
                g14,
                g9_classes={
                    "chainsaw",
                    "engine",
                    "vacuum_cleaner",
                    "washing_machine",
                },
                forbidden_hashes=frozenset(),
            )


if __name__ == "__main__":
    unittest.main()
