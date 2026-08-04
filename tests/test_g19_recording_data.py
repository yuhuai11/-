from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from dads_crnn.g19_recording_data import (
    RecordingExample,
    RecordingFeatureDataset,
    assert_disjoint_recordings,
    balanced_recording_batches,
    build_recording_examples,
    collate_recordings,
)


def _row(
    audio_sha256: str,
    segment_index: int | str,
    target: int,
    model_id: str,
    is_known: bool | str = True,
) -> dict[str, object]:
    return {
        "audio_sha256": audio_sha256,
        "segment_index": segment_index,
        "target_index": target,
        "model_id": model_id,
        "is_known": is_known,
    }


def _example(index: int, target: int) -> RecordingExample:
    indices = np.asarray([index], dtype=np.int64)
    indices.flags.writeable = False
    return RecordingExample(
        indices=indices,
        target=target,
        model_id=f"MODEL_{target}",
        audio_sha256=f"{target}-{index}",
        is_known=True,
    )


class G19RecordingGroupingTests(unittest.TestCase):
    def test_groups_by_hash_and_sorts_segment_index_numerically(self) -> None:
        frame = pd.DataFrame(
            [
                _row("B", "10", 1, "B_MODEL"),
                _row("a", "10", 0, "A_MODEL"),
                _row("A", "2", 0, "A_MODEL"),
                _row("b", "1", 1, "B_MODEL"),
                _row("a", "1", 0, "A_MODEL"),
            ]
        )
        examples = build_recording_examples(frame)

        self.assertEqual(
            [example.audio_sha256 for example in examples], ["a", "b"]
        )
        np.testing.assert_array_equal(examples[0].indices, [4, 2, 1])
        np.testing.assert_array_equal(examples[1].indices, [3, 0])
        self.assertEqual(examples[0].target, 0)
        self.assertEqual(examples[0].model_id, "A_MODEL")
        self.assertTrue(examples[0].is_known)
        self.assertFalse(examples[0].indices.flags.writeable)

    def test_rejects_conflicting_recording_metadata(self) -> None:
        variants = {
            "target_index": [
                _row("a", 0, 0, "A"),
                _row("a", 1, 1, "A"),
            ],
            "model_id": [
                _row("a", 0, 0, "A"),
                _row("a", 1, 0, "B"),
            ],
            "is_known": [
                _row("a", 0, 0, "A", True),
                _row("a", 1, 0, "A", False),
            ],
        }
        for field, rows in variants.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, f"conflicting {field}"):
                    build_recording_examples(pd.DataFrame(rows))

    def test_rejects_non_numeric_or_duplicate_segment_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "segment_index"):
            build_recording_examples(
                pd.DataFrame(
                    [_row("a", 0, 0, "A"), _row("a", "second", 0, "A")]
                )
            )
        with self.assertRaisesRegex(ValueError, "duplicate segment_index"):
            build_recording_examples(
                pd.DataFrame(
                    [_row("a", 0, 0, "A"), _row("a", "0", 0, "A")]
                )
            )


class G19RecordingDatasetTests(unittest.TestCase):
    def test_dataset_and_collate_preserve_order_and_pad_with_mask(self) -> None:
        frame = pd.DataFrame(
            [
                _row("b", 1, 1, "B"),
                _row("a", 2, 0, "A"),
                _row("a", 0, 0, "A"),
            ]
        )
        features = np.asarray(
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
            dtype=np.float32,
        )
        dataset = RecordingFeatureDataset(frame, features)

        first_sequence, first_example = dataset[0]
        torch.testing.assert_close(
            first_sequence,
            torch.tensor([[30.0, 31.0], [20.0, 21.0]]),
        )
        self.assertEqual(first_example.audio_sha256, "a")

        padded, mask, targets, metadata = collate_recordings(
            [dataset[0], dataset[1]]
        )
        self.assertEqual(tuple(padded.shape), (2, 2, 2))
        self.assertEqual(mask.dtype, torch.bool)
        torch.testing.assert_close(
            mask, torch.tensor([[True, True], [True, False]])
        )
        torch.testing.assert_close(padded[1, 1], torch.zeros(2))
        torch.testing.assert_close(targets, torch.tensor([0, 1]))
        self.assertEqual(
            [example.audio_sha256 for example in metadata], ["a", "b"]
        )

    def test_dataset_rejects_feature_row_misalignment(self) -> None:
        frame = pd.DataFrame([_row("a", 0, 0, "A")])
        with self.assertRaisesRegex(ValueError, "not aligned"):
            RecordingFeatureDataset(
                frame, np.zeros((2, 3), dtype=np.float32)
            )


class G19BalancedRecordingBatchTests(unittest.TestCase):
    def test_batches_are_deterministic_balanced_and_cycle_minority(self) -> None:
        examples = tuple(
            [_example(index, 0) for index in range(5)]
            + [_example(index, 1) for index in range(5, 7)]
        )
        first = balanced_recording_batches(
            examples, recordings_per_class=2, seed=42, epoch=3
        )
        second = balanced_recording_batches(
            examples, recordings_per_class=2, seed=42, epoch=3
        )

        self.assertEqual(len(first), 3)
        for left, right in zip(first, second, strict=True):
            np.testing.assert_array_equal(left, right)
        for batch in first:
            self.assertEqual(len(batch), len(set(batch.tolist())))
            targets = [examples[int(index)].target for index in batch]
            self.assertEqual(targets.count(0), 2)
            self.assertEqual(targets.count(1), 2)
        majority_seen = {
            int(index)
            for batch in first
            for index in batch
            if examples[int(index)].target == 0
        }
        self.assertEqual(majority_seen, set(range(5)))

    def test_requires_two_distinct_recordings_per_class(self) -> None:
        examples = (_example(0, 0), _example(1, 1), _example(2, 1))
        with self.assertRaisesRegex(ValueError, "at least 2"):
            balanced_recording_batches(
                examples, recordings_per_class=1, seed=0, epoch=0
            )
        with self.assertRaisesRegex(ValueError, "Each class needs"):
            balanced_recording_batches(
                examples, recordings_per_class=2, seed=0, epoch=0
            )

    def test_rejects_duplicate_recording_identity(self) -> None:
        examples = (
            _example(0, 0),
            RecordingExample(
                indices=np.asarray([1]),
                target=0,
                model_id="MODEL_0",
                audio_sha256="0-0",
                is_known=True,
            ),
            _example(2, 1),
            _example(3, 1),
        )
        with self.assertRaisesRegex(ValueError, "unique recording"):
            balanced_recording_batches(
                examples, recordings_per_class=2, seed=0, epoch=0
            )


class G19RecordingFirewallTests(unittest.TestCase):
    def test_disjoint_splits_return_zero_overlap_audit(self) -> None:
        audit = assert_disjoint_recordings(
            {
                "known_train": pd.DataFrame({"audio_sha256": ["a", "a"]}),
                "known_tune": pd.DataFrame({"audio_sha256": ["b"]}),
                "unknown_tune": pd.DataFrame({"audio_sha256": ["c"]}),
            }
        )
        self.assertEqual(
            audit,
            {
                "known_train__known_tune": 0,
                "known_train__unknown_tune": 0,
                "known_tune__unknown_tune": 0,
            },
        )

    def test_rejects_cross_split_and_consumed_hash_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "recording overlap"):
            assert_disjoint_recordings(
                {
                    "known_train": pd.DataFrame({"audio_sha256": ["A"]}),
                    "known_tune": pd.DataFrame({"audio_sha256": ["a"]}),
                }
            )
        with self.assertRaisesRegex(ValueError, "Consumed recording hash"):
            assert_disjoint_recordings(
                {"known_train": pd.DataFrame({"audio_sha256": ["a"]})},
                forbidden_hashes={"A"},
            )


if __name__ == "__main__":
    unittest.main()
