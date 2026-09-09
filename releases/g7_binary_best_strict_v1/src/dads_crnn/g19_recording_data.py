from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations
from typing import Collection, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


RECORDING_COLUMNS = frozenset(
    {"audio_sha256", "segment_index", "target_index", "model_id", "is_known"}
)


@dataclass(frozen=True)
class RecordingExample:
    """One recording and the segment-feature rows that belong to it."""

    indices: np.ndarray
    target: int
    model_id: str
    audio_sha256: str
    is_known: bool


def _strict_boolean(value: object, *, context: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"{context} must be a true/false boolean, got {value!r}")


def _strict_integer(value: object, *, context: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context} must be an integer, got {value!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be an integer, got {value!r}") from error
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{context} must be an integer, got {value!r}")
    return int(numeric)


def _normalized_recording_hashes(
    frame: pd.DataFrame, *, context: str
) -> np.ndarray:
    if "audio_sha256" not in frame:
        raise ValueError(f"{context} is missing column 'audio_sha256'")
    hashes: list[str] = []
    for row_index, value in enumerate(frame["audio_sha256"]):
        if pd.isna(value):
            raise ValueError(
                f"{context} row {row_index} has an empty audio_sha256"
            )
        normalized = str(value).strip().lower()
        if not normalized:
            raise ValueError(
                f"{context} row {row_index} has an empty audio_sha256"
            )
        hashes.append(normalized)
    return np.asarray(hashes, dtype=object)


def build_recording_examples(frame: pd.DataFrame) -> tuple[RecordingExample, ...]:
    """Group segment rows into deterministic, metadata-consistent recordings.

    Recordings are ordered by normalized ``audio_sha256``.  Within a recording,
    row positions are ordered by the numeric value of ``segment_index`` (so,
    for example, segment ``"2"`` precedes ``"10"``).
    """

    missing = sorted(RECORDING_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Recording manifest is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Recording manifest is empty")

    hashes = _normalized_recording_hashes(frame, context="recording manifest")
    examples: list[RecordingExample] = []
    for audio_sha256 in sorted(set(hashes.tolist())):
        positions = np.flatnonzero(hashes == audio_sha256).astype(
            np.int64, copy=False
        )
        rows = frame.iloc[positions]

        segment_indices = np.asarray(
            [
                _strict_integer(
                    value,
                    context=(
                        f"recording {audio_sha256} segment_index at row "
                        f"{int(position)}"
                    ),
                )
                for position, value in zip(
                    positions, rows["segment_index"], strict=True
                )
            ],
            dtype=np.int64,
        )
        if np.any(segment_indices < 0):
            raise ValueError(
                f"Recording {audio_sha256} contains a negative segment_index"
            )
        if len(np.unique(segment_indices)) != len(segment_indices):
            raise ValueError(
                f"Recording {audio_sha256} contains duplicate segment_index values"
            )
        order = np.argsort(segment_indices, kind="stable")
        ordered_positions = positions[order].copy()
        ordered_positions.flags.writeable = False

        targets = {
            _strict_integer(
                value,
                context=f"recording {audio_sha256} target_index",
            )
            for value in rows["target_index"]
        }
        if len(targets) != 1:
            raise ValueError(
                f"Recording {audio_sha256} has conflicting target_index values"
            )

        model_ids: set[str] = set()
        for value in rows["model_id"]:
            if pd.isna(value) or not str(value).strip():
                raise ValueError(
                    f"Recording {audio_sha256} contains an empty model_id"
                )
            model_ids.add(str(value).strip())
        if len(model_ids) != 1:
            raise ValueError(
                f"Recording {audio_sha256} has conflicting model_id values"
            )

        known_values = {
            _strict_boolean(
                value,
                context=f"recording {audio_sha256} is_known",
            )
            for value in rows["is_known"]
        }
        if len(known_values) != 1:
            raise ValueError(
                f"Recording {audio_sha256} has conflicting is_known values"
            )

        examples.append(
            RecordingExample(
                indices=ordered_positions,
                target=targets.pop(),
                model_id=model_ids.pop(),
                audio_sha256=audio_sha256,
                is_known=known_values.pop(),
            )
        )
    return tuple(examples)


class RecordingFeatureDataset:
    """Expose a segment-feature matrix as variable-length recordings."""

    def __init__(
        self,
        frame: pd.DataFrame,
        features: np.ndarray | torch.Tensor,
        *,
        examples: Sequence[RecordingExample] | None = None,
    ) -> None:
        if features.ndim != 2:
            raise ValueError(
                "Segment features must be a 2-D [segments, dimensions] matrix"
            )
        if int(features.shape[0]) != len(frame):
            raise ValueError(
                "Segment features and manifest rows are not aligned: "
                f"{int(features.shape[0])} != {len(frame)}"
            )
        if int(features.shape[1]) <= 0:
            raise ValueError("Segment features must have a positive dimension")

        self.features = features
        self.examples = tuple(
            build_recording_examples(frame) if examples is None else examples
        )
        if not self.examples:
            raise ValueError("Recording dataset contains no recordings")
        recording_hashes = [
            example.audio_sha256.strip().lower() for example in self.examples
        ]
        if (
            any(not value for value in recording_hashes)
            or len(recording_hashes) != len(set(recording_hashes))
        ):
            raise ValueError(
                "Recording examples must have unique, non-empty audio_sha256 values"
            )

        expected_positions = np.concatenate(
            [
                np.asarray(example.indices, dtype=np.int64)
                for example in self.examples
            ]
        )
        if (
            len(expected_positions) != len(frame)
            or not np.array_equal(
                np.sort(expected_positions), np.arange(len(frame), dtype=np.int64)
            )
        ):
            raise ValueError(
                "Recording examples must cover every feature row exactly once"
            )

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, RecordingExample]:
        example = self.examples[index]
        positions = np.asarray(example.indices, dtype=np.int64)
        if isinstance(self.features, torch.Tensor):
            index_tensor = torch.as_tensor(
                positions, dtype=torch.long, device=self.features.device
            )
            sequence = self.features.index_select(0, index_tensor).to(
                dtype=torch.float32
            )
        else:
            selected = np.asarray(self.features[positions], dtype=np.float32)
            sequence = torch.from_numpy(selected)
        if not bool(torch.isfinite(sequence).all()):
            raise ValueError(
                f"Recording {example.audio_sha256} contains non-finite features"
            )
        return sequence, example


def collate_recordings(
    items: Sequence[tuple[torch.Tensor, RecordingExample]],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[RecordingExample, ...],
]:
    """Pad recording sequences and return ``features, mask, targets, metadata``."""

    if not items:
        raise ValueError("Cannot collate an empty recording batch")
    feature_dim = int(items[0][0].shape[-1]) if items[0][0].ndim == 2 else -1
    if feature_dim <= 0:
        raise ValueError("Recording features must be non-empty 2-D tensors")
    dtype = items[0][0].dtype
    device = items[0][0].device
    lengths: list[int] = []
    metadata: list[RecordingExample] = []
    for sequence, example in items:
        if sequence.ndim != 2 or int(sequence.shape[1]) != feature_dim:
            raise ValueError(
                "All recording sequences must have the same feature dimension"
            )
        if int(sequence.shape[0]) <= 0:
            raise ValueError("A recording sequence cannot be empty")
        if sequence.dtype != dtype or sequence.device != device:
            raise ValueError(
                "All recording sequences must share a dtype and device"
            )
        if int(sequence.shape[0]) != len(example.indices):
            raise ValueError(
                f"Recording {example.audio_sha256} sequence length is misaligned"
            )
        if not bool(torch.isfinite(sequence).all()):
            raise ValueError(
                f"Recording {example.audio_sha256} contains non-finite features"
            )
        lengths.append(int(sequence.shape[0]))
        metadata.append(example)

    padded = torch.zeros(
        (len(items), max(lengths), feature_dim), dtype=dtype, device=device
    )
    mask = torch.zeros(
        (len(items), max(lengths)), dtype=torch.bool, device=device
    )
    for row, ((sequence, _), length) in enumerate(
        zip(items, lengths, strict=True)
    ):
        padded[row, :length] = sequence
        mask[row, :length] = True
    targets = torch.tensor(
        [example.target for example in metadata],
        dtype=torch.long,
        device=device,
    )
    return padded, mask, targets, tuple(metadata)


def balanced_recording_batches(
    examples: Sequence[RecordingExample],
    *,
    recordings_per_class: int,
    seed: int,
    epoch: int,
) -> tuple[np.ndarray, ...]:
    """Build deterministic all-class recording batches.

    Every batch contains exactly ``recordings_per_class`` distinct recordings
    from every class.  The epoch length is set by the largest class.  Smaller
    classes are reshuffled and cycled between batches, but never duplicated
    within one batch.
    """

    if recordings_per_class < 2:
        raise ValueError("recordings_per_class must be at least 2 for SupCon")
    if not examples:
        raise ValueError("Cannot build batches from an empty recording set")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")

    by_class: dict[int, list[int]] = {}
    recording_hashes: set[str] = set()
    for index, example in enumerate(examples):
        if not example.is_known or example.target < 0:
            raise ValueError(
                "Balanced G19 representation batches accept known recordings only"
            )
        normalized_hash = example.audio_sha256.strip().lower()
        if not normalized_hash or normalized_hash in recording_hashes:
            raise ValueError(
                "Balanced G19 batches require unique recording audio_sha256 values"
            )
        recording_hashes.add(normalized_hash)
        by_class.setdefault(int(example.target), []).append(index)
    if len(by_class) < 2:
        raise ValueError("Balanced G19 batches require at least two classes")
    undersized = {
        target: len(indices)
        for target, indices in by_class.items()
        if len(indices) < recordings_per_class
    }
    if undersized:
        raise ValueError(
            "Each class needs at least recordings_per_class distinct "
            f"recordings, got {undersized}"
        )

    classes = sorted(by_class)
    batch_count = int(
        np.ceil(
            max(len(by_class[target]) for target in classes)
            / recordings_per_class
        )
    )
    seed_sequence = np.random.SeedSequence([int(seed), int(epoch), 19])
    class_sequences = seed_sequence.spawn(len(classes) + 1)
    class_rngs = {
        target: np.random.default_rng(class_sequences[position])
        for position, target in enumerate(classes)
    }
    batch_rng = np.random.default_rng(class_sequences[-1])
    queues: dict[int, deque[int]] = {target: deque() for target in classes}

    def take_distinct(target: int) -> list[int]:
        source = by_class[target]
        queue = queues[target]
        rng = class_rngs[target]
        chosen: list[int] = []
        chosen_set: set[int] = set()
        while len(chosen) < recordings_per_class:
            if not queue:
                queue.extend(int(value) for value in rng.permutation(source))
            candidate = queue.popleft()
            if candidate in chosen_set:
                queue.append(candidate)
                continue
            chosen.append(candidate)
            chosen_set.add(candidate)
        return chosen

    batches: list[np.ndarray] = []
    for _ in range(batch_count):
        batch = np.asarray(
            [
                index
                for target in classes
                for index in take_distinct(target)
            ],
            dtype=np.int64,
        )
        batch_rng.shuffle(batch)
        if len(batch) != len(set(batch.tolist())):
            raise RuntimeError("Internal error: duplicate recording in G19 batch")
        batches.append(batch)
    return tuple(batches)


def assert_disjoint_recordings(
    named_frames: Mapping[str, pd.DataFrame],
    *,
    forbidden_hashes: Collection[str] = (),
) -> dict[str, int]:
    """Fail closed on recording-hash leakage across splits or a firewall."""

    if not named_frames:
        raise ValueError("No recording splits were provided")
    hashes_by_name: dict[str, set[str]] = {}
    for name, frame in named_frames.items():
        if not str(name).strip():
            raise ValueError("Recording split names cannot be empty")
        if frame.empty:
            raise ValueError(f"Recording split {name!r} is empty")
        hashes_by_name[str(name)] = set(
            _normalized_recording_hashes(
                frame, context=f"recording split {name!r}"
            ).tolist()
        )

    normalized_forbidden = {
        str(value).strip().lower()
        for value in forbidden_hashes
        if str(value).strip()
    }
    for name, hashes in hashes_by_name.items():
        overlap = hashes & normalized_forbidden
        if overlap:
            raise ValueError(
                f"Consumed recording hash overlap in {name}: {len(overlap)}"
            )

    overlaps: dict[str, int] = {}
    for left, right in combinations(sorted(hashes_by_name), 2):
        overlap = hashes_by_name[left] & hashes_by_name[right]
        key = f"{left}__{right}"
        overlaps[key] = len(overlap)
        if overlap:
            raise ValueError(
                f"G19 recording overlap between {left} and {right}: "
                f"{len(overlap)}"
            )
    return overlaps
