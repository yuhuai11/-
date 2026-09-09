from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler


class ClassSourceBalancedSampler(Sampler[int]):
    """Sample class, source, then segment uniformly (in expectation)."""

    def __init__(
        self,
        rows: pd.DataFrame,
        *,
        source_column: str,
        num_samples: int | None = None,
        seed: int,
    ) -> None:
        rows = rows.reset_index(drop=True)
        if source_column not in rows.columns:
            raise ValueError(f"Missing sampling source column: {source_column!r}")
        if rows[source_column].isna().any() or rows["label"].isna().any():
            raise ValueError("Balanced sampling requires non-null label and source values")
        labels = sorted(int(value) for value in rows["label"].unique())
        if labels != [0, 1]:
            raise ValueError(f"Binary class-source sampling requires labels [0, 1], got {labels}")

        weights = torch.empty(len(rows), dtype=torch.float64)
        source_counts: dict[int, int] = {}
        for label in labels:
            label_rows = rows[rows["label"] == label]
            counts = label_rows.groupby(source_column, sort=False)[source_column].transform("size")
            number_of_sources = int(label_rows[source_column].nunique())
            source_counts[label] = number_of_sources
            positions = torch.tensor(label_rows.index.to_numpy(copy=True), dtype=torch.long)
            weights[positions] = torch.tensor(
                1.0 / (len(labels) * number_of_sources * counts.to_numpy(copy=True)),
                dtype=torch.float64,
            )

        self.weights = weights
        self.num_samples = int(num_samples if num_samples is not None else len(rows))
        self.generator = torch.Generator().manual_seed(int(seed))
        self.source_column = source_column
        self.source_counts = source_counts
        self.expected_label_probabilities = {0: 0.5, 1: 0.5}

    def __iter__(self) -> Iterator[int]:
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=True,
            generator=self.generator,
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


class ClassSourceBalancedBatchSampler(Sampler[list[int]]):
    """Build exact class-balanced batches with tempered source balancing."""

    def __init__(
        self,
        rows: pd.DataFrame,
        *,
        source_column: str,
        batch_size: int,
        num_batches: int,
        source_weight_exponent: float,
        seed: int,
    ) -> None:
        rows = rows.reset_index(drop=True)
        if batch_size <= 0 or batch_size % 2:
            raise ValueError("Class-balanced batch_size must be a positive even integer")
        if num_batches <= 0:
            raise ValueError("num_batches must be positive")
        if not 0.0 <= source_weight_exponent <= 1.0:
            raise ValueError("source_weight_exponent must be in [0, 1]")
        if source_column not in rows.columns:
            raise ValueError(f"Missing sampling source column: {source_column!r}")
        if rows[source_column].isna().any() or rows["label"].isna().any():
            raise ValueError("Balanced batch sampling requires non-null labels and sources")
        labels = sorted(int(value) for value in rows["label"].unique())
        if labels != [0, 1]:
            raise ValueError(f"Binary balanced batches require labels [0, 1], got {labels}")

        self.source_indices: dict[int, dict[str, np.ndarray]] = {}
        self.source_draws: dict[int, dict[str, int]] = {}
        draws_per_class = num_batches * (batch_size // 2)
        for label in labels:
            label_rows = rows[rows["label"] == label]
            grouped = {
                str(source): values.index.to_numpy(dtype=np.int64, copy=True)
                for source, values in label_rows.groupby(source_column, sort=True)
            }
            sizes = np.asarray([len(grouped[source]) for source in grouped], dtype=np.float64)
            weights = np.power(sizes, source_weight_exponent)
            quotas = draws_per_class * weights / weights.sum()
            draws = np.floor(quotas).astype(np.int64)
            remainder = draws_per_class - int(draws.sum())
            order = np.argsort(-(quotas - draws), kind="stable")
            draws[order[:remainder]] += 1
            self.source_indices[label] = grouped
            self.source_draws[label] = {
                source: int(draw)
                for source, draw in zip(grouped, draws, strict=True)
            }

        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.epoch = 0
        self.source_column = source_column
        self.source_weight_exponent = float(source_weight_exponent)
        self.source_counts = {
            label: len(values) for label, values in self.source_indices.items()
        }
        self.expected_label_probabilities = {0: 0.5, 1: 0.5}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _draw_indices(
        indices: np.ndarray, count: int, rng: np.random.Generator
    ) -> list[int]:
        output: list[int] = []
        while len(output) < count:
            cycle = rng.permutation(indices)
            take = min(count - len(output), len(cycle))
            output.extend(int(value) for value in cycle[:take])
        return output

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        class_draws: dict[int, list[int]] = {}
        for label in (0, 1):
            values: list[int] = []
            for source, count in self.source_draws[label].items():
                values.extend(
                    self._draw_indices(self.source_indices[label][source], count, rng)
                )
            rng.shuffle(values)
            class_draws[label] = values

        half = self.batch_size // 2
        batches = []
        for batch_index in range(self.num_batches):
            start = batch_index * half
            batch = (
                class_draws[0][start : start + half]
                + class_draws[1][start : start + half]
            )
            rng.shuffle(batch)
            batches.append(batch)
        return iter(batches)

    def __len__(self) -> int:
        return self.num_batches


class ClassDomainQuotaBatchSampler(Sampler[list[int]]):
    """Build deterministic batches with exact class and domain exposure.

    Positive rows are drawn from the configured positive domain. Negative rows
    are split across named domains according to epoch-level quotas. Rows inside
    each domain are shuffled naturally and only repeat after the domain pool has
    been exhausted. This keeps the experimental variable at the domain dose,
    rather than silently introducing inverse-source weighting.
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        *,
        domain_column: str,
        batch_size: int,
        positive_per_batch: int,
        num_batches: int,
        positive_domain: str,
        negative_domain_fractions: dict[str, float],
        seed: int,
    ) -> None:
        rows = rows.reset_index(drop=True)
        if batch_size <= 1:
            raise ValueError("batch_size must be greater than one")
        if not 0 < positive_per_batch < batch_size:
            raise ValueError("positive_per_batch must be between zero and batch_size")
        if num_batches <= 0:
            raise ValueError("num_batches must be positive")
        if domain_column not in rows.columns:
            raise ValueError(f"Missing sampling domain column: {domain_column!r}")
        if rows[domain_column].isna().any() or rows["label"].isna().any():
            raise ValueError("Domain-quota sampling requires non-null labels and domains")
        labels = sorted(int(value) for value in rows["label"].unique())
        if labels != [0, 1]:
            raise ValueError(f"Binary domain-quota batches require labels [0, 1], got {labels}")

        fractions = {str(key): float(value) for key, value in negative_domain_fractions.items()}
        if not fractions:
            raise ValueError("negative_domain_fractions must not be empty")
        if any(not np.isfinite(value) or value < 0.0 for value in fractions.values()):
            raise ValueError("Negative-domain fractions must be finite and non-negative")
        if not np.isclose(sum(fractions.values()), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("Negative-domain fractions must sum to one")

        domains = rows[domain_column].astype(str)
        positive_mask = rows["label"].astype(int).eq(1)
        positive_domains = set(domains[positive_mask].unique())
        if positive_domains != {str(positive_domain)}:
            raise ValueError(
                "Positive rows must belong only to the configured positive domain; "
                f"observed={sorted(positive_domains)}"
            )
        for domain in fractions:
            mask = rows["label"].astype(int).eq(0) & domains.eq(domain)
            if not bool(mask.any()):
                raise ValueError(f"Configured negative domain has no rows: {domain!r}")
        unexpected_negative_domains = set(domains[rows["label"].astype(int).eq(0)].unique()) - set(fractions)
        if unexpected_negative_domains:
            raise ValueError(
                "Every negative domain must have an explicit quota, missing: "
                f"{sorted(unexpected_negative_domains)}"
            )

        self.domain_indices = {
            str(positive_domain): rows.index[positive_mask].to_numpy(dtype=np.int64, copy=True)
        }
        for domain in fractions:
            mask = rows["label"].astype(int).eq(0) & domains.eq(domain)
            self.domain_indices[domain] = rows.index[mask].to_numpy(dtype=np.int64, copy=True)

        self.batch_size = int(batch_size)
        self.positive_per_batch = int(positive_per_batch)
        self.negative_per_batch = self.batch_size - self.positive_per_batch
        self.num_batches = int(num_batches)
        self.positive_domain = str(positive_domain)
        self.negative_domain_fractions = fractions
        self.seed = int(seed)
        self.epoch = 0
        self.domain_column = str(domain_column)
        self.expected_label_probabilities = {
            0: self.negative_per_batch / self.batch_size,
            1: self.positive_per_batch / self.batch_size,
        }
        self.negative_domain_draws = self._apportion(
            self.num_batches * self.negative_per_batch,
            self.negative_domain_fractions,
        )

    @staticmethod
    def _apportion(total: int, fractions: dict[str, float]) -> dict[str, int]:
        domains = list(fractions)
        quotas = np.asarray([total * fractions[domain] for domain in domains])
        draws = np.floor(quotas).astype(np.int64)
        remainder = total - int(draws.sum())
        order = np.argsort(-(quotas - draws), kind="stable")
        draws[order[:remainder]] += 1
        return {domain: int(count) for domain, count in zip(domains, draws, strict=True)}

    @staticmethod
    def _draw_indices(
        indices: np.ndarray, count: int, rng: np.random.Generator
    ) -> list[int]:
        output: list[int] = []
        while len(output) < count:
            cycle = rng.permutation(indices)
            take = min(count - len(output), len(cycle))
            output.extend(int(value) for value in cycle[:take])
        return output

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        positive = self._draw_indices(
            self.domain_indices[self.positive_domain],
            self.num_batches * self.positive_per_batch,
            rng,
        )
        rng.shuffle(positive)

        negative_domains: list[str] = []
        negative_indices: dict[str, list[int]] = {}
        for domain, count in self.negative_domain_draws.items():
            negative_domains.extend([domain] * count)
            negative_indices[domain] = self._draw_indices(
                self.domain_indices[domain], count, rng
            )
        rng.shuffle(negative_domains)
        negative_offsets = {domain: 0 for domain in negative_indices}

        batches: list[list[int]] = []
        for batch_index in range(self.num_batches):
            positive_start = batch_index * self.positive_per_batch
            domain_start = batch_index * self.negative_per_batch
            batch = positive[
                positive_start : positive_start + self.positive_per_batch
            ]
            for domain in negative_domains[
                domain_start : domain_start + self.negative_per_batch
            ]:
                offset = negative_offsets[domain]
                batch.append(negative_indices[domain][offset])
                negative_offsets[domain] = offset + 1
            rng.shuffle(batch)
            batches.append(batch)
        return iter(batches)

    def __len__(self) -> int:
        return self.num_batches
