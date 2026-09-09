from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.covariance import OAS
from sklearn.decomposition import PCA


SCHEMA_VERSION = 1
DISTANCE_KIND = "pca_oas_normalized_squared_mahalanobis"


def _float_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _integer_vector(
    value: np.ndarray, *, name: str, expected_rows: int | None = None
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if expected_rows is not None and len(raw) != expected_rows:
        raise ValueError(f"{name} is not aligned with the input rows")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must contain integer class indices")
    return raw.astype(np.int64, copy=False)


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class OpenSetDecision:
    """An open-set decision that preserves the classifier's predicted class."""

    predicted_classes: np.ndarray
    distances: np.ndarray
    thresholds: np.ndarray
    accepted: np.ndarray
    used_conservative_fallback: np.ndarray


@dataclass(frozen=True)
class ClassConditionalBoundary:
    """PCA/OAS class geometry and calibrated distance thresholds.

    Distances are normalized squared Mahalanobis distances in the global PCA
    space. A sample is accepted as its classifier-predicted known class exactly
    when ``distance <= threshold``. This object never replaces or changes the
    classifier prediction.
    """

    known_models: tuple[str, ...]
    pca_mean: np.ndarray
    pca_components: np.ndarray
    class_centers: np.ndarray
    class_precisions: np.ndarray
    class_thresholds: np.ndarray
    fallback_threshold: float
    fallback_mask: np.ndarray
    representation_checkpoint_sha256: str = ""
    calibration_identity_sha256: str = ""
    schema_version: int = SCHEMA_VERSION
    distance_kind: str = DISTANCE_KIND

    def __post_init__(self) -> None:
        models = tuple(str(item) for item in self.known_models)
        if not models or any(not item for item in models):
            raise ValueError("known_models must contain non-empty model names")
        if len(set(models)) != len(models):
            raise ValueError("known_models must be unique and ordered by class index")
        if int(self.schema_version) != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported G19 boundary schema version: {self.schema_version}"
            )
        if str(self.distance_kind) != DISTANCE_KIND:
            raise ValueError(f"Unsupported G19 distance kind: {self.distance_kind}")
        identities = (
            str(self.representation_checkpoint_sha256),
            str(self.calibration_identity_sha256),
        )
        if any(identities) and not all(
            re.fullmatch(r"[0-9a-f]{64}", value) for value in identities
        ):
            raise ValueError(
                "G19 boundary identities must both be lowercase SHA256 values"
            )

        mean = np.asarray(self.pca_mean, dtype=np.float64)
        components = np.asarray(self.pca_components, dtype=np.float64)
        centers = np.asarray(self.class_centers, dtype=np.float64)
        precisions = np.asarray(self.class_precisions, dtype=np.float64)
        thresholds = np.asarray(self.class_thresholds, dtype=np.float64)
        fallback_mask = np.asarray(self.fallback_mask)
        class_count = len(models)

        if mean.ndim != 1 or mean.size == 0:
            raise ValueError("pca_mean must be a non-empty vector")
        if (
            components.ndim != 2
            or components.shape[0] == 0
            or components.shape[1] != len(mean)
        ):
            raise ValueError("pca_components has an invalid shape")
        pca_dim = components.shape[0]
        if pca_dim > components.shape[1]:
            raise ValueError("PCA output dimension cannot exceed input dimension")
        if centers.shape != (class_count, pca_dim):
            raise ValueError("class_centers has an invalid shape")
        if precisions.shape != (class_count, pca_dim, pca_dim):
            raise ValueError("class_precisions has an invalid shape")
        if thresholds.shape != (class_count,):
            raise ValueError("class_thresholds has an invalid shape")
        if fallback_mask.shape != (class_count,) or fallback_mask.dtype != np.bool_:
            raise ValueError("fallback_mask must be a boolean vector per class")
        if not all(
            np.isfinite(item).all()
            for item in (mean, components, centers, precisions, thresholds)
        ):
            raise ValueError("Boundary arrays must contain only finite values")
        if np.any(thresholds < 0.0):
            raise ValueError("Class thresholds must be non-negative")
        if (
            not np.isfinite(float(self.fallback_threshold))
            or float(self.fallback_threshold) < 0.0
        ):
            raise ValueError("fallback_threshold must be finite and non-negative")
        gram = components @ components.T
        if not np.allclose(gram, np.eye(pca_dim), rtol=1.0e-5, atol=1.0e-7):
            raise ValueError("pca_components rows must be orthonormal")
        for class_index, precision in enumerate(precisions):
            if not np.allclose(precision, precision.T, rtol=1.0e-7, atol=1.0e-9):
                raise ValueError(
                    f"Precision matrix for class {class_index} is not symmetric"
                )
            if float(np.linalg.eigvalsh(precision).min()) <= 0.0:
                raise ValueError(
                    f"Precision matrix for class {class_index} is not positive definite"
                )

        object.__setattr__(self, "known_models", models)
        object.__setattr__(self, "pca_mean", _readonly(mean))
        object.__setattr__(self, "pca_components", _readonly(components))
        object.__setattr__(self, "class_centers", _readonly(centers))
        object.__setattr__(self, "class_precisions", _readonly(precisions))
        object.__setattr__(self, "class_thresholds", _readonly(thresholds))
        object.__setattr__(self, "fallback_threshold", float(self.fallback_threshold))
        object.__setattr__(self, "fallback_mask", _readonly(fallback_mask))
        object.__setattr__(
            self,
            "representation_checkpoint_sha256",
            identities[0],
        )
        object.__setattr__(
            self,
            "calibration_identity_sha256",
            identities[1],
        )
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "distance_kind", str(self.distance_kind))

    @property
    def input_dimension(self) -> int:
        return int(self.pca_components.shape[1])

    @property
    def pca_dimension(self) -> int:
        return int(self.pca_components.shape[0])

    @property
    def class_count(self) -> int:
        return len(self.known_models)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        values = _float_matrix(embeddings, name="embeddings")
        if values.shape[1] != self.input_dimension:
            raise ValueError(
                "Embedding dimension does not match the fitted G19 boundary"
            )
        return (values - self.pca_mean) @ self.pca_components.T

    def distances(
        self, embeddings: np.ndarray, predicted_classes: np.ndarray
    ) -> np.ndarray:
        values = _float_matrix(embeddings, name="embeddings")
        predictions = _integer_vector(
            predicted_classes,
            name="predicted_classes",
            expected_rows=len(values),
        )
        if np.any((predictions < 0) | (predictions >= self.class_count)):
            raise ValueError("predicted_classes contains an out-of-range class index")
        projected = self.transform(values)
        residual = projected - self.class_centers[predictions]
        precision = self.class_precisions[predictions]
        squared = np.einsum(
            "ni,nij,nj->n", residual, precision, residual, optimize=True
        )
        if np.any(squared < -1.0e-8) or not np.isfinite(squared).all():
            raise RuntimeError("The fitted G19 boundary produced invalid distances")
        return np.maximum(squared, 0.0) / float(self.pca_dimension)

    def score(self, embeddings: np.ndarray, logits: np.ndarray) -> OpenSetDecision:
        values = _float_matrix(embeddings, name="embeddings")
        classifier_logits = _float_matrix(logits, name="logits")
        if len(values) != len(classifier_logits):
            raise ValueError("embeddings and logits must contain the same rows")
        if classifier_logits.shape[1] != self.class_count:
            raise ValueError("logits class dimension does not match known_models")
        predictions = np.argmax(classifier_logits, axis=1).astype(np.int64)
        distances = self.distances(values, predictions)
        thresholds = self.class_thresholds[predictions]
        return OpenSetDecision(
            predicted_classes=predictions,
            distances=distances,
            thresholds=np.asarray(thresholds, dtype=np.float64),
            accepted=distances <= thresholds,
            used_conservative_fallback=self.fallback_mask[predictions],
        )


def _minimum_acceptance_threshold(
    distances: np.ndarray, *, minimum_acceptance: float
) -> float:
    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Distances must be a finite non-empty vector")
    if np.any(values < 0.0):
        raise ValueError("Distances must be non-negative")
    if not (0.0 < minimum_acceptance <= 1.0):
        raise ValueError("minimum_acceptance must be inside (0, 1]")
    rank = max(0, int(np.ceil(minimum_acceptance * len(values))) - 1)
    return float(np.sort(values)[rank])


def fit_class_conditional_boundary(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    known_models: Sequence[str],
    *,
    pca_components: int,
    initial_known_acceptance: float = 0.95,
) -> ClassConditionalBoundary:
    """Fit global PCA and one OAS Gaussian geometry per true training class."""

    embeddings = _float_matrix(train_embeddings, name="train_embeddings")
    targets = _integer_vector(
        train_targets, name="train_targets", expected_rows=len(embeddings)
    )
    models = tuple(str(item) for item in known_models)
    if not models:
        raise ValueError("known_models cannot be empty")
    if any(not item for item in models) or len(set(models)) != len(models):
        raise ValueError("known_models must contain unique non-empty names")
    class_count = len(models)
    if np.any((targets < 0) | (targets >= class_count)):
        raise ValueError("train_targets contains an out-of-range class index")
    counts = np.bincount(targets, minlength=class_count)
    if np.any(counts < 2):
        missing = np.flatnonzero(counts < 2).tolist()
        raise ValueError(
            "OAS fitting requires at least two recordings per class; "
            f"insufficient classes: {missing}"
        )
    if not isinstance(pca_components, (int, np.integer)) or pca_components <= 0:
        raise ValueError("pca_components must be a positive integer")
    maximum_components = min(embeddings.shape)
    if pca_components > maximum_components:
        raise ValueError(
            f"pca_components={pca_components} exceeds {maximum_components}"
        )
    if not (0.0 < initial_known_acceptance <= 1.0):
        raise ValueError("initial_known_acceptance must be inside (0, 1]")

    pca = PCA(n_components=int(pca_components), svd_solver="full")
    projected = np.asarray(pca.fit_transform(embeddings), dtype=np.float64)
    centers: list[np.ndarray] = []
    precisions: list[np.ndarray] = []
    for class_index in range(class_count):
        estimator = OAS(store_precision=True, assume_centered=False)
        estimator.fit(projected[targets == class_index])
        center = np.asarray(estimator.location_, dtype=np.float64)
        precision = np.asarray(estimator.precision_, dtype=np.float64)
        precision = 0.5 * (precision + precision.T)
        centers.append(center)
        precisions.append(precision)

    provisional = ClassConditionalBoundary(
        known_models=models,
        pca_mean=np.asarray(pca.mean_, dtype=np.float64),
        pca_components=np.asarray(pca.components_, dtype=np.float64),
        class_centers=np.stack(centers),
        class_precisions=np.stack(precisions),
        class_thresholds=np.zeros(class_count, dtype=np.float64),
        fallback_threshold=0.0,
        fallback_mask=np.ones(class_count, dtype=np.bool_),
    )
    training_distances = provisional.distances(embeddings, targets)
    fallback_threshold = _minimum_acceptance_threshold(
        training_distances, minimum_acceptance=initial_known_acceptance
    )
    return replace(
        provisional,
        class_thresholds=np.full(class_count, fallback_threshold, dtype=np.float64),
        fallback_threshold=fallback_threshold,
    )


def distance_threshold_metrics(
    known_distances: np.ndarray,
    unknown_distances: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    known = np.asarray(known_distances, dtype=np.float64)
    unknown = np.asarray(unknown_distances, dtype=np.float64)
    if (
        known.ndim != 1
        or unknown.ndim != 1
        or known.size == 0
        or unknown.size == 0
        or not np.isfinite(known).all()
        or not np.isfinite(unknown).all()
        or np.any(known < 0.0)
        or np.any(unknown < 0.0)
    ):
        raise ValueError(
            "Threshold metrics require finite non-empty non-negative distance vectors"
        )
    if not np.isfinite(float(threshold)) or float(threshold) < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    known_acceptance = float(np.mean(known <= threshold))
    unknown_recall = float(np.mean(unknown > threshold))
    return {
        "known_acceptance_rate": known_acceptance,
        "known_rejection_rate": 1.0 - known_acceptance,
        "unknown_recall": unknown_recall,
        "unknown_false_acceptance_rate": 1.0 - unknown_recall,
        "balanced_accuracy": 0.5 * (known_acceptance + unknown_recall),
    }


def select_distance_threshold(
    known_distances: np.ndarray,
    unknown_distances: np.ndarray,
    *,
    minimum_known_acceptance: float,
) -> tuple[float, dict[str, float]]:
    if not (0.0 < minimum_known_acceptance <= 1.0):
        raise ValueError("minimum_known_acceptance must be inside (0, 1]")
    known = np.asarray(known_distances, dtype=np.float64)
    unknown = np.asarray(unknown_distances, dtype=np.float64)
    # Validate both arrays, including non-negativity, before building candidates.
    distance_threshold_metrics(known, unknown, 0.0)
    candidates = np.unique(np.concatenate((known, unknown)))
    eligible: list[tuple[float, dict[str, float]]] = []
    for candidate in candidates:
        threshold = float(candidate)
        metrics = distance_threshold_metrics(known, unknown, threshold)
        if (
            metrics["known_acceptance_rate"] + 1.0e-12
            >= minimum_known_acceptance
        ):
            eligible.append((threshold, metrics))
    if not eligible:
        raise RuntimeError("No distance threshold satisfies known acceptance")
    return max(
        eligible,
        key=lambda item: (
            item[1]["balanced_accuracy"],
            item[1]["unknown_recall"],
            item[1]["known_acceptance_rate"],
            -item[0],
        ),
    )


def calibrate_class_thresholds(
    boundary: ClassConditionalBoundary,
    known_embeddings: np.ndarray,
    known_logits: np.ndarray,
    unknown_embeddings: np.ndarray,
    unknown_logits: np.ndarray,
    *,
    minimum_known_acceptance: float,
    minimum_known_support: int = 5,
    minimum_unknown_support: int = 3,
) -> tuple[ClassConditionalBoundary, dict[str, Any]]:
    """Calibrate global and per-predicted-class thresholds.

    A class uses the global threshold whenever either its known or unknown
    predicted-class subset lacks the configured support. Geometry remains
    class-conditional and classifier predictions are never relabelled here.
    """

    if (
        not isinstance(minimum_known_support, (int, np.integer))
        or minimum_known_support < 1
    ):
        raise ValueError("minimum_known_support must be a positive integer")
    if (
        not isinstance(minimum_unknown_support, (int, np.integer))
        or minimum_unknown_support < 1
    ):
        raise ValueError("minimum_unknown_support must be a positive integer")

    known_decision = boundary.score(known_embeddings, known_logits)
    unknown_decision = boundary.score(unknown_embeddings, unknown_logits)
    global_threshold, global_metrics = select_distance_threshold(
        known_decision.distances,
        unknown_decision.distances,
        minimum_known_acceptance=minimum_known_acceptance,
    )
    thresholds = np.full(
        boundary.class_count, global_threshold, dtype=np.float64
    )
    fallback_mask = np.ones(boundary.class_count, dtype=np.bool_)
    per_class: list[dict[str, Any]] = []

    for class_index, model_id in enumerate(boundary.known_models):
        known_mask = known_decision.predicted_classes == class_index
        unknown_mask = unknown_decision.predicted_classes == class_index
        known_values = known_decision.distances[known_mask]
        unknown_values = unknown_decision.distances[unknown_mask]
        known_support = int(known_mask.sum())
        unknown_support = int(unknown_mask.sum())
        reasons: list[str] = []
        if known_support < minimum_known_support:
            reasons.append("insufficient_known_support")
        if unknown_support < minimum_unknown_support:
            reasons.append("insufficient_unknown_support")

        if not reasons:
            threshold, metrics = select_distance_threshold(
                known_values,
                unknown_values,
                minimum_known_acceptance=minimum_known_acceptance,
            )
            thresholds[class_index] = threshold
            fallback_mask[class_index] = False
            applied_metrics: dict[str, float] | None = metrics
        else:
            threshold = global_threshold
            if known_support > 0:
                threshold = max(
                    threshold,
                    _minimum_acceptance_threshold(
                        known_values,
                        minimum_acceptance=minimum_known_acceptance,
                    ),
                )
            thresholds[class_index] = threshold
            applied_metrics = (
                distance_threshold_metrics(known_values, unknown_values, threshold)
                if known_support > 0 and unknown_support > 0
                else None
            )
        per_class.append(
            {
                "class_index": class_index,
                "model_id": model_id,
                "known_support": known_support,
                "unknown_support": unknown_support,
                "threshold": float(threshold),
                "used_conservative_fallback": bool(reasons),
                "fallback_policy": (
                    "global_with_predicted_class_known_protection"
                    if reasons
                    else "class_calibrated"
                ),
                "fallback_reasons": reasons,
                "applied_metrics": applied_metrics,
            }
        )

    calibrated = replace(
        boundary,
        class_thresholds=thresholds,
        fallback_threshold=global_threshold,
        fallback_mask=fallback_mask,
    )
    diagnostics: dict[str, Any] = {
        "minimum_known_acceptance": float(minimum_known_acceptance),
        "minimum_known_support": int(minimum_known_support),
        "minimum_unknown_support": int(minimum_unknown_support),
        "global_threshold": global_threshold,
        "global_metrics": global_metrics,
        "fallback_classes": np.flatnonzero(fallback_mask).astype(int).tolist(),
        "per_class": per_class,
    }
    return calibrated, diagnostics


def score_class_conditional_boundary(
    boundary: ClassConditionalBoundary,
    embeddings: np.ndarray,
    logits: np.ndarray,
) -> OpenSetDecision:
    return boundary.score(embeddings, logits)


def save_class_conditional_boundary(
    path: str | Path,
    boundary: ClassConditionalBoundary,
    *,
    require_identity: bool = False,
) -> None:
    if require_identity and not boundary.representation_checkpoint_sha256:
        raise ValueError("Refusing to save an unbound G19 production boundary")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray([boundary.schema_version], dtype=np.int64),
            distance_kind=np.asarray([boundary.distance_kind], dtype=np.str_),
            known_models=np.asarray(boundary.known_models, dtype=np.str_),
            pca_mean=boundary.pca_mean,
            pca_components=boundary.pca_components,
            class_centers=boundary.class_centers,
            class_precisions=boundary.class_precisions,
            class_thresholds=boundary.class_thresholds,
            fallback_threshold=np.asarray(
                [boundary.fallback_threshold], dtype=np.float64
            ),
            fallback_mask=boundary.fallback_mask,
            representation_checkpoint_sha256=np.asarray(
                [boundary.representation_checkpoint_sha256], dtype=np.str_
            ),
            calibration_identity_sha256=np.asarray(
                [boundary.calibration_identity_sha256], dtype=np.str_
            ),
        )
    os.replace(temporary, output)


def load_class_conditional_boundary(
    path: str | Path,
    *,
    expected_representation_checkpoint_sha256: str | None = None,
    expected_calibration_identity_sha256: str | None = None,
) -> ClassConditionalBoundary:
    required = {
        "schema_version",
        "distance_kind",
        "known_models",
        "pca_mean",
        "pca_components",
        "class_centers",
        "class_precisions",
        "class_thresholds",
        "fallback_threshold",
        "fallback_mask",
        "representation_checkpoint_sha256",
        "calibration_identity_sha256",
    }
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        unexpected = sorted(set(archive.files).difference(required))
        if missing or unexpected:
            raise ValueError(
                "Invalid G19 boundary archive fields: "
                f"missing={missing}, unexpected={unexpected}"
            )
        schema = np.asarray(archive["schema_version"])
        kind = np.asarray(archive["distance_kind"])
        fallback_threshold = np.asarray(archive["fallback_threshold"])
        representation_identity = np.asarray(
            archive["representation_checkpoint_sha256"]
        )
        calibration_identity = np.asarray(archive["calibration_identity_sha256"])
        if schema.shape != (1,) or kind.shape != (1,):
            raise ValueError("Invalid scalar metadata in G19 boundary archive")
        if fallback_threshold.shape != (1,):
            raise ValueError("Invalid fallback threshold in G19 boundary archive")
        if representation_identity.shape != (1,) or calibration_identity.shape != (
            1,
        ):
            raise ValueError("Invalid identity metadata in G19 boundary archive")
        boundary = ClassConditionalBoundary(
            known_models=tuple(str(item) for item in archive["known_models"].tolist()),
            pca_mean=archive["pca_mean"],
            pca_components=archive["pca_components"],
            class_centers=archive["class_centers"],
            class_precisions=archive["class_precisions"],
            class_thresholds=archive["class_thresholds"],
            fallback_threshold=float(fallback_threshold[0]),
            fallback_mask=np.asarray(archive["fallback_mask"], dtype=np.bool_),
            representation_checkpoint_sha256=str(representation_identity[0]),
            calibration_identity_sha256=str(calibration_identity[0]),
            schema_version=int(schema[0]),
            distance_kind=str(kind[0]),
        )
    if boundary.representation_checkpoint_sha256 and (
        expected_representation_checkpoint_sha256 is None
        or expected_calibration_identity_sha256 is None
    ):
        raise ValueError(
            "Loading a bound G19 boundary requires both expected identities"
        )
    if (
        expected_representation_checkpoint_sha256 is not None
        and boundary.representation_checkpoint_sha256
        != expected_representation_checkpoint_sha256
    ):
        raise ValueError("G19 boundary representation checkpoint identity mismatch")
    if (
        expected_calibration_identity_sha256 is not None
        and boundary.calibration_identity_sha256
        != expected_calibration_identity_sha256
    ):
        raise ValueError("G19 boundary calibration identity mismatch")
    return boundary
