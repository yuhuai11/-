from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from dads_crnn.g19_open_set import (
    ClassConditionalBoundary,
    calibrate_class_thresholds,
    distance_threshold_metrics,
    fit_class_conditional_boundary,
    load_class_conditional_boundary,
    save_class_conditional_boundary,
    select_distance_threshold,
)


def _identity_boundary() -> ClassConditionalBoundary:
    return ClassConditionalBoundary(
        known_models=("A", "B"),
        pca_mean=np.zeros(2),
        pca_components=np.eye(2),
        class_centers=np.asarray([[0.0, 0.0], [10.0, 0.0]]),
        class_precisions=np.stack([np.eye(2), np.eye(2)]),
        class_thresholds=np.asarray([1.0, 1.0]),
        fallback_threshold=1.0,
        fallback_mask=np.asarray([False, False]),
    )


class G19OpenSetTests(unittest.TestCase):
    def test_fit_uses_global_pca_and_one_oas_model_per_class(self) -> None:
        embeddings = np.asarray(
            [
                [-2.0, -1.0, 0.0],
                [-1.8, -1.1, 0.1],
                [-2.2, -0.9, -0.1],
                [2.0, 1.0, 0.0],
                [1.8, 1.1, 0.1],
                [2.2, 0.9, -0.1],
            ]
        )
        boundary = fit_class_conditional_boundary(
            embeddings,
            np.asarray([0, 0, 0, 1, 1, 1]),
            ("A", "B"),
            pca_components=2,
        )
        self.assertEqual(boundary.pca_components.shape, (2, 3))
        self.assertEqual(boundary.class_centers.shape, (2, 2))
        self.assertEqual(boundary.class_precisions.shape, (2, 2, 2))
        self.assertTrue(np.isfinite(boundary.class_precisions).all())
        self.assertTrue(np.all(np.linalg.eigvalsh(boundary.class_precisions) > 0))

    def test_distance_is_conditioned_on_classifier_prediction(self) -> None:
        boundary = _identity_boundary()
        embeddings = np.asarray([[0.0, 0.0], [0.0, 0.0]])
        logits = np.asarray([[4.0, 0.0], [0.0, 4.0]])
        decision = boundary.score(embeddings, logits)
        np.testing.assert_array_equal(decision.predicted_classes, [0, 1])
        np.testing.assert_allclose(decision.distances, [0.0, 50.0])
        np.testing.assert_array_equal(decision.accepted, [True, False])

    def test_boundary_equality_is_accepted(self) -> None:
        metrics = distance_threshold_metrics(
            np.asarray([1.0]), np.asarray([1.0]), threshold=1.0
        )
        self.assertEqual(metrics["known_acceptance_rate"], 1.0)
        self.assertEqual(metrics["unknown_recall"], 0.0)

    def test_threshold_selection_respects_known_acceptance(self) -> None:
        threshold, metrics = select_distance_threshold(
            np.asarray([0.1, 0.2, 0.3, 0.9]),
            np.asarray([0.4, 0.5, 0.6, 1.0]),
            minimum_known_acceptance=0.75,
        )
        self.assertEqual(threshold, 0.3)
        self.assertGreaterEqual(metrics["known_acceptance_rate"], 0.75)
        self.assertEqual(metrics["unknown_recall"], 1.0)

    def test_calibration_marks_explicit_global_fallback(self) -> None:
        boundary = _identity_boundary()
        known_embeddings = np.asarray(
            [[0.1, 0.0], [0.2, 0.0], [9.9, 0.0], [10.2, 0.0]]
        )
        known_logits = np.asarray(
            [[3.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 3.0]]
        )
        unknown_embeddings = np.asarray([[2.0, 0.0], [3.0, 0.0]])
        unknown_logits = np.asarray([[3.0, 0.0], [3.0, 0.0]])
        calibrated, diagnostics = calibrate_class_thresholds(
            boundary,
            known_embeddings,
            known_logits,
            unknown_embeddings,
            unknown_logits,
            minimum_known_acceptance=1.0,
            minimum_known_support=2,
            minimum_unknown_support=2,
        )
        np.testing.assert_array_equal(calibrated.fallback_mask, [False, True])
        self.assertEqual(
            calibrated.class_thresholds[1], calibrated.fallback_threshold
        )
        self.assertEqual(diagnostics["fallback_classes"], [1])
        self.assertEqual(
            diagnostics["per_class"][1]["fallback_reasons"],
            ["insufficient_unknown_support"],
        )

    def test_fallback_can_widen_threshold_to_protect_predicted_class_known(
        self,
    ) -> None:
        boundary = _identity_boundary()
        calibrated, diagnostics = calibrate_class_thresholds(
            boundary,
            np.asarray(
                [[0.0, 0.0], [0.0, 0.0], [20.0, 0.0], [20.0, 0.0]]
            ),
            np.asarray(
                [[3.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 3.0]]
            ),
            np.asarray([[2.0, 0.0], [3.0, 0.0]]),
            np.asarray([[3.0, 0.0], [3.0, 0.0]]),
            minimum_known_acceptance=0.5,
            minimum_known_support=2,
            minimum_unknown_support=2,
        )

        self.assertTrue(calibrated.fallback_mask[1])
        self.assertGreater(
            calibrated.class_thresholds[1],
            calibrated.fallback_threshold,
        )
        self.assertEqual(
            diagnostics["per_class"][1]["fallback_policy"],
            "global_with_predicted_class_known_protection",
        )

    def test_open_set_decision_does_not_change_classifier_label(self) -> None:
        boundary = _identity_boundary()
        embeddings = np.asarray([[100.0, 100.0], [-100.0, -100.0]])
        logits = np.asarray([[0.0, 5.0], [5.0, 0.0]])
        decision = boundary.score(embeddings, logits)
        np.testing.assert_array_equal(
            decision.predicted_classes, np.argmax(logits, axis=1)
        )
        np.testing.assert_array_equal(decision.accepted, [False, False])

    def test_npz_round_trip_without_object_arrays(self) -> None:
        boundary = _identity_boundary()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boundary.npz"
            save_class_conditional_boundary(path, boundary)
            with np.load(path, allow_pickle=False) as archive:
                self.assertTrue(
                    all(archive[name].dtype.kind != "O" for name in archive.files)
                )
            restored = load_class_conditional_boundary(path)
        self.assertEqual(restored.known_models, boundary.known_models)
        np.testing.assert_allclose(restored.pca_components, boundary.pca_components)
        np.testing.assert_allclose(
            restored.class_precisions, boundary.class_precisions
        )
        np.testing.assert_array_equal(restored.fallback_mask, boundary.fallback_mask)

    def test_bound_npz_rejects_wrong_representation_checkpoint(self) -> None:
        boundary = replace(
            _identity_boundary(),
            representation_checkpoint_sha256="a" * 64,
            calibration_identity_sha256="b" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boundary.npz"
            save_class_conditional_boundary(
                path,
                boundary,
                require_identity=True,
            )
            with self.assertRaisesRegex(ValueError, "requires both"):
                load_class_conditional_boundary(path)
            restored = load_class_conditional_boundary(
                path,
                expected_representation_checkpoint_sha256="a" * 64,
                expected_calibration_identity_sha256="b" * 64,
            )
            self.assertEqual(
                restored.representation_checkpoint_sha256,
                "a" * 64,
            )
            with self.assertRaisesRegex(ValueError, "checkpoint identity"):
                load_class_conditional_boundary(
                    path,
                    expected_representation_checkpoint_sha256="c" * 64,
                    expected_calibration_identity_sha256="b" * 64,
                )

    def test_rejects_missing_training_class(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two recordings per class"):
            fit_class_conditional_boundary(
                np.asarray([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0]]),
                np.asarray([0, 0, 1]),
                ("A", "B"),
                pca_components=2,
            )

    def test_rejects_non_integer_predictions(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer class indices"):
            _identity_boundary().distances(
                np.asarray([[0.0, 0.0]]), np.asarray([0.0])
            )


if __name__ == "__main__":
    unittest.main()
