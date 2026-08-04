import numpy as np
import pytest

from dads_crnn.evaluate_g22_internal_holdout import (
    aggregate_reports,
    classification_report,
)


MODELS = ["A", "B", "C"]


def test_classification_report_contains_requested_metrics() -> None:
    targets = np.asarray([0, 0, 1, 1, 2, 2])
    predictions = np.asarray([0, 1, 1, 1, 2, 0])
    scores = np.eye(3)[predictions]
    report = classification_report(targets, scores, MODELS)

    assert report["accuracy"] == pytest.approx(4 / 6)
    assert report["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert report["per_class"]["A"] == {
        "precision": pytest.approx(0.5),
        "recall": pytest.approx(0.5),
        "f1": pytest.approx(0.5),
        "support": 2,
    }
    assert report["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]


def test_three_seed_aggregate_uses_sample_standard_deviation() -> None:
    reports = {}
    for seed, accuracy in zip((42, 43, 44), (0.8, 0.9, 1.0), strict=True):
        reports[seed] = {
            "accuracy": accuracy,
            "macro_f1": accuracy - 0.1,
            "per_class": {
                model: {
                    "precision": accuracy,
                    "recall": accuracy,
                    "f1": accuracy,
                    "support": 2,
                }
                for model in MODELS
            },
        }
    aggregate = aggregate_reports(reports, MODELS)

    assert aggregate["accuracy"]["mean"] == pytest.approx(0.9)
    assert aggregate["accuracy"]["sample_std"] == pytest.approx(0.1)
    assert aggregate["per_class"]["A"]["f1"]["mean"] == pytest.approx(0.9)
