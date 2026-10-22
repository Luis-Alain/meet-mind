import pytest
from evals.metrics import (
    brier_score,
    confusion_matrix,
    coverage_accuracy_curve,
    expected_calibration_error,
    macro_f1,
    per_class_scores,
    percentile,
    precision_recall_f1,
)


def test_precision_recall_f1_by_hand():
    # 6 found right, 2 invented, 4 missed -> P = 6/8, R = 6/10, F1 = 2PR/(P+R)
    scores = precision_recall_f1(tp=6, fp=2, fn=4)
    assert scores["precision"] == pytest.approx(0.75)
    assert scores["recall"] == pytest.approx(0.6)
    assert scores["f1"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)


def test_undefined_metrics_are_none_not_zero():
    nothing_predicted = precision_recall_f1(tp=0, fp=0, fn=3)
    assert nothing_predicted["precision"] is None and nothing_predicted["recall"] == 0.0
    assert nothing_predicted["f1"] == 0.0
    nothing_at_all = precision_recall_f1(tp=0, fp=0, fn=0)
    assert nothing_at_all == {"precision": None, "recall": None, "f1": None}


def test_percentile_interpolates():
    assert percentile([10, 20, 30, 40], 50) == 25
    assert percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)
    assert percentile([7], 95) == 7
    assert percentile([], 50) is None


def test_confusion_matrix_and_per_class_scores():
    pairs = (
        [("task", "task")] * 3
        + [("task", "other")]
        + [("other", "other")] * 2
        + [("other", "task")]
    )
    matrix = confusion_matrix(pairs, ["task", "other"])
    assert matrix == {"task": {"task": 3, "other": 1}, "other": {"task": 1, "other": 2}}
    scores = per_class_scores(matrix)
    assert scores["task"]["precision"] == pytest.approx(3 / 4)
    assert scores["task"]["recall"] == pytest.approx(3 / 4)
    assert scores["other"]["support"] == 3
    assert macro_f1(scores) == pytest.approx((0.75 + 2 / 3) / 2)


def test_macro_f1_ignores_classes_that_never_occur():
    matrix = confusion_matrix([("task", "task")], ["task", "risk"])
    assert macro_f1(per_class_scores(matrix)) == 1.0


def test_brier_score():
    assert brier_score([1.0, 0.0], [True, False]) == 0.0
    assert brier_score([0.8, 0.2], [True, True]) == pytest.approx((0.04 + 0.64) / 2)
    assert brier_score([], []) is None


def test_ece_is_zero_when_calibrated_and_large_when_overconfident():
    # Says 100 % and is right half the time -> gap of 0.5.
    assert expected_calibration_error([1.0] * 4, [True, True, False, False]) == pytest.approx(0.5)
    # Says 50 % and is right half the time -> perfectly calibrated.
    assert expected_calibration_error([0.5] * 4, [True, False, True, False]) == pytest.approx(0.0)
    assert expected_calibration_error([], []) is None


def test_coverage_accuracy_curve():
    confidences = [0.9, 0.8, 0.6, 0.4]
    correct = [True, True, False, True]
    curve = coverage_accuracy_curve(confidences, correct, [0.5, 0.85])
    assert curve[0] == {
        "threshold": 0.5,
        "answered": 3,
        "coverage": 0.75,
        "accuracy": pytest.approx(2 / 3),
    }
    assert curve[1]["answered"] == 1 and curve[1]["accuracy"] == 1.0
    assert coverage_accuracy_curve(confidences, correct, [0.99])[0]["accuracy"] is None
