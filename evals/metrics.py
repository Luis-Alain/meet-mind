"""Metric formulas, written without any library so each one can be read and checked by hand.

All functions are pure: numbers in, numbers out. ``None`` is returned (never a made-up value)
when a metric is undefined, for example precision when nothing was predicted.
"""

import math
from collections import Counter


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    """Precision: of what was predicted, how much was right. Recall: of what existed, how much
    was found. F1 is their harmonic mean, so it is low if either one is low."""
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = 0.0 if (tp + fp + fn) else None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def percentile(values: list[float], q: float) -> float | None:
    """The q-th percentile (0 to 100) with linear interpolation between neighbouring values."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def confusion_matrix(pairs: list[tuple[str, str]], labels: list[str]) -> dict[str, dict[str, int]]:
    """``matrix[true][predicted]`` = how many times ``true`` was predicted as ``predicted``."""
    counts = Counter(pairs)
    return {t: {p: counts[(t, p)] for p in labels} for t in labels}


def per_class_scores(matrix: dict[str, dict[str, int]]) -> dict[str, dict[str, float | None]]:
    scores = {}
    for label in matrix:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in matrix if other != label)
        fn = sum(matrix[label][other] for other in matrix[label] if other != label)
        scores[label] = {**precision_recall_f1(tp, fp, fn), "support": tp + fn}
    return scores


def macro_f1(per_class: dict[str, dict[str, float | None]]) -> float | None:
    """Average F1 over the classes that actually occur, so a rare class weighs as much as a
    common one. Classes with no true examples are left out instead of dragging the mean down."""
    values = [s["f1"] for s in per_class.values() if s["support"] and s["f1"] is not None]
    return sum(values) / len(values) if values else None


def brier_score(probabilities: list[float], outcomes: list[bool]) -> float | None:
    """Mean squared gap between a predicted probability and what happened (0 is perfect)."""
    if not probabilities:
        return None
    return sum((p - float(o)) ** 2 for p, o in zip(probabilities, outcomes, strict=True)) / len(
        probabilities
    )


def expected_calibration_error(
    confidences: list[float], correct: list[bool], bins: int = 10
) -> float | None:
    """How far stated confidence is from real accuracy, averaged over confidence bins.

    Sort predictions into equal-width confidence bins; in each, compare the average confidence
    with the fraction that was actually right, and weight the gap by the bin's share of the data.
    A model that says "80 %" and is right 80 % of the time scores 0.
    """
    total = len(confidences)
    if not total:
        return None
    ece = 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        members = [
            i for i, c in enumerate(confidences) if (low < c <= high) or (b == 0 and c == 0.0)
        ]
        if not members:
            continue
        avg_confidence = sum(confidences[i] for i in members) / len(members)
        accuracy = sum(correct[i] for i in members) / len(members)
        ece += len(members) / total * abs(avg_confidence - accuracy)
    return ece


def coverage_accuracy_curve(
    confidences: list[float], correct: list[bool], thresholds: list[float]
) -> list[dict[str, float | int | None]]:
    """For each confidence threshold: if the system only answered at or above it, what share of
    cases would it answer (coverage) and how often would it be right (accuracy)? This is the
    table to read when choosing the threshold under which answers go to a human."""
    total = len(confidences)
    curve = []
    for threshold in thresholds:
        kept = [i for i, c in enumerate(confidences) if c >= threshold]
        curve.append(
            {
                "threshold": threshold,
                "answered": len(kept),
                "coverage": ratio(len(kept), total),
                "accuracy": ratio(sum(correct[i] for i in kept), len(kept)),
            }
        )
    return curve
