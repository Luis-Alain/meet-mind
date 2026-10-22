"""Corpus-level numbers, computed from per-case predictions and labels.

The SDK scores each case separately. Some metrics only make sense over the whole set (an overall
F1, a confusion matrix, calibration, latency percentiles), so they are computed here from the same
predictions the evaluators saw.
"""

from evals.matching import analyse_case
from evals.metrics import (
    brier_score,
    confusion_matrix,
    coverage_accuracy_curve,
    expected_calibration_error,
    macro_f1,
    per_class_scores,
    percentile,
    precision_recall_f1,
    ratio,
)

STATEMENT_TYPES = ["task", "decision", "suggestion", "risk", "question", "other"]
CURVE_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def aggregate_pipeline(rows: list[tuple[dict, dict]]) -> dict:
    """Overall quality of the extraction.

    ``rows`` holds ``(expected, predicted)`` pairs: a meeting's labels and the pipeline output for
    that meeting. Task counts are pooled over all meetings (micro-averaged), so a meeting with
    many tasks weighs more than one with few.
    """
    tp = fp = fn = 0
    matched = assignee_ok = date_ok = 0
    traps_total = trap_hits = 0
    predicted = flagged = wrong = caught = 0
    evidence_total = evidence_unverified = 0
    input_tokens = output_tokens = 0
    extraction_ms: list[float] = []
    decider_ms: list[float] = []
    decider_calls = decider_abstained = decider_errors = 0

    for expected, output in rows:
        tasks = output["tasks"]
        analysis = analyse_case(expected, tasks)
        tp += len(analysis.matches)
        fp += len(analysis.extra)
        fn += len(analysis.missed)
        matched += len(analysis.matches)
        assignee_ok += sum(analysis.assignee_correct.values())
        date_ok += sum(analysis.date_correct.values())
        traps_total += len(expected["not_task_quotes"])
        trap_hits += len(analysis.trap_hits)

        for index, task in enumerate(tasks):
            predicted += 1
            is_wrong = index in analysis.wrong
            wrong += is_wrong
            flagged += task["needs_review"]
            caught += is_wrong and task["needs_review"]
            evidence_total += task["evidence_total"]
            evidence_unverified += task["evidence_unverified"]

        usage = output["usage"]
        input_tokens += usage["input_tokens"]
        output_tokens += usage["output_tokens"]
        extraction_ms.append(usage["extraction_latency_ms"])
        for call in output["decider"]["calls"]:
            if call is None:
                continue
            decider_calls += 1
            decider_abstained += call["abstained"]
            decider_errors += call["error"] is not None
            if call["latency_ms"] is not None:
                decider_ms.append(call["latency_ms"])

    return {
        "meetings": len(rows),
        "tasks": {"tp": tp, "fp": fp, "fn": fn, **precision_recall_f1(tp, fp, fn)},
        "fields": {
            "matched_tasks": matched,
            "assignee_accuracy": ratio(assignee_ok, matched),
            "due_date_accuracy": ratio(date_ok, matched),
        },
        "traps": {"hits": trap_hits, "total": traps_total},
        # How well the "needs review" flag points at the tasks that are really wrong.
        "review": {
            "predicted_tasks": predicted,
            "flagged": flagged,
            "wrong": wrong,
            "caught": caught,
            "silent_errors": wrong - caught,  # wrong and NOT flagged: the dangerous ones
            "false_alarms": flagged - caught,  # flagged although correct: wasted reviewer time
            "flag_precision": ratio(caught, flagged),
            "flag_recall": ratio(caught, wrong),
        },
        "evidence": {
            "quotes": evidence_total,
            "unverified": evidence_unverified,
            "unsupported_rate": ratio(evidence_unverified, evidence_total),
        },
        "cost": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "extraction_latency_ms_p50": percentile(extraction_ms, 50),
            "extraction_latency_ms_p95": percentile(extraction_ms, 95),
        },
        "decider": {
            "calls": decider_calls,
            "abstained": decider_abstained,
            "abstention_rate": ratio(decider_abstained, decider_calls),
            "errors": decider_errors,
            "latency_ms_p50": percentile(decider_ms, 50),
            "latency_ms_p95": percentile(decider_ms, 95),
        },
    }


def aggregate_decider(rows: list[tuple[dict, dict]]) -> dict:
    """Quality of Decider on single sentences.

    ``rows`` holds ``(label, prediction)`` pairs. Rows where the call failed are counted as errors
    and left out of the accuracy figures, so a server outage is not mistaken for wrong answers.
    """
    answered = [(label, pred) for label, pred in rows if pred["error"] is None]
    errors = len(rows) - len(answered)

    pairs = [(label["statement_type"], pred["statement_type"]) for label, pred in answered]
    matrix = confusion_matrix(pairs, STATEMENT_TYPES)
    per_class = per_class_scores(matrix)
    confidences = [pred["confidence"] for _, pred in answered]
    correct = [label["statement_type"] == pred["statement_type"] for label, pred in answered]

    def yes_no(key: str, prob_key: str) -> dict:
        """Accuracy (probability >= 0.5 counts as "yes") and Brier score of a yes/no question."""
        probs = [pred[prob_key] for _, pred in answered]
        truth = [label[key] for label, _ in answered]
        hits = [(p >= 0.5) == t for p, t in zip(probs, truth, strict=True)]
        return {"accuracy": ratio(sum(hits), len(hits)), "brier": brier_score(probs, truth)}

    latencies = [pred["latency_ms"] for _, pred in answered if pred["latency_ms"] is not None]
    return {
        "fragments": len(rows),
        "errors": errors,
        "statement_type": {
            "accuracy": ratio(sum(correct), len(correct)),
            "macro_f1": macro_f1(per_class),
            "per_class": per_class,
            "confusion_matrix": matrix,
            "ece": expected_calibration_error(confidences, correct),
            "coverage_accuracy": coverage_accuracy_curve(confidences, correct, CURVE_THRESHOLDS),
        },
        "explicit_commitment": yes_no("explicit_commitment", "explicit_commitment"),
        "evidence_sufficient": yes_no("evidence_sufficient", "evidence_sufficient"),
        "abstention_rate": ratio(sum(pred["abstained"] for _, pred in answered), len(answered)),
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
    }
