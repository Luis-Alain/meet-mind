"""Decides which predicted task corresponds to which expected task.

The system words task titles its own way, so titles cannot be compared. What both sides share is
the evidence: the sentence of the transcript that justifies the task. A predicted task is paired
with an expected one when their evidence sentences overlap enough.
"""

from dataclasses import dataclass, field

from meet_mind.services.validation import normalize

# Share of the expected quote's words that must also appear in a predicted quote to pair them.
OVERLAP_THRESHOLD = 0.5


def _words(text: str) -> set[str]:
    return set(normalize(text).split())


def overlap(expected_quote: str, predicted_quote: str) -> float:
    """Fraction of the expected quote's words found in the predicted quote (0 to 1).

    Measured from the expected side so a prediction that quotes a longer span around the expected
    sentence still counts, while a prediction that quotes an unrelated sentence does not.
    """
    expected = _words(expected_quote)
    if not expected:
        return 0.0
    return len(expected & _words(predicted_quote)) / len(expected)


def _best_overlap(expected_quote: str, predicted_quotes: list[str]) -> float:
    return max((overlap(expected_quote, quote) for quote in predicted_quotes), default=0.0)


def same_person(a: str | None, b: str | None) -> bool:
    """Two assignees agree if both are empty, or both name the same first name."""
    if a is None or b is None:
        return a is None and b is None
    first_a, first_b = normalize(a).split(), normalize(b).split()
    return bool(first_a and first_b) and first_a[0] == first_b[0]


@dataclass
class CaseAnalysis:
    """How the predicted tasks of one meeting line up with the expected ones."""

    matches: list[tuple[int, int]] = field(default_factory=list)  # (expected idx, predicted idx)
    missed: list[int] = field(default_factory=list)  # expected tasks nobody found
    extra: list[int] = field(default_factory=list)  # predicted tasks that match nothing expected
    trap_hits: list[tuple[int, int]] = field(default_factory=list)  # (predicted idx, trap idx)
    assignee_correct: dict[int, bool] = field(default_factory=dict)  # by predicted idx
    date_correct: dict[int, bool] = field(default_factory=dict)  # by predicted idx
    wrong: set[int] = field(default_factory=set)  # predicted tasks with any error


def analyse_case(expected: dict, predicted_tasks: list[dict]) -> CaseAnalysis:
    """Compare predicted tasks with a meeting's labels.

    ``expected`` is a ``MeetingLabels`` dump; each predicted task needs ``evidence_quotes``,
    ``assignee`` and ``due_date`` (ISO string or None).
    """
    gold = expected["tasks"]
    quotes = [task["evidence_quotes"] for task in predicted_tasks]

    # Pair greedily from the strongest overlap down; each task is used at most once.
    candidates = sorted(
        (
            (_best_overlap(g["evidence_quote"], quotes[p]), gi, p)
            for gi, g in enumerate(gold)
            for p in range(len(predicted_tasks))
        ),
        reverse=True,
    )
    analysis = CaseAnalysis()
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    for score, gi, p in candidates:
        if score < OVERLAP_THRESHOLD:
            break
        if gi in used_gold or p in used_pred:
            continue
        used_gold.add(gi)
        used_pred.add(p)
        analysis.matches.append((gi, p))

    analysis.missed = [gi for gi in range(len(gold)) if gi not in used_gold]
    analysis.extra = [p for p in range(len(predicted_tasks)) if p not in used_pred]

    for gi, p in analysis.matches:
        predicted = predicted_tasks[p]
        analysis.assignee_correct[p] = same_person(gold[gi]["assignee"], predicted["assignee"])
        expected_date = gold[gi]["due_date"]
        analysis.date_correct[p] = expected_date == predicted["due_date"]

    # A trap is hit when a task the system invented (matching nothing expected) is based on a
    # sentence that was labelled "this is not a task".
    for p in analysis.extra:
        for ti, trap in enumerate(expected["not_task_quotes"]):
            if _best_overlap(trap, quotes[p]) >= OVERLAP_THRESHOLD:
                analysis.trap_hits.append((p, ti))
                break

    analysis.wrong = set(analysis.extra) | {
        p
        for _, p in analysis.matches
        if not (analysis.assignee_correct[p] and analysis.date_correct[p])
    }
    return analysis
