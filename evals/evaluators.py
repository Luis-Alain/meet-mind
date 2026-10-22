"""Evaluators for the Strands Evals SDK.

Each evaluator looks at one case and returns a score between 0 and 1 plus the reason, so the
SDK can show and save per-case results. None of them calls a model: every judgement is a plain
comparison with a human label, which keeps the evaluation deterministic and free.
"""

from strands_evals.evaluators import Evaluator
from strands_evals.types.evaluation import EvaluationData, EvaluationOutput

from evals.matching import analyse_case
from evals.metrics import precision_recall_f1


def _titles(output: dict, indexes) -> list[str]:
    return [output["tasks"][i]["title"] for i in indexes]


class TaskExtractionEvaluator(Evaluator):
    """F1 of the tasks found in one meeting: did it find the real ones without inventing any?"""

    def __init__(self, pass_threshold: float = 0.8):
        super().__init__()
        self.pass_threshold = pass_threshold

    def evaluate(self, evaluation_case: EvaluationData) -> list[EvaluationOutput]:
        expected, output = evaluation_case.expected_output, evaluation_case.actual_output
        analysis = analyse_case(expected, output["tasks"])
        tp, fp, fn = len(analysis.matches), len(analysis.extra), len(analysis.missed)
        f1 = precision_recall_f1(tp, fp, fn)["f1"]
        score = 1.0 if f1 is None else f1  # nothing expected and nothing predicted is a success
        missed = [expected["tasks"][i]["evidence_quote"] for i in analysis.missed]
        return [
            EvaluationOutput(
                score=score,
                test_pass=score >= self.pass_threshold,
                reason=f"found {tp}, invented {fp}, missed {fn}",
                label=f"{tp}/{tp + fn} expected tasks found",
                metadata={"invented": _titles(output, analysis.extra), "missed": missed},
            )
        ]


class AssigneeEvaluator(Evaluator):
    """Of the tasks that were found, how many got the right person (or correctly no one)?"""

    def evaluate(self, evaluation_case: EvaluationData) -> list[EvaluationOutput]:
        expected, output = evaluation_case.expected_output, evaluation_case.actual_output
        analysis = analyse_case(expected, output["tasks"])
        if not analysis.matches:
            ok = not expected["tasks"]
            return [EvaluationOutput(score=float(ok), test_pass=ok, reason="no task to compare")]
        wrong = [i for i, correct in analysis.assignee_correct.items() if not correct]
        score = 1 - len(wrong) / len(analysis.matches)
        return [
            EvaluationOutput(
                score=score,
                test_pass=not wrong,
                reason=f"{len(analysis.matches) - len(wrong)}/{len(analysis.matches)} correct",
                metadata={
                    "wrong": [
                        {"task": output["tasks"][i]["title"], "got": output["tasks"][i]["assignee"]}
                        for i in wrong
                    ]
                },
            )
        ]


class DueDateEvaluator(Evaluator):
    """Same as the assignee check, for the due date (exact day, or correctly none)."""

    def evaluate(self, evaluation_case: EvaluationData) -> list[EvaluationOutput]:
        expected, output = evaluation_case.expected_output, evaluation_case.actual_output
        analysis = analyse_case(expected, output["tasks"])
        if not analysis.matches:
            ok = not expected["tasks"]
            return [EvaluationOutput(score=float(ok), test_pass=ok, reason="no task to compare")]
        wrong = [i for i, correct in analysis.date_correct.items() if not correct]
        score = 1 - len(wrong) / len(analysis.matches)
        return [
            EvaluationOutput(
                score=score,
                test_pass=not wrong,
                reason=f"{len(analysis.matches) - len(wrong)}/{len(analysis.matches)} correct",
                metadata={
                    "wrong": [
                        {"task": output["tasks"][i]["title"], "got": output["tasks"][i]["due_date"]}
                        for i in wrong
                    ]
                },
            )
        ]


class TrapEvaluator(Evaluator):
    """Did the system avoid turning "sounds like a task but is not" sentences into tasks?"""

    def evaluate(self, evaluation_case: EvaluationData) -> list[EvaluationOutput]:
        expected, output = evaluation_case.expected_output, evaluation_case.actual_output
        analysis = analyse_case(expected, output["tasks"])
        traps = expected["not_task_quotes"]
        if not traps:
            return [EvaluationOutput(score=1.0, test_pass=True, reason="no traps in this meeting")]
        hit = {trap for _, trap in analysis.trap_hits}
        return [
            EvaluationOutput(
                score=1 - len(hit) / len(traps),
                test_pass=not hit,
                reason=f"fell for {len(hit)} of {len(traps)} traps",
                metadata={"fell_for": [traps[i] for i in sorted(hit)]},
            )
        ]


class ReviewFlagEvaluator(Evaluator):
    """Does "needs review" point at the tasks that are really wrong?

    The score is the share of wrong tasks that were flagged. A wrong task that was not flagged
    would reach the user as if it were trustworthy, so any such task fails the case.
    """

    def evaluate(self, evaluation_case: EvaluationData) -> list[EvaluationOutput]:
        expected, output = evaluation_case.expected_output, evaluation_case.actual_output
        tasks = output["tasks"]
        analysis = analyse_case(expected, tasks)
        silent = [i for i in sorted(analysis.wrong) if not tasks[i]["needs_review"]]
        flagged_wrong = len(analysis.wrong) - len(silent)
        score = flagged_wrong / len(analysis.wrong) if analysis.wrong else 1.0
        return [
            EvaluationOutput(
                score=score,
                test_pass=not silent,
                reason=f"{flagged_wrong}/{len(analysis.wrong)} wrong tasks were flagged",
                metadata={"silent_errors": _titles(output, silent)},
            )
        ]


class DeciderTypeEvaluator(Evaluator):
    """Single-sentence check: did Decider pick the right statement type?"""

    def evaluate(self, evaluation_case: EvaluationData) -> list[EvaluationOutput]:
        expected, output = evaluation_case.expected_output, evaluation_case.actual_output
        if output["error"] is not None:
            return [EvaluationOutput(score=0.0, test_pass=False, reason=output["error"])]
        ok = output["statement_type"] == expected["statement_type"]
        return [
            EvaluationOutput(
                score=float(ok),
                test_pass=ok,
                reason=f"expected {expected['statement_type']}, got {output['statement_type']}",
                label=f"confidence {output['confidence']:.2f}",
                metadata={"abstained": output["abstained"], "note": expected.get("note")},
            )
        ]


class _YesNoEvaluator(Evaluator):
    """Shared logic for the two yes/no questions: a probability of 0.5 or more means yes."""

    key = ""

    def evaluate(self, evaluation_case: EvaluationData) -> list[EvaluationOutput]:
        expected, output = evaluation_case.expected_output, evaluation_case.actual_output
        if output["error"] is not None:
            return [EvaluationOutput(score=0.0, test_pass=False, reason=output["error"])]
        probability = output[self.key]
        ok = (probability >= 0.5) == expected[self.key]
        return [
            EvaluationOutput(
                score=float(ok),
                test_pass=ok,
                reason=f"expected {expected[self.key]}, probability {probability:.2f}",
                metadata={"note": expected.get("note")},
            )
        ]


class DeciderCommitmentEvaluator(_YesNoEvaluator):
    key = "explicit_commitment"


class DeciderEvidenceEvaluator(_YesNoEvaluator):
    key = "evidence_sufficient"
