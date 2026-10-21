"""Processing pipeline for one meeting.

Processing happens in three steps, always in this order:

1. The LLM extracts candidate items from the transcript.
2. Deterministic rules verify the candidates against the transcript (quotes, assignees, dates).
3. Decider, if enabled, gives a second opinion on each task. It is consulted from explicit points
   in this Python code (the LLM has no way to call it), so its contribution stays measurable:
   each task gets its own ``DeciderOutcome`` that can be compared with reference labels later.

Steps 2 and 3 only ever flag items for review or remove unverifiable claims; they never write
new content.
"""

from dataclasses import dataclass, field

from meet_mind.models.meeting import Meeting
from meet_mind.models.task import Task
from meet_mind.services.decider import CLEAR_NO, DeciderClient, DeciderOutcome
from meet_mind.services.extraction import AgentFactory, ExtractionOutcome, extract_meeting
from meet_mind.services.validation import ValidationReport, validate_result


@dataclass
class ProcessingOutcome:
    extraction: ExtractionOutcome
    validation: ValidationReport = field(default_factory=ValidationReport)
    # One entry per task, in the same order as ``extraction.result.tasks``.
    # ``None`` means Decider was not consulted for that task.
    decider_outcomes: list[DeciderOutcome | None] = field(default_factory=list)
    decider_used: bool = False
    # False when Decider was requested but the server could not be reached.
    decider_available: bool = True


def apply_decider_review(task: Task, outcome: DeciderOutcome) -> Task:
    """Flag a task for human review when Decider doubts it.

    Decider only adds review reasons; it never edits, removes or rewrites a task. A doubtful
    task stays visible so a person makes the final call.
    """
    reasons: list[str] = []
    if outcome.abstained:
        reasons.extend(f"decider abstained: {reason}" for reason in outcome.abstain_reasons)
    # Only trust the predicted type when the model was confident about it.
    if outcome.statement_type_reliable and outcome.statement_type != "task":
        reasons.append(f"decider classified the evidence as '{outcome.statement_type}'")
    # A clear "no" is a real signal; an ambiguous probability is already covered by abstention.
    if outcome.explicit_commitment is not None and outcome.explicit_commitment <= CLEAR_NO:
        reasons.append("decider found no explicit commitment in the evidence")
    if outcome.evidence_sufficient is not None and outcome.evidence_sufficient <= CLEAR_NO:
        reasons.append("decider considers the evidence insufficient")

    if not reasons:
        return task
    return task.model_copy(
        update={
            "needs_review": True,
            "review_reasons": [*task.review_reasons, *reasons],
        }
    )


def process_meeting(
    meeting: Meeting,
    agent_factory: AgentFactory,
    decider: DeciderClient | None = None,
) -> ProcessingOutcome:
    """Extract items from the meeting and, if a Decider client is given, double-check the tasks."""
    extraction = extract_meeting(meeting, agent_factory)
    # Validate before consulting Decider so it only judges evidence that survived the checks.
    extraction.result, validation = validate_result(extraction.result, meeting)
    outcome = ProcessingOutcome(
        extraction=extraction, validation=validation, decider_used=decider is not None
    )
    if decider is None:
        outcome.decider_outcomes = [None] * len(extraction.result.tasks)
        return outcome

    reviewed: list[Task] = []
    for task in extraction.result.tasks:
        if not outcome.decider_available:
            # The server already failed once; do not wait for a timeout on every task.
            outcome.decider_outcomes.append(None)
            reviewed.append(task)
            continue
        decision = decider.classify_task(task)
        outcome.decider_outcomes.append(decision)
        if decision.unreachable:
            # No answer at all is not evidence against the task, so it is left untouched.
            outcome.decider_available = False
            reviewed.append(task)
        else:
            reviewed.append(apply_decider_review(task, decision))

    extraction.result = extraction.result.model_copy(update={"tasks": reviewed})
    return outcome
