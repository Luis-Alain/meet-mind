"""Client for the local Strands Decider server.

The strands server exposes a single endpoint, ``POST /v1/systemone``. A request carries a ``state``
(the text to judge) and a dictionary of named ``questions``; the response has one answer per
question name:

* ``choice`` -> ``{"choice", "probabilities", "confidence"}``
* ``noul``   -> ``{"noul": p}`` where ``p`` is the probability that the statement is true
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from meet_mind.models.task import Task

logger = logging.getLogger(__name__)

# The local model has a 4096-token window; keep the state well below it so the question
# definitions still fit and the server never has to truncate silently.
MAX_STATE_CHARS = 6_000

STATEMENT_TYPES = {
    "task": "someone commits to doing an action, or is asked to do one",
    "decision": "the group agrees on something that is now settled",
    "suggestion": "a non-binding idea or proposal that nobody committed to",
    "risk": "a problem or blocker that may happen or is already happening",
    "question": "an open question that still needs an answer",
    "other": "small talk, status updates or anything not listed above",
}

# A yes/no probability close to 0.5 means the model cannot tell. Anything inside this band is
# treated as ambiguous instead of being forced into yes or no.
AMBIGUOUS_BAND = (0.35, 0.65)

# Probabilities beyond these bounds are clear enough to act on.
CLEAR_YES = AMBIGUOUS_BAND[1]
CLEAR_NO = AMBIGUOUS_BAND[0]


@dataclass
class DeciderOutcome:
    """Everything we learned (or failed to learn) from one classification call.

    The outcome is meant to be logged and compared against reference labels later, so it keeps
    the raw numbers as well as the interpreted flags.
    """

    statement_type: str | None = None
    statement_type_confidence: float | None = None
    # True when the statement type reached the confidence threshold and can be acted upon.
    statement_type_reliable: bool = False
    # Probability that the speaker explicitly committed to doing something.
    explicit_commitment: float | None = None
    # Probability that the quoted evidence is enough to justify the extracted item.
    evidence_sufficient: float | None = None
    model: str | None = None
    latency_ms: float | None = None
    error: str | None = None
    # True when the server could not be reached at all (not just a bad answer).
    unreachable: bool = False
    # True when the answer is missing, invalid, or too uncertain to be trusted.
    abstained: bool = False
    abstain_reasons: list[str] = field(default_factory=list)


def build_state(task: Task) -> str:
    """Render a task candidate and its evidence as the text Decider will judge."""
    lines = [f"Candidate task: {task.title}"]
    if task.description:
        lines.append(f"Description: {task.description}")
    lines.append("Evidence from the meeting transcript:")
    for item in task.evidence:
        speaker = f"{item.speaker}: " if item.speaker else ""
        lines.append(f'- {speaker}"{item.quote}"')
    return "\n".join(lines)[:MAX_STATE_CHARS]


def _questions() -> dict[str, dict[str, Any]]:
    return {
        "statement_type": {
            "type": "choice",
            "instructions": "What kind of statement is the quoted evidence?",
            "criteria": STATEMENT_TYPES,
        },
        "explicit_commitment": {
            "type": "noul",
            "instructions": "Does a speaker explicitly commit to doing the task, or is someone "
            "explicitly asked to do it?",
        },
        "evidence_sufficient": {
            "type": "noul",
            "instructions": "Is the quoted evidence enough, on its own, to justify the "
            "candidate task?",
        },
    }


class DeciderClient:
    def __init__(
        self,
        base_url: str,
        confidence_threshold: float = 0.6,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._threshold = confidence_threshold
        self._http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def close(self) -> None:
        self._http.close()

    def is_available(self) -> bool:
        """Cheap liveness probe used by readiness checks; it never raises."""
        try:
            return self._http.get("/openapi.json", timeout=2.0).is_success
        except httpx.HTTPError:
            return False

    def classify_task(self, task: Task) -> DeciderOutcome:
        """Ask Decider the three bounded questions about one extracted task.

        Failures never propagate: a dead server, a bad status or a malformed body produce an
        outcome with ``abstained=True`` so the caller can route the task to human review.
        """
        outcome = DeciderOutcome()
        try:
            response = self._http.post(
                "/v1/systemone", json={"state": build_state(task), "questions": _questions()}
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            return self._failed(outcome, f"decider unreachable: {exc!r}", unreachable=True)
        except (httpx.HTTPError, ValueError) as exc:
            return self._failed(outcome, f"decider request failed: {exc!r}")

        try:
            answers = body["answers"]
            choice = answers["statement_type"]
            outcome.statement_type = str(choice["choice"])
            outcome.statement_type_confidence = float(choice["confidence"])
            outcome.explicit_commitment = float(answers["explicit_commitment"]["noul"])
            outcome.evidence_sufficient = float(answers["evidence_sufficient"]["noul"])
            outcome.model = body.get("model")
            outcome.latency_ms = body.get("latency_ms")
        except (KeyError, TypeError, ValueError) as exc:
            return self._failed(outcome, f"invalid decider response: {exc!r}")

        if outcome.statement_type not in STATEMENT_TYPES:
            return self._failed(outcome, f"unknown statement type {outcome.statement_type!r}")
        self._flag_uncertainty(outcome)
        return outcome

    def _failed(self, outcome: DeciderOutcome, message: str, unreachable: bool = False):
        logger.warning(message)
        outcome.error = message
        outcome.unreachable = unreachable
        outcome.abstained = True
        outcome.abstain_reasons.append(message)
        return outcome

    def _flag_uncertainty(self, outcome: DeciderOutcome) -> None:
        """Mark the outcome as abstained when any answer is too close to a coin flip."""
        outcome.statement_type_reliable = (
            outcome.statement_type_confidence or 0.0
        ) >= self._threshold
        if not outcome.statement_type_reliable:
            outcome.abstain_reasons.append(
                f"statement type confidence {outcome.statement_type_confidence:.2f} is below "
                f"the threshold {self._threshold:.2f}"
            )
        for name in ("explicit_commitment", "evidence_sufficient"):
            probability = getattr(outcome, name)
            if CLEAR_NO < probability < CLEAR_YES:
                outcome.abstain_reasons.append(f"{name} is ambiguous (p={probability:.2f})")
        outcome.abstained = bool(outcome.abstain_reasons)
