import json

import httpx
import pytest

from meet_mind.agent.workflow import apply_decider_review, process_meeting
from meet_mind.models.meeting import Meeting
from meet_mind.models.task import Evidence, ExtractionResult, Task
from meet_mind.services.decider import DeciderClient, DeciderOutcome, build_state
from meet_mind.services.ingestion import parse_txt


def _task(title: str = "Cerrar auth", segment: str = "s0001") -> Task:
    return Task(
        title=title,
        evidence=[Evidence(quote="yo me encargo", segment_ids=[segment], speaker="Luis")],
    )


def _body(task_type="task", confidence=0.95, commitment=0.9, sufficient=0.9) -> dict:
    return {
        "model": "strands-decider-test",
        "latency_ms": 12.5,
        "answers": {
            "statement_type": {
                "type": "choice",
                "choice": task_type,
                "probabilities": {task_type: confidence},
                "confidence": confidence,
            },
            "explicit_commitment": {"type": "noul", "noul": commitment},
            "evidence_sufficient": {"type": "noul", "noul": sufficient},
        },
        "usage": {"input_tokens": 100, "output_tokens": 2},
    }


def _client(handler, threshold=0.85) -> DeciderClient:
    return DeciderClient("http://decider.test", threshold, transport=httpx.MockTransport(handler))


def test_build_state_includes_title_speaker_and_quote():
    state = build_state(_task())
    assert "Cerrar auth" in state and 'Luis: "yo me encargo"' in state


def test_confident_task_is_not_abstained():
    client = _client(lambda request: httpx.Response(200, json=_body()))
    outcome = client.classify_task(_task())
    assert outcome.statement_type == "task" and outcome.statement_type_reliable
    assert not outcome.abstained
    assert outcome.model == "strands-decider-test" and outcome.latency_ms == 12.5


def test_request_shape_matches_server_contract():
    seen = {}

    def handler(request):
        seen["json"] = json.loads(request.content)
        return httpx.Response(200, json=_body())

    _client(handler).classify_task(_task())
    questions = seen["json"]["questions"]
    assert questions["statement_type"]["type"] == "choice"
    assert "task" in questions["statement_type"]["criteria"]
    assert questions["explicit_commitment"]["type"] == "noul"


def test_low_confidence_and_ambiguous_probability_abstain():
    client = _client(lambda r: httpx.Response(200, json=_body(confidence=0.6, commitment=0.5)))
    outcome = client.classify_task(_task())
    assert outcome.abstained and not outcome.statement_type_reliable
    assert len(outcome.abstain_reasons) == 2


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"answers": {}}),
        httpx.Response(200, json=_body(task_type="banana")),
    ],
)
def test_invalid_responses_abstain_without_raising(response):
    outcome = _client(lambda r: response).classify_task(_task())
    assert outcome.abstained and outcome.error and not outcome.unreachable


def test_unreachable_server_is_reported():
    def handler(request):
        raise httpx.ConnectError("refused")

    outcome = _client(handler).classify_task(_task())
    assert outcome.unreachable and outcome.abstained
    assert _client(handler).is_available() is False


def test_review_flags_only_clear_negative_signals():
    good = DeciderOutcome(
        statement_type="task",
        statement_type_reliable=True,
        explicit_commitment=0.9,
        evidence_sufficient=0.9,
    )
    assert apply_decider_review(_task(), good).needs_review is False

    bad = DeciderOutcome(
        statement_type="suggestion",
        statement_type_reliable=True,
        explicit_commitment=0.1,
        evidence_sufficient=0.1,
    )
    reviewed = apply_decider_review(_task(), bad)
    assert reviewed.needs_review and len(reviewed.review_reasons) == 3
    assert reviewed.title == "Cerrar auth"  # content is never rewritten


class _FakeAgent:
    def __call__(self, prompt, structured_output_model=None):
        result = ExtractionResult(tasks=[_task("Uno"), _task("Dos")])
        return type("R", (), {"structured_output": result, "metrics": None})()


def test_workflow_without_decider_leaves_tasks_untouched():
    meeting = Meeting(title="m", segments=parse_txt("Luis: yo me encargo"))
    outcome = process_meeting(meeting, _FakeAgent)
    assert not outcome.decider_used
    assert outcome.decider_outcomes == [None, None]
    assert not any(t.needs_review for t in outcome.extraction.result.tasks)


def test_workflow_stops_calling_dead_server_after_first_failure():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ConnectError("refused")

    meeting = Meeting(title="m", segments=parse_txt("Luis: yo me encargo"))
    outcome = process_meeting(meeting, _FakeAgent, _client(handler))
    assert len(calls) == 1
    assert outcome.decider_used and not outcome.decider_available
    assert not any(t.needs_review for t in outcome.extraction.result.tasks)


def test_workflow_applies_decider_review():
    meeting = Meeting(title="m", segments=parse_txt("Luis: yo me encargo"))
    client = _client(lambda r: httpx.Response(200, json=_body(task_type="question")))
    outcome = process_meeting(meeting, _FakeAgent, client)
    assert all(t.needs_review for t in outcome.extraction.result.tasks)
    assert all(o is not None and o.statement_type == "question" for o in outcome.decider_outcomes)
