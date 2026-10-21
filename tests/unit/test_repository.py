from datetime import date
from uuid import uuid4

import pytest
from pydantic import ValidationError

from meet_mind.agent.workflow import ProcessingOutcome
from meet_mind.models.common import FieldStatus, MeetingStatus, Provenance, ReviewStatus
from meet_mind.models.meeting import Meeting
from meet_mind.models.task import Decision, Evidence, ExtractionResult, Task
from meet_mind.repositories.db import Database
from meet_mind.repositories.meetings import MeetingRepository
from meet_mind.services.decider import DeciderOutcome
from meet_mind.services.extraction import ExtractionMetrics, ExtractionOutcome
from meet_mind.services.ingestion import parse_txt


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite:///{tmp_path / 'test.db'}"


def _repo(url: str) -> tuple[Database, MeetingRepository]:
    database = Database(url)
    database.create_all()
    return database, MeetingRepository(database)


def _meeting() -> Meeting:
    return Meeting(
        title="Sync",
        meeting_date=date(2026, 10, 19),
        participants=["Ana", "Luis"],
        segments=parse_txt("Ana: cerramos auth\nLuis: yo me encargo el viernes"),
    )


def _outcome(tasks: list[Task]) -> ProcessingOutcome:
    result = ExtractionResult(
        summary="Resumen",
        tasks=tasks,
        decisions=[
            Decision(text="Usar Cognito", evidence=[Evidence(quote="auth", segment_ids=["s0001"])])
        ],
    )
    decisions = [DeciderOutcome(statement_type="task", model="m", latency_ms=5.0)] * len(tasks)
    return ProcessingOutcome(
        extraction=ExtractionOutcome(
            result=result, metrics=ExtractionMetrics(chunks=1, attempts=1)
        ),
        decider_outcomes=decisions,
        decider_used=True,
    )


def _task() -> Task:
    return Task(
        title="Cerrar auth",
        assignee="Luis",
        assignee_status=FieldStatus.EXPLICIT,
        evidence=[Evidence(quote="yo me encargo", segment_ids=["s0002"], speaker="Luis")],
        needs_review=True,
        review_reasons=["why"],
    )


def test_meeting_round_trip_keeps_segment_order(db_url):
    _, repo = _repo(db_url)
    meeting = _meeting()
    repo.create_meeting(meeting)
    loaded = repo.get_meeting(meeting.id)
    assert loaded is not None
    assert [s.id for s in loaded.segments] == ["s0001", "s0002"]
    assert loaded.meeting_date == date(2026, 10, 19) and loaded.participants == ["Ana", "Luis"]
    assert loaded.created_at.tzinfo is not None
    assert repo.get_meeting(uuid4()) is None


def test_unprocessed_meeting_has_no_results(db_url):
    _, repo = _repo(db_url)
    meeting = _meeting()
    repo.create_meeting(meeting)
    assert repo.get_results(meeting.id) is None


def test_results_survive_a_restart(db_url):
    database, repo = _repo(db_url)
    meeting, task = _meeting(), _task()
    repo.create_meeting(meeting)
    run_id = repo.save_processing(meeting.id, _outcome([task]), "model-x")
    database.dispose()  # simulate shutting the backend down

    _, reopened = _repo(db_url)
    results = reopened.get_results(meeting.id)
    assert results is not None and results.summary == "Resumen"
    assert results.tasks[0].id == task.id and results.tasks[0].review_reasons == ["why"]
    assert results.tasks[0].evidence[0].quote == "yo me encargo"
    assert [d.text for d in results.decisions] == ["Usar Cognito"]
    assert reopened.get_meeting(meeting.id).status == MeetingStatus.COMPLETED
    calls = reopened.list_decider_calls(run_id)
    assert len(calls) == 1 and calls[0].task_id == str(task.id)


def test_reprocessing_replaces_results_but_keeps_run_history(db_url):
    _, repo = _repo(db_url)
    meeting = _meeting()
    repo.create_meeting(meeting)
    first = repo.save_processing(meeting.id, _outcome([_task()]), "m")
    second = repo.save_processing(meeting.id, _outcome([_task(), _task()]), "m")
    assert second != first
    assert len(repo.list_tasks(meeting.id)) == 2


def test_saving_for_unknown_meeting_fails(db_url):
    _, repo = _repo(db_url)
    with pytest.raises(LookupError):
        repo.save_processing(uuid4(), _outcome([]), "m")


def test_update_task_marks_it_as_edited_by_the_user(db_url):
    _, repo = _repo(db_url)
    meeting, task = _meeting(), _task()
    repo.create_meeting(meeting)
    repo.save_processing(meeting.id, _outcome([task]), "m")

    updated = repo.update_task(task.id, {"assignee": "Ana", "due_date": date(2026, 10, 30)})
    assert updated.assignee == "Ana" and updated.assignee_status == FieldStatus.EXPLICIT
    assert updated.due_date_status == FieldStatus.EXPLICIT
    assert updated.review_status == ReviewStatus.EDITED and updated.provenance == Provenance.USER
    assert repo.get_task(task.id) == updated  # the edit is persisted, not just returned


def test_update_task_can_clear_a_field_and_rejects_inconsistent_edits(db_url):
    _, repo = _repo(db_url)
    meeting, task = _meeting(), _task()
    repo.create_meeting(meeting)
    repo.save_processing(meeting.id, _outcome([task]), "m")

    cleared = repo.update_task(task.id, {"assignee": None})
    assert cleared.assignee is None and cleared.assignee_status == FieldStatus.UNKNOWN
    with pytest.raises(ValidationError):
        repo.update_task(task.id, {"title": ""})
    assert repo.update_task(uuid4(), {"title": "x"}) is None
