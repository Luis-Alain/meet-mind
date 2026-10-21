from datetime import date

from meet_mind.models.common import FieldStatus
from meet_mind.models.meeting import Meeting
from meet_mind.models.task import Decision, Evidence, ExtractionResult, Task
from meet_mind.services.ingestion import parse_txt
from meet_mind.services.validation import validate_result

TRANSCRIPT = (
    "Ana: Necesitamos cerrar la integración de autenticación.\n"
    "Luis: Yo me encargo, la tengo lista el viernes.\n"
    "Ana: Marta revisará el diseño antes del 30 de octubre."
)
# 2026-10-19 is a Monday, so Friday is 2026-10-23.
MEETING_DATE = date(2026, 10, 19)


def _meeting(meeting_date=MEETING_DATE) -> Meeting:
    return Meeting(
        title="Sync",
        meeting_date=meeting_date,
        participants=["Ana", "Luis", "Marta"],
        segments=parse_txt(TRANSCRIPT),
    )


def _task(**kwargs) -> Task:
    defaults = dict(
        title="Cerrar auth",
        evidence=[
            Evidence(quote="Yo me encargo, la tengo lista el viernes", segment_ids=["s0002"])
        ],
    )
    return Task(**{**defaults, **kwargs})


def _validate_one(task: Task, meeting: Meeting | None = None) -> Task:
    result, _ = validate_result(ExtractionResult(tasks=[task]), meeting or _meeting())
    return result.tasks[0]


def test_task_id_is_not_part_of_the_llm_schema():
    assert "id" not in Task.model_json_schema()["properties"]
    assert _task().id != _task().id


def test_clean_task_passes_untouched():
    task = _task(
        assignee="Luis",
        assignee_status=FieldStatus.EXPLICIT,
        due_date=date(2026, 10, 23),
        due_date_raw="el viernes",
        due_date_status=FieldStatus.INFERRED,
    )
    checked = _validate_one(task)
    assert not checked.needs_review and checked.assignee == "Luis"
    assert checked.due_date_status == FieldStatus.INFERRED


def test_unverified_quote_flags_task():
    task = _task(evidence=[Evidence(quote="frase inventada que nadie dijo", segment_ids=["s0002"])])
    checked = _validate_one(task)
    assert checked.needs_review
    assert "not found in the cited segments" in checked.review_reasons[0]


def test_quote_match_ignores_case_and_punctuation():
    task = _task(
        evidence=[Evidence(quote="YO ME ENCARGO la tengo lista el viernes!", segment_ids=["s0002"])]
    )
    assert not _validate_one(task).needs_review


def test_assignee_without_evidence_is_cleared():
    # Marta is a participant but is neither named in this evidence nor speaking.
    task = _task(assignee="Marta", assignee_status=FieldStatus.INFERRED)
    checked = _validate_one(task)
    assert checked.assignee is None and checked.assignee_status == FieldStatus.UNKNOWN
    assert checked.needs_review


def test_assignee_named_in_quote_is_kept():
    task = _task(
        assignee="Marta",
        assignee_status=FieldStatus.EXPLICIT,
        evidence=[Evidence(quote="Marta revisará el diseño", segment_ids=["s0003"])],
    )
    assert _validate_one(task).assignee == "Marta"


def test_speaker_counts_as_supported_assignee():
    task = _task(assignee="Luis Alain", assignee_status=FieldStatus.INFERRED)
    assert _validate_one(task).assignee == "Luis Alain"


def test_explicit_date_status_is_derived_from_phrase():
    task = _task(
        due_date=date(2026, 10, 30),
        due_date_raw="30 de octubre",
        due_date_status=FieldStatus.INFERRED,
        evidence=[
            Evidence(
                quote="Marta revisará el diseño antes del 30 de octubre", segment_ids=["s0003"]
            )
        ],
    )
    assert _validate_one(task).due_date_status == FieldStatus.EXPLICIT


def test_wrong_weekday_is_flagged():
    task = _task(
        due_date=date(2026, 10, 22),  # a Thursday
        due_date_raw="el viernes",
        due_date_status=FieldStatus.INFERRED,
    )
    checked = _validate_one(task)
    assert checked.needs_review
    assert any("weekday" in r for r in checked.review_reasons)


def test_date_before_meeting_and_missing_phrase_are_flagged():
    task = _task(due_date=date(2026, 10, 1), due_date_status=FieldStatus.EXPLICIT)
    reasons = " | ".join(_validate_one(task).review_reasons)
    assert "no supporting phrase" in reasons and "earlier than the meeting date" in reasons


def test_relative_date_without_meeting_date_is_flagged():
    task = _task(
        due_date=date(2026, 10, 23), due_date_raw="el viernes", due_date_status=FieldStatus.INFERRED
    )
    checked = _validate_one(task, _meeting(meeting_date=None))
    assert any("without the meeting date" in r for r in checked.review_reasons)


def test_non_task_items_keep_only_verified_evidence_and_are_dropped_when_empty():
    good = Decision(
        text="Usar Cognito",
        evidence=[Evidence(quote="cerrar la integración", segment_ids=["s0001"])],
    )
    bad = Decision(
        text="Inventada", evidence=[Evidence(quote="nunca se dijo", segment_ids=["s0001"])]
    )
    result, report = validate_result(ExtractionResult(decisions=[good, bad]), _meeting())
    assert [d.text for d in result.decisions] == ["Usar Cognito"]
    assert report.dropped_items == 1 and report.unverified_quotes == 1
