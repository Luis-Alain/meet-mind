from datetime import date

import pytest

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


def test_wrong_weekday_is_corrected_from_the_phrase():
    task = _task(
        due_date=date(2026, 10, 22),  # a Thursday, but the phrase says Friday
        due_date_raw="el viernes",
        due_date_status=FieldStatus.INFERRED,
    )
    result, report = validate_result(ExtractionResult(tasks=[task]), _meeting())
    assert result.tasks[0].due_date == date(2026, 10, 23)
    assert not result.tasks[0].needs_review and report.dates_corrected == 1


def test_weekday_mismatch_is_flagged_when_it_cannot_be_recomputed():
    # Without the meeting date a relative phrase cannot be resolved, only cross-checked.
    task = _task(
        due_date=date(2026, 10, 22),
        due_date_raw="el viernes",
        due_date_status=FieldStatus.INFERRED,
    )
    checked = _validate_one(task, _meeting(meeting_date=None))
    assert any("weekday" in r for r in checked.review_reasons)


def test_missing_date_is_filled_from_a_resolvable_phrase_but_not_from_a_vague_one():
    # A phrase that is not in the cited evidence still gets its date, but the task is flagged.
    unsupported = _task(
        due_date_raw="mañana", evidence=[Evidence(quote="mañana", segment_ids=["s0002"])]
    )
    checked = _validate_one(unsupported)
    assert checked.due_date == date(2026, 10, 20) and checked.needs_review
    spoken = _task(
        due_date_raw="el viernes",
        evidence=[Evidence(quote="la tengo lista el viernes", segment_ids=["s0002"])],
    )
    filled = _validate_one(spoken)
    assert filled.due_date == date(2026, 10, 23) and filled.due_date_status == FieldStatus.INFERRED
    vague = _task(due_date_raw="esta tarde")
    assert _validate_one(vague).due_date is None


def test_before_weekday_is_flagged_as_ambiguous_unless_a_time_is_given():
    plain = _task(
        due_date=date(2026, 10, 23),
        due_date_raw="antes del viernes",
        due_date_status=FieldStatus.INFERRED,
        evidence=[Evidence(quote="antes del viernes", segment_ids=["s0001"])],
    )
    meeting = _meeting()
    meeting.segments[0].text = "Hay que terminarlo antes del viernes"
    checked = _validate_one(plain, meeting)
    assert checked.due_date == date(2026, 10, 23)
    assert any("day before" in r for r in checked.review_reasons)

    timed = plain.model_copy(update={"due_date_raw": "antes del viernes a las 12:00"})
    meeting.segments[0].text = "Hay que terminarlo antes del viernes a las 12:00"
    timed = timed.model_copy(
        update={
            "evidence": [Evidence(quote="antes del viernes a las 12:00", segment_ids=["s0001"])]
        }
    )
    assert not any("day before" in r for r in _validate_one(timed, meeting).review_reasons)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("tomorrow at 9:00 a.m.", date(2026, 10, 22)),
        ("mañana a las 9", date(2026, 10, 22)),
        ("pasado mañana", date(2026, 10, 23)),
        ("day after tomorrow", date(2026, 10, 23)),
        ("hoy", date(2026, 10, 21)),
        ("before Thursday at noon", date(2026, 10, 22)),
        ("before Friday", date(2026, 10, 23)),
        ("el viernes", date(2026, 10, 23)),
        ("el miércoles", date(2026, 10, 28)),  # same weekday as the meeting -> following week
        ("por la mañana", None),  # "morning", not "tomorrow"
        ("esta tarde", None),
        ("next Friday", None),
        ("el viernes que viene", None),
        ("el lunes o el martes", None),
    ],
)
def test_resolve_relative_date(phrase, expected):
    from meet_mind.services.validation import resolve_relative_date

    assert resolve_relative_date(phrase, date(2026, 10, 21)) == expected  # a Wednesday


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
