"""Deterministic checks applied to what the LLM extracted.

Nothing here depends on a probabilistic model: each rule either holds or it does not. The rules
exist because a language model can sound convincing while being wrong, so every claim that can
be verified against the transcript is verified, and anything doubtful is flagged for a person
instead of being silently trusted or silently deleted.

* Evidence: a quote must really appear in the segments it cites.
* Assignees: a person is kept only if the transcript names them or they are the one speaking.
* Dates: the status (explicit/inferred) is derived from the phrase actually spoken, and the
  resolved date is cross-checked against that phrase and the meeting date.
"""

import re
from dataclasses import dataclass
from datetime import date, timedelta

from meet_mind.models.common import FieldStatus
from meet_mind.models.meeting import Meeting, TranscriptSegment
from meet_mind.models.task import Evidence, ExtractionResult, Task

_MONTHS = (
    "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|"
    "diciembre|january|february|march|april|may|june|july|august|september|october|november|"
    "december"
)

# A date is "explicit" when the speaker gave a calendar date (2026-10-30, 30/10, "30 de octubre",
# "October 30"), as opposed to a relative expression such as "mañana" or "el viernes".
_EXPLICIT_DATE = re.compile(
    rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b"
    rf"|\b\d{{1,2}}[/-]\d{{1,2}}(?:[/-]\d{{2,4}})?\b"
    rf"|\b\d{{1,2}}\s+(?:de\s+)?(?:{_MONTHS})\b"
    rf"|\b(?:{_MONTHS})\s+\d{{1,2}}\b"
)

# Weekday names (Spanish and English) mapped to ``date.weekday()`` numbering (Monday = 0).
_WEEKDAYS = {
    "lunes": 0, "monday": 0,
    "martes": 1, "tuesday": 1,
    "miércoles": 2, "miercoles": 2, "wednesday": 2,
    "jueves": 3, "thursday": 3,
    "viernes": 4, "friday": 4,
    "sábado": 5, "sabado": 5, "saturday": 5,
    "domingo": 6, "sunday": 6,
}  # fmt: skip


# "tomorrow" / "mañana". In Spanish "mañana" also means "morning" ("por la mañana"), so it only
# counts as "tomorrow" when it is not preceded by an article or "esta".
_TOMORROW = re.compile(r"\btomorrow\b|(?<!la )(?<!esta )(?<!de la )\bma[ñn]ana\b")
_DAY_AFTER_TOMORROW = re.compile(r"day after tomorrow|pasado ma[ñn]ana")
_TODAY = re.compile(r"\btoday\b|\bhoy\b")
# "next Friday" / "el viernes que viene" is ambiguous (this week's or the following one), so no
# date is computed for it.
_NEXT_WEEK_PHRASE = re.compile(r"\bnext\b|\bpr[óo]xim[oa]\b|que viene|siguiente")
# "before Friday" can mean Friday itself or the day before, unless a time of day pins it down.
_BEFORE = re.compile(r"\bbefore\b|\bantes\b")
_TIME_OF_DAY = re.compile(r"\d{1,2}:\d{2}|\d\s*[ap]\.?m|noon|mediod[íi]a|medianoche|midnight")


@dataclass
class ValidationReport:
    """Counters describing what the rules changed, kept for logging and evaluation."""

    dropped_items: int = 0
    unverified_quotes: int = 0
    assignees_cleared: int = 0
    dates_flagged: int = 0
    # Dates the model got wrong (or left empty) that were recomputed from the spoken phrase.
    dates_corrected: int = 0


def normalize(text: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace for forgiving text comparison."""
    return re.sub(r"\W+", " ", text.casefold()).strip()


def _cited_text(evidence: Evidence, segments: dict[str, TranscriptSegment]) -> str:
    return " ".join(segments[sid].text for sid in evidence.segment_ids if sid in segments)


def quote_is_verified(evidence: Evidence, segments: dict[str, TranscriptSegment]) -> bool:
    """True when the quote appears (ignoring case and punctuation) in the segments it cites.

    The comparison is word-based rather than character-exact, so a model that drops a comma
    still passes, but a paraphrase or an invented sentence does not.
    """
    quote = normalize(evidence.quote)
    return bool(quote) and quote in normalize(_cited_text(evidence, segments))


def _names_match(a: str, b: str) -> bool:
    """Two names refer to the same person if they are equal or share the first name."""
    a_tokens, b_tokens = normalize(a).split(), normalize(b).split()
    return bool(a_tokens and b_tokens) and a_tokens[0] == b_tokens[0]


def _assignee_is_supported(
    assignee: str, evidence: list[Evidence], segments: dict[str, TranscriptSegment]
) -> bool:
    """An assignee is supported if named in a quote, or if they are the one who spoke.

    Being listed as a meeting participant is deliberately not enough: that would let the model
    pick any attendee. The name must be tied to this specific piece of evidence.
    """
    name = normalize(assignee)
    first_name = name.split()[0] if name else ""
    for item in evidence:
        quote = f" {normalize(item.quote)} "
        if f" {name} " in quote or (first_name and f" {first_name} " in quote):
            return True
        speakers = {item.speaker} | {
            segments[sid].speaker for sid in item.segment_ids if sid in segments
        }
        if any(speaker and _names_match(assignee, speaker) for speaker in speakers):
            return True
    return False


def _add_reason(task: Task, reasons: list[str], reason: str) -> None:
    if reason not in task.review_reasons and reason not in reasons:
        reasons.append(reason)


def resolve_relative_date(raw: str, meeting_date: date) -> date | None:
    """Turn a relative phrase into a calendar date, or None when it cannot be done reliably.

    Language models are poor at calendar arithmetic (they placed "before Thursday" on a Friday),
    while this is exact. Supported: today, tomorrow, the day after tomorrow, and a single weekday
    name, which means its next occurrence after the meeting. Anything else is left unresolved.
    """
    text = raw.casefold()
    if _NEXT_WEEK_PHRASE.search(text):
        return None
    if _DAY_AFTER_TOMORROW.search(text):
        return meeting_date + timedelta(days=2)
    if _TOMORROW.search(text):
        return meeting_date + timedelta(days=1)
    if _TODAY.search(text):
        return meeting_date

    weekdays = {n for word, n in _WEEKDAYS.items() if re.search(rf"\b{word}\b", text)}
    if len(weekdays) != 1:
        return None
    # Between 1 and 7 days ahead: a weekday said on that same weekday means the following week.
    days_ahead = (weekdays.pop() - meeting_date.weekday() - 1) % 7 + 1
    return meeting_date + timedelta(days=days_ahead)


def _recompute_relative_date(
    task: Task, meeting_date: date | None, report: ValidationReport, reasons: list[str]
) -> Task:
    """Replace the model's date with the one computed from the phrase, when they disagree."""
    raw = (task.due_date_raw or "").strip()
    if not raw or meeting_date is None or _EXPLICIT_DATE.search(raw.casefold()):
        return task
    resolved = resolve_relative_date(raw, meeting_date)
    if resolved is None:
        return task

    if _BEFORE.search(raw.casefold()) and not _TIME_OF_DAY.search(raw.casefold()):
        reasons.append(f"'{raw}' may mean the named day or the day before it; used the named day")
    if resolved == task.due_date:
        return task
    report.dates_corrected += 1
    return task.model_copy(update={"due_date": resolved, "due_date_status": FieldStatus.INFERRED})


def _check_dates(
    task: Task, meeting_date: date | None, evidence_text: str, reasons: list[str]
) -> Task:
    """Derive the date status from the spoken phrase and cross-check the resolved date."""
    if task.due_date is None:
        return task

    raw = (task.due_date_raw or "").strip()
    if not raw:
        # A date with no phrase behind it cannot be audited, so it is treated as a guess.
        status = FieldStatus.INFERRED
        reasons.append("due date has no supporting phrase in the transcript")
    else:
        status = (
            FieldStatus.EXPLICIT if _EXPLICIT_DATE.search(raw.casefold()) else FieldStatus.INFERRED
        )
        if normalize(raw) not in normalize(evidence_text):
            reasons.append("the date phrase does not appear in the cited evidence")

    # "el viernes" must resolve to a Friday; a mismatch means the model computed the wrong day.
    weekdays = {n for word, n in _WEEKDAYS.items() if re.search(rf"\b{word}\b", raw.casefold())}
    if status == FieldStatus.INFERRED and len(weekdays) == 1:
        if task.due_date.weekday() not in weekdays:
            reasons.append("the due date does not fall on the weekday that was mentioned")

    if meeting_date is None:
        if status == FieldStatus.INFERRED:
            reasons.append("a relative date cannot be checked without the meeting date")
    elif task.due_date < meeting_date:
        reasons.append("the due date is earlier than the meeting date")

    return task.model_copy(update={"due_date_status": status})


def validate_task(
    task: Task, meeting: Meeting, segments: dict[str, TranscriptSegment], report: ValidationReport
) -> Task:
    """Apply every rule to one task. The task is flagged or corrected, never deleted."""
    reasons: list[str] = []
    update: dict = {}

    task = _recompute_relative_date(task, meeting.meeting_date, report, reasons)

    unverified = sum(not quote_is_verified(e, segments) for e in task.evidence)
    if unverified:
        report.unverified_quotes += unverified
        reasons.append(f"{unverified} evidence quote(s) not found in the cited segments")

    if task.assignee and not _assignee_is_supported(task.assignee, task.evidence, segments):
        # Remove the claim rather than keep a doubtful name: an empty assignee is honest,
        # a wrong one sends work to the wrong person.
        update.update(assignee=None, assignee_status=FieldStatus.UNKNOWN)
        report.assignees_cleared += 1
        reasons.append(f"assignee '{task.assignee}' is not supported by the evidence")

    evidence_text = " ".join(_cited_text(e, segments) for e in task.evidence)
    date_reasons: list[str] = []
    checked = _check_dates(task, meeting.meeting_date, evidence_text, date_reasons)
    if checked.due_date_status != task.due_date_status:
        update["due_date_status"] = checked.due_date_status
    if date_reasons:
        report.dates_flagged += 1
        reasons.extend(date_reasons)

    if not reasons:
        return task.model_copy(update=update) if update else task
    merged = [*task.review_reasons, *[r for r in reasons if r not in task.review_reasons]]
    return task.model_copy(update={**update, "needs_review": True, "review_reasons": merged})


def _keep_verified[T](items: list[T], segments, report: ValidationReport) -> list[T]:
    """For items without a review flag (decisions, risks...), keep only verified evidence.

    Such an item is dropped when none of its evidence can be verified: showing a decision that
    nobody can find in the transcript would be worse than showing nothing.
    """
    kept = []
    for item in items:
        verified = [e for e in item.evidence if quote_is_verified(e, segments)]  # type: ignore[attr-defined]
        report.unverified_quotes += len(item.evidence) - len(verified)  # type: ignore[attr-defined]
        if verified:
            kept.append(item.model_copy(update={"evidence": verified}))  # type: ignore[attr-defined]
        else:
            report.dropped_items += 1
    return kept


def validate_result(
    result: ExtractionResult, meeting: Meeting
) -> tuple[ExtractionResult, ValidationReport]:
    """Validate a whole extraction and return the corrected result with a report."""
    report = ValidationReport()
    segments = {s.id: s for s in meeting.segments}
    validated = result.model_copy(
        update={
            "tasks": [validate_task(t, meeting, segments, report) for t in result.tasks],
            "decisions": _keep_verified(result.decisions, segments, report),
            "risks": _keep_verified(result.risks, segments, report),
            "insights": _keep_verified(result.insights, segments, report),
            "open_questions": _keep_verified(result.open_questions, segments, report),
        }
    )
    return validated, report
