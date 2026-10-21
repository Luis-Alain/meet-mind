"""Persistence of meetings, extracted results and processing runs.

The repository speaks in domain models (``Meeting``, ``Task``, ``ExtractionResult``) and hides
the table layout, so callers never touch SQLAlchemy objects. Results are stored once and read
back from the database; viewing a meeting never calls a model again.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import delete, select

from meet_mind.models.common import FieldStatus, MeetingStatus, Provenance, ReviewStatus
from meet_mind.models.meeting import Meeting, TranscriptSegment
from meet_mind.models.task import (
    Decision,
    Evidence,
    ExtractionResult,
    Insight,
    OpenQuestion,
    Risk,
    Task,
)
from meet_mind.repositories.db import (
    Database,
    DeciderCallRow,
    ItemRow,
    MeetingRow,
    ProcessingRunRow,
    SegmentRow,
    TaskRow,
)

if TYPE_CHECKING:
    from meet_mind.agent.workflow import ProcessingOutcome

# Maps the name of each list in ``ExtractionResult`` to its model, and doubles as the ``kind``
# value stored in the shared ``items`` table.
_ITEM_KINDS: dict[str, type] = {
    "decisions": Decision,
    "risks": Risk,
    "insights": Insight,
    "open_questions": OpenQuestion,
}


def _aware(value: datetime) -> datetime:
    """SQLite drops time zones; everything we store is UTC, so put it back on the way out."""
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _task_from_row(row: TaskRow) -> Task:
    return Task(
        id=UUID(row.id),
        title=row.title,
        description=row.description,
        assignee=row.assignee,
        assignee_status=FieldStatus(row.assignee_status),
        due_date=row.due_date,
        due_date_raw=row.due_date_raw,
        due_date_status=FieldStatus(row.due_date_status),
        priority=row.priority,
        evidence=[Evidence.model_validate(e) for e in row.evidence],
        needs_review=row.needs_review,
        review_reasons=list(row.review_reasons),
        review_status=ReviewStatus(row.review_status),
        provenance=Provenance(row.provenance),
    )


def _task_to_row(task: Task, meeting_id: str, position: int) -> TaskRow:
    return TaskRow(
        id=str(task.id),
        meeting_id=meeting_id,
        position=position,
        title=task.title,
        description=task.description,
        assignee=task.assignee,
        assignee_status=task.assignee_status.value,
        due_date=task.due_date,
        due_date_raw=task.due_date_raw,
        due_date_status=task.due_date_status.value,
        priority=task.priority.value,
        needs_review=task.needs_review,
        review_reasons=task.review_reasons,
        review_status=task.review_status.value,
        provenance=task.provenance.value,
        evidence=[e.model_dump(mode="json") for e in task.evidence],
    )


class MeetingRepository:
    def __init__(self, database: Database) -> None:
        self._db = database

    # --- meetings -------------------------------------------------------------------------

    def create_meeting(self, meeting: Meeting) -> None:
        with self._db.session() as session:
            session.add(
                MeetingRow(
                    id=str(meeting.id),
                    title=meeting.title,
                    meeting_date=meeting.meeting_date,
                    participants=meeting.participants,
                    source_format=meeting.source_format.value,
                    status=meeting.status.value,
                    created_at=meeting.created_at,
                )
            )
            session.add_all(
                SegmentRow(
                    meeting_id=str(meeting.id),
                    id=segment.id,
                    position=position,
                    speaker=segment.speaker,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                )
                for position, segment in enumerate(meeting.segments)
            )

    def get_meeting(self, meeting_id: UUID) -> Meeting | None:
        with self._db.session() as session:
            row = session.get(MeetingRow, str(meeting_id))
            if row is None:
                return None
            segments = session.scalars(
                select(SegmentRow)
                .where(SegmentRow.meeting_id == row.id)
                .order_by(SegmentRow.position)
            ).all()
            return Meeting(
                id=UUID(row.id),
                title=row.title,
                meeting_date=row.meeting_date,
                participants=list(row.participants),
                source_format=row.source_format,
                status=MeetingStatus(row.status),
                created_at=_aware(row.created_at),
                segments=[
                    TranscriptSegment(
                        id=s.id, text=s.text, speaker=s.speaker, start=s.start, end=s.end
                    )
                    for s in segments
                ],
            )

    def set_status(self, meeting_id: UUID, status: MeetingStatus) -> None:
        with self._db.session() as session:
            row = session.get(MeetingRow, str(meeting_id))
            if row is not None:
                row.status = status.value

    # --- processing results ---------------------------------------------------------------

    def save_processing(self, meeting_id: UUID, outcome: "ProcessingOutcome", model_id: str) -> int:
        """Store the output of one pipeline run and return the run id.

        Reprocessing replaces the meeting's previous tasks and items (the latest result is
        what users see and edit), while the run record and its Decider calls are kept as history.
        """
        key = str(meeting_id)
        result = outcome.extraction.result
        metrics = outcome.extraction.metrics
        with self._db.session() as session:
            meeting = session.get(MeetingRow, key)
            if meeting is None:
                raise LookupError(f"meeting {key} does not exist")

            session.execute(delete(TaskRow).where(TaskRow.meeting_id == key))
            session.execute(delete(ItemRow).where(ItemRow.meeting_id == key))
            session.add_all(
                _task_to_row(task, key, position) for position, task in enumerate(result.tasks)
            )
            for kind in _ITEM_KINDS:
                session.add_all(
                    ItemRow(
                        meeting_id=key,
                        kind=kind,
                        position=position,
                        text=item.text,
                        evidence=[e.model_dump(mode="json") for e in item.evidence],
                    )
                    for position, item in enumerate(getattr(result, kind))
                )

            meeting.summary = result.summary
            meeting.status = MeetingStatus.COMPLETED.value

            run = ProcessingRunRow(
                meeting_id=key,
                model_id=model_id,
                decider_enabled=outcome.decider_used,
                decider_available=outcome.decider_available,
                chunks=metrics.chunks,
                attempts=metrics.attempts,
                input_tokens=metrics.input_tokens,
                output_tokens=metrics.output_tokens,
                extraction_latency_ms=metrics.latency_ms,
                validation=outcome.validation.__dict__,
            )
            session.add(run)
            session.flush()  # assigns run.id, needed by the Decider call rows below

            # Decider outcomes line up with the tasks by position.
            for task, decision in zip(result.tasks, outcome.decider_outcomes, strict=True):
                if decision is None:
                    continue
                session.add(
                    DeciderCallRow(
                        run_id=run.id,
                        task_id=str(task.id),
                        statement_type=decision.statement_type,
                        statement_type_confidence=decision.statement_type_confidence,
                        explicit_commitment=decision.explicit_commitment,
                        evidence_sufficient=decision.evidence_sufficient,
                        model=decision.model,
                        latency_ms=decision.latency_ms,
                        error=decision.error,
                        abstained=decision.abstained,
                        abstain_reasons=decision.abstain_reasons,
                    )
                )
            return run.id

    def get_results(self, meeting_id: UUID) -> ExtractionResult | None:
        """Rebuild the stored result of a meeting, or None if it was never processed."""
        key = str(meeting_id)
        with self._db.session() as session:
            meeting = session.get(MeetingRow, key)
            if meeting is None or meeting.status != MeetingStatus.COMPLETED.value:
                return None
            items = session.scalars(
                select(ItemRow).where(ItemRow.meeting_id == key).order_by(ItemRow.position)
            ).all()
            grouped: dict[str, list] = {kind: [] for kind in _ITEM_KINDS}
            for item in items:
                grouped[item.kind].append(
                    _ITEM_KINDS[item.kind](
                        text=item.text,
                        evidence=[Evidence.model_validate(e) for e in item.evidence],
                    )
                )
            return ExtractionResult(
                summary=meeting.summary, tasks=self._tasks(session, key), **grouped
            )

    # --- tasks ------------------------------------------------------------------------------

    @staticmethod
    def _tasks(session, meeting_id: str) -> list[Task]:
        rows = session.scalars(
            select(TaskRow).where(TaskRow.meeting_id == meeting_id).order_by(TaskRow.position)
        ).all()
        return [_task_from_row(row) for row in rows]

    def list_tasks(self, meeting_id: UUID) -> list[Task]:
        with self._db.session() as session:
            return self._tasks(session, str(meeting_id))

    def get_task(self, task_id: UUID) -> Task | None:
        with self._db.session() as session:
            row = session.get(TaskRow, str(task_id))
            return _task_from_row(row) if row else None

    def update_task(self, task_id: UUID, changes: dict[str, Any]) -> Task | None:
        """Apply a person's edit to a task and return the stored result.

        The merged task is validated again as a whole, so an edit can never leave it in an
        inconsistent state (for example an assignee that is still marked "unknown"). Values
        entered by a person are by definition explicit, and the task is attributed to the user.
        """
        with self._db.session() as session:
            row = session.get(TaskRow, str(task_id))
            if row is None:
                return None
            current = _task_from_row(row)

            merged = {**current.model_dump(), **changes}
            if "assignee" in changes and "assignee_status" not in changes:
                merged["assignee_status"] = (
                    FieldStatus.EXPLICIT if changes["assignee"] else FieldStatus.UNKNOWN
                )
            if "due_date" in changes and "due_date_status" not in changes:
                merged["due_date_status"] = (
                    FieldStatus.EXPLICIT if changes["due_date"] else FieldStatus.UNKNOWN
                )
            if "review_status" not in changes:
                merged["review_status"] = ReviewStatus.EDITED
            merged["provenance"] = Provenance.USER

            updated = Task.model_validate(merged)  # raises ValidationError if inconsistent
            replacement = _task_to_row(updated, row.meeting_id, row.position)
            for column in TaskRow.__table__.columns.keys():
                setattr(row, column, getattr(replacement, column))
            row.updated_at = datetime.now(UTC)
            return updated

    # --- history ----------------------------------------------------------------------------

    def list_decider_calls(self, run_id: int) -> list[DeciderCallRow]:
        with self._db.session() as session:
            return list(
                session.scalars(select(DeciderCallRow).where(DeciderCallRow.run_id == run_id))
            )
