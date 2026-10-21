"""HTTP contracts: what clients send to the API and what they get back.

These are separate from the domain models on purpose. The domain models describe what the
system knows; these describe what the API promises, so one can change without breaking the other.
"""

from datetime import date, datetime
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from meet_mind.models.common import MeetingStatus, Priority, ReviewStatus, SourceFormat
from meet_mind.models.task import ExtractionResult, Task


class MeetingCreate(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    meeting_date: date | None = None
    participants: list[str] = Field(default_factory=list, max_length=100)
    transcript: str
    source_format: SourceFormat = SourceFormat.TEXT

    @field_validator("transcript")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("transcript must not be empty")
        return value

    @field_validator("participants")
    @classmethod
    def _clean_participants(cls, value: list[str]) -> list[str]:
        return [name.strip() for name in value if name.strip()]


class MeetingCreated(BaseModel):
    meeting_id: UUID
    title: str
    status: MeetingStatus
    segments_count: int


class MeetingDetail(BaseModel):
    """A meeting and, once processed, everything extracted from it."""

    meeting_id: UUID
    title: str
    meeting_date: date | None
    participants: list[str]
    source_format: SourceFormat
    status: MeetingStatus
    created_at: datetime
    segments_count: int
    results: ExtractionResult | None = None


class ProcessResponse(BaseModel):
    meeting_id: UUID
    status: MeetingStatus
    summary: str
    tasks_count: int
    decider_enabled: bool
    review_required_count: int


class TaskOut(Task):
    """A task as returned by the API.

    The domain model hides ``id`` from the schema the LLM fills in; clients need it to edit a
    task, so it is declared again here as a normal, documented field.
    """

    id: UUID


class TaskPatch(BaseModel):
    """Fields a person may change on a task. Anything omitted is left as it is.

    Sending ``null`` for ``assignee`` or ``due_date`` clears the value. Evidence is not editable:
    it is the record of where the task came from.
    """

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=500)
    description: str | None = None
    assignee: str | None = Field(default=None, min_length=1, max_length=200)
    due_date: date | None = None
    priority: Priority | None = None
    review_status: ReviewStatus | None = None
    needs_review: bool | None = None

    @model_validator(mode="after")
    def _at_least_one_field(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("at least one field must be provided")
        return self

    @model_validator(mode="after")
    def _title_not_null(self) -> Self:
        # ``title`` and the enums are required on a task, so null means "invalid", not "clear".
        for name in ("title", "priority", "review_status", "needs_review"):
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError(f"{name} cannot be null")
        return self
