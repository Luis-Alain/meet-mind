from datetime import date
from typing import Self

from pydantic import BaseModel, Field, model_validator

from meet_mind.models.common import FieldStatus, Priority, Provenance, ReviewStatus


class Evidence(BaseModel):
    quote: str = Field(min_length=1, description="Literal text taken from the transcript")
    segment_ids: list[str] = Field(min_length=1)
    speaker: str | None = None
    start: float | None = Field(default=None, ge=0)


class Task(BaseModel):
    title: str = Field(min_length=1)
    description: str | None = None
    assignee: str | None = None
    assignee_status: FieldStatus = FieldStatus.UNKNOWN
    due_date: date | None = None
    due_date_raw: str | None = Field(default=None, description="Date as phrased in the transcript")
    due_date_status: FieldStatus = FieldStatus.UNKNOWN
    priority: Priority = Priority.UNKNOWN
    evidence: list[Evidence] = Field(min_length=1)
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.PENDING
    provenance: Provenance = Provenance.LLM

    @model_validator(mode="after")
    def _unknown_means_empty(self) -> Self:
        if (self.assignee is None) != (self.assignee_status == FieldStatus.UNKNOWN):
            raise ValueError("assignee must be null exactly when assignee_status is 'unknown'")
        if (self.due_date is None) != (self.due_date_status == FieldStatus.UNKNOWN):
            raise ValueError("due_date must be null exactly when due_date_status is 'unknown'")
        return self


class Decision(BaseModel):
    text: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)


class Risk(BaseModel):
    text: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)


class Insight(BaseModel):
    text: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)


class OpenQuestion(BaseModel):
    text: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)


class ExtractionResult(BaseModel):
    summary: str = ""
    tasks: list[Task] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    insights: list[Insight] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
