"""Label files: the human-written ground truth the system is measured against.

There are two kinds, both stored as JSON Lines (one JSON object per line):

* meeting labels: which tasks a whole meeting should produce, and which sentences only *sound*
  like tasks but are not (the "traps");
* fragment labels: how Decider should classify a single sentence.

Every row has a ``status``. Only ``reviewed`` rows are used by default, so a draft prepared by
someone else (or a model) can never be mistaken for ground truth until a person has checked it.
"""

import json
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

DATASETS_DIR = Path(__file__).parent / "datasets"
REPO_ROOT = Path(__file__).parent.parent

StatementType = Literal["task", "decision", "suggestion", "risk", "question", "other"]


class LabelStatus(StrEnum):
    TEMPLATE = "template"  # skeleton with empty labels, waiting for a person to fill it in
    DRAFT = "draft"  # filled in, but nobody has verified it yet
    REVIEWED = "reviewed"  # checked by a person: usable as ground truth


class GoldTask(BaseModel):
    """A task the system is expected to find."""

    # A sentence copied from the transcript that proves the task. It is how a predicted task is
    # recognised as this one, so it does not have to match the system's wording of the title.
    evidence_quote: str = Field(min_length=1)
    assignee: str | None = None
    due_date: date | None = None
    commitment: Literal["firm", "conditional"] = "firm"


class MeetingLabels(BaseModel):
    meeting: str = Field(min_length=1)
    status: LabelStatus
    transcript_file: str  # path relative to the repository root
    title: str
    meeting_date: date
    participants: list[str]
    tasks: list[GoldTask] = Field(default_factory=list)
    # Sentences that sound like tasks but must NOT become one (vague ideas, small talk...).
    not_task_quotes: list[str] = Field(default_factory=list)


class FragmentLabels(BaseModel):
    id: str
    meeting: str
    status: LabelStatus
    speaker: str
    quote: str = Field(min_length=1)
    # What the extraction model might have called the task; defaults to the quote itself.
    candidate_title: str | None = None
    statement_type: StatementType | None = None
    explicit_commitment: bool | None = None
    evidence_sufficient: bool | None = None
    note: str | None = (
        None  # e.g. "borderline" for cases a reasonable person could label differently
    )


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _usable(rows, allow_draft: bool, source: Path):
    """Keep reviewed rows (and drafts if asked); templates are never usable."""
    allowed = {LabelStatus.REVIEWED} | ({LabelStatus.DRAFT} if allow_draft else set())
    usable = [row for row in rows if row.status in allowed]
    if not usable:
        statuses = sorted({row.status.value for row in rows})
        raise ValueError(
            f"{source.name} has no usable labels (statuses found: {statuses}). "
            "Fill in the labels and set status to 'reviewed', or pass --allow-draft "
            "to include drafts."
        )
    return usable


def load_meeting_labels(path: Path, allow_draft: bool = False) -> list[MeetingLabels]:
    rows = [MeetingLabels.model_validate(row) for row in _read_jsonl(path)]
    return _usable(rows, allow_draft, path)


def load_fragment_labels(path: Path, allow_draft: bool = False) -> list[FragmentLabels]:
    rows = [FragmentLabels.model_validate(row) for row in _read_jsonl(path)]
    usable = _usable(rows, allow_draft, path)
    for row in usable:
        if None in (row.statement_type, row.explicit_commitment, row.evidence_sufficient):
            raise ValueError(f"fragment {row.id} is marked {row.status.value} but has empty labels")
    return usable
