from datetime import UTC, date, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from meet_mind.models.common import MeetingStatus, SourceFormat


class TranscriptSegment(BaseModel):
    id: str
    text: str = Field(min_length=1)
    speaker: str | None = None
    start: float | None = Field(default=None, ge=0, description="Seconds from meeting start")
    end: float | None = Field(default=None, ge=0)


class Meeting(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str = Field(min_length=1)
    meeting_date: date | None = None
    participants: list[str] = Field(default_factory=list)
    source_format: SourceFormat = SourceFormat.TEXT
    segments: list[TranscriptSegment] = Field(default_factory=list)
    status: MeetingStatus = MeetingStatus.CREATED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
