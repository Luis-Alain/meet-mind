from enum import StrEnum


# State of the human review of an extracted item.
class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"


# Who or what produced a value, so reviewers know how much to trust it.
class Provenance(StrEnum):
    LLM = "llm"
    DECIDER = "decider"
    RULE = "rule"
    USER = "user"


# How well a field is backed by the transcript: stated, derived from context, or missing.
class FieldStatus(StrEnum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class Priority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class MeetingStatus(StrEnum):
    CREATED = "created"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SourceFormat(StrEnum):
    TEXT = "text"
    TXT = "txt"
    VTT = "vtt"
    SRT = "srt"
