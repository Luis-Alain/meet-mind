from enum import StrEnum


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"


class Provenance(StrEnum):
    LLM = "llm"
    DECIDER = "decider"
    RULE = "rule"
    USER = "user"


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
