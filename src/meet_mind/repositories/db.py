"""Database engine and table definitions (SQLite by default).

The tables mirror the domain models, with two deliberate simplifications: lists such as a
task's evidence or review reasons are stored as JSON columns instead of child tables, because
they are always read and written together with their parent and never queried on their own.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class MeetingRow(Base):
    __tablename__ = "meetings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(300))
    meeting_date: Mapped[date | None] = mapped_column(Date)
    participants: Mapped[list] = mapped_column(JSON, default=list)
    source_format: Mapped[str] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(20))
    summary: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SegmentRow(Base):
    __tablename__ = "segments"

    meeting_id: Mapped[str] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), primary_key=True
    )
    id: Mapped[str] = mapped_column(String(20), primary_key=True)
    # Segments are returned in transcript order, which the id alone does not guarantee.
    position: Mapped[int] = mapped_column(Integer)
    speaker: Mapped[str | None] = mapped_column(String(200))
    start: Mapped[float | None] = mapped_column(Float)
    end: Mapped[float | None] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)


class TaskRow(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    meeting_id: Mapped[str] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    assignee: Mapped[str | None] = mapped_column(String(200))
    assignee_status: Mapped[str] = mapped_column(String(10))
    due_date: Mapped[date | None] = mapped_column(Date)
    due_date_raw: Mapped[str | None] = mapped_column(String(200))
    due_date_status: Mapped[str] = mapped_column(String(10))
    priority: Mapped[str] = mapped_column(String(10))
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    review_reasons: Mapped[list] = mapped_column(JSON, default=list)
    review_status: Mapped[str] = mapped_column(String(10))
    provenance: Mapped[str] = mapped_column(String(10))
    evidence: Mapped[list] = mapped_column(JSON, default=list)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ItemRow(Base):
    """Decisions, risks, insights and open questions share one table, told apart by ``kind``."""

    __tablename__ = "items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(20))
    position: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    evidence: Mapped[list] = mapped_column(JSON, default=list)


class ProcessingRunRow(Base):
    """One execution of the pipeline over a meeting, with its cost and configuration.

    Keeping every run (instead of overwriting) is what makes runs comparable later, for example
    the same meeting processed with and without Decider.
    """

    __tablename__ = "processing_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    model_id: Mapped[str] = mapped_column(String(200))
    decider_enabled: Mapped[bool] = mapped_column(Boolean)
    decider_available: Mapped[bool] = mapped_column(Boolean)
    chunks: Mapped[int] = mapped_column(Integer)
    attempts: Mapped[int] = mapped_column(Integer)
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    extraction_latency_ms: Mapped[float] = mapped_column(Float)
    validation: Mapped[dict] = mapped_column(JSON, default=dict)


class DeciderCallRow(Base):
    """One Decider classification, stored raw so it can be compared with reference labels."""

    __tablename__ = "decider_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("processing_runs.id", ondelete="CASCADE"), index=True
    )
    task_id: Mapped[str] = mapped_column(String(36))
    statement_type: Mapped[str | None] = mapped_column(String(20))
    statement_type_confidence: Mapped[float | None] = mapped_column(Float)
    explicit_commitment: Mapped[float | None] = mapped_column(Float)
    evidence_sufficient: Mapped[float | None] = mapped_column(Float)
    model: Mapped[str | None] = mapped_column(String(200))
    latency_ms: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    abstained: Mapped[bool] = mapped_column(Boolean)
    abstain_reasons: Mapped[list] = mapped_column(JSON, default=list)


class Database:
    """Owns the engine and hands out short-lived sessions."""

    def __init__(self, url: str) -> None:
        # FastAPI runs sync endpoints in a thread pool, so SQLite connections must be usable
        # from threads other than the one that created them.
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine: Engine = create_engine(url, connect_args=connect_args)
        if url.startswith("sqlite"):
            # SQLite ignores foreign keys (and therefore ON DELETE CASCADE) unless asked.
            event.listen(self.engine, "connect", _enable_sqlite_foreign_keys)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    def dispose(self) -> None:
        self.engine.dispose()

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Open a session that commits on success and rolls back on any error."""
        session = self._sessions()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


def _enable_sqlite_foreign_keys(dbapi_connection, _record) -> None:
    dbapi_connection.execute("PRAGMA foreign_keys=ON")
