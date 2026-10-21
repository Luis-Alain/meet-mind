"""REST endpoints.

Handlers are deliberately thin: they validate input, call the services and repository, and turn
failures into the right HTTP status. Every dependency (database, model factory, Decider client)
is read from ``app.state`` so tests can replace them without touching the network.
"""

import logging
from datetime import date
from pathlib import PurePath
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from meet_mind.agent.factory import ModelNotConfiguredError
from meet_mind.agent.workflow import process_meeting
from meet_mind.api.schemas import (
    MeetingCreate,
    MeetingCreated,
    MeetingDetail,
    ProcessResponse,
    TaskOut,
    TaskPatch,
)
from meet_mind.config import Settings
from meet_mind.models.common import MeetingStatus, SourceFormat
from meet_mind.models.meeting import Meeting
from meet_mind.repositories.meetings import MeetingRepository
from meet_mind.services.ingestion import (
    TranscriptTooLongError,
    UnsupportedFormatError,
    parse_transcript,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# File extensions accepted by the upload endpoint, and the transcript format each implies.
_UPLOAD_FORMATS = {".txt": SourceFormat.TXT, ".vtt": SourceFormat.VTT, ".srt": SourceFormat.SRT}


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_repo(request: Request) -> MeetingRepository:
    return request.app.state.repo


def _not_found(what: str) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, f"{what} not found")


def _load_meeting(repo: MeetingRepository, meeting_id: UUID) -> Meeting:
    meeting = repo.get_meeting(meeting_id)
    if meeting is None:
        raise _not_found("meeting")
    return meeting


def _create_meeting(
    repo: MeetingRepository,
    settings: Settings,
    *,
    title: str,
    meeting_date: date | None,
    participants: list[str],
    transcript: str,
    source_format: SourceFormat,
) -> MeetingCreated:
    """Parse the transcript and store the meeting. Shared by the JSON and upload endpoints."""
    try:
        segments = parse_transcript(transcript, source_format, settings.max_transcript_chars)
    except TranscriptTooLongError as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(exc)) from exc
    except UnsupportedFormatError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    if not segments:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "the transcript contains no readable text"
        )

    meeting = Meeting(
        title=title,
        meeting_date=meeting_date,
        participants=participants,
        source_format=source_format,
        segments=segments,
    )
    repo.create_meeting(meeting)
    return MeetingCreated(
        meeting_id=meeting.id,
        title=meeting.title,
        status=meeting.status,
        segments_count=len(segments),
    )


# --- health -----------------------------------------------------------------------------------


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness: the process is up. It checks nothing else, so it never fails because of a
    dependency."""
    return {"status": "ok"}


@router.get("/health/ready")
def ready(request: Request, settings: Settings = Depends(get_settings)):
    """Readiness: are the things the backend needs actually usable?

    The database and the Bedrock model id are required. Decider is optional: when it is down the
    pipeline still works (tasks just skip the second opinion), so that is reported as "degraded"
    and does not make the backend unavailable.
    """
    checks = {
        "database": "ok" if request.app.state.database.ping() else "unavailable",
        "bedrock_model": "ok" if settings.aws_bedrock_model_id else "not_configured",
    }
    if settings.decider_enabled:
        decider = request.app.state.decider
        checks["decider"] = "ok" if decider and decider.is_available() else "unavailable"
    else:
        checks["decider"] = "disabled"

    required_ok = checks["database"] == "ok" and checks["bedrock_model"] == "ok"
    if not required_ok:
        overall, code = "unavailable", status.HTTP_503_SERVICE_UNAVAILABLE
    elif checks["decider"] == "unavailable":
        overall, code = "degraded", status.HTTP_200_OK
    else:
        overall, code = "ok", status.HTTP_200_OK
    return JSONResponse({"status": overall, "checks": checks}, status_code=code)


# --- meetings ---------------------------------------------------------------------------------


@router.post("/api/v1/meetings", status_code=status.HTTP_201_CREATED)
def create_meeting(
    body: MeetingCreate,
    repo: MeetingRepository = Depends(get_repo),
    settings: Settings = Depends(get_settings),
) -> MeetingCreated:
    return _create_meeting(
        repo,
        settings,
        title=body.title,
        meeting_date=body.meeting_date,
        participants=body.participants,
        transcript=body.transcript,
        source_format=body.source_format,
    )


@router.post("/api/v1/meetings/upload", status_code=status.HTTP_201_CREATED)
async def upload_meeting(
    file: UploadFile = File(...),
    title: str = Form(..., min_length=1, max_length=300),
    meeting_date: date | None = Form(None),
    participants: str = Form(""),
    repo: MeetingRepository = Depends(get_repo),
    settings: Settings = Depends(get_settings),
) -> MeetingCreated:
    """Create a meeting from a .txt, .vtt or .srt file (multipart/form-data).

    ``participants`` is a comma-separated list. The file is only ever read as transcript text:
    its contents are never interpreted as instructions.
    """
    extension = PurePath(file.filename or "").suffix.lower()
    source_format = _UPLOAD_FORMATS.get(extension)
    if source_format is None:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported file type '{extension}'; use .txt, .vtt or .srt",
        )

    # Read one byte past a generous byte limit so an oversized file is detected without ever
    # loading all of it. The limit is in bytes, the character limit is enforced after decoding.
    byte_limit = settings.max_transcript_chars * 4
    raw = await file.read(byte_limit + 1)
    if len(raw) > byte_limit:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "the file is too large")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "the file must be UTF-8 text"
        ) from exc
    if not text.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "the file is empty")

    names = [name.strip() for name in participants.split(",") if name.strip()]
    return _create_meeting(
        repo,
        settings,
        title=title,
        meeting_date=meeting_date,
        participants=names,
        transcript=text,
        source_format=source_format,
    )


@router.post("/api/v1/meetings/{meeting_id}/process")
def process(
    meeting_id: UUID,
    request: Request,
    force: bool = False,
    repo: MeetingRepository = Depends(get_repo),
    settings: Settings = Depends(get_settings),
) -> ProcessResponse:
    """Run the extraction pipeline synchronously and store the result.

    Processing a meeting costs model tokens, so an already processed meeting is refused unless
    ``force=true``. The claim below also stops two requests from processing the same meeting
    at once.
    """
    claim = repo.claim_for_processing(meeting_id, force=force)
    if claim == "missing":
        raise _not_found("meeting")
    if claim == "busy":
        raise HTTPException(status.HTTP_409_CONFLICT, "the meeting is already being processed")
    if claim == "done":
        raise HTTPException(
            status.HTTP_409_CONFLICT, "the meeting was already processed; use force=true to redo it"
        )

    meeting = _load_meeting(repo, meeting_id)
    decider = request.app.state.decider if settings.decider_enabled else None
    try:
        outcome = process_meeting(meeting, request.app.state.agent_factory, decider)
        repo.save_processing(meeting_id, outcome, settings.aws_bedrock_model_id)
    except ModelNotConfiguredError as exc:
        repo.set_status(meeting_id, MeetingStatus.FAILED)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except Exception as exc:
        # Provider errors can contain account details, so they go to the log, not the client.
        logger.exception("processing meeting %s failed", meeting_id)
        repo.set_status(meeting_id, MeetingStatus.FAILED)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "the extraction service failed; try again later"
        ) from exc

    result = outcome.extraction.result
    return ProcessResponse(
        meeting_id=meeting_id,
        status=MeetingStatus.COMPLETED,
        summary=result.summary,
        tasks_count=len(result.tasks),
        decider_enabled=outcome.decider_used and outcome.decider_available,
        review_required_count=sum(task.needs_review for task in result.tasks),
    )


@router.get("/api/v1/meetings/{meeting_id}")
def get_meeting(meeting_id: UUID, repo: MeetingRepository = Depends(get_repo)) -> MeetingDetail:
    meeting = _load_meeting(repo, meeting_id)
    return MeetingDetail(
        meeting_id=meeting.id,
        title=meeting.title,
        meeting_date=meeting.meeting_date,
        participants=meeting.participants,
        source_format=meeting.source_format,
        status=meeting.status,
        created_at=meeting.created_at,
        segments_count=len(meeting.segments),
        results=repo.get_results(meeting_id),
    )


@router.get("/api/v1/meetings/{meeting_id}/tasks")
def list_tasks(meeting_id: UUID, repo: MeetingRepository = Depends(get_repo)) -> list[TaskOut]:
    _load_meeting(repo, meeting_id)
    return [TaskOut.model_validate(task.model_dump()) for task in repo.list_tasks(meeting_id)]


# --- tasks ------------------------------------------------------------------------------------


@router.patch("/api/v1/tasks/{task_id}")
def patch_task(
    task_id: UUID, body: TaskPatch, repo: MeetingRepository = Depends(get_repo)
) -> TaskOut:
    # ``exclude_unset`` is what separates "leave it alone" (field omitted) from "clear it"
    # (field sent as null).
    changes = body.model_dump(exclude_unset=True)
    try:
        task = repo.update_task(task_id, changes)
    except ValidationError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, exc.errors(include_url=False)
        ) from exc
    if task is None:
        raise _not_found("task")
    return TaskOut.model_validate(task.model_dump())
