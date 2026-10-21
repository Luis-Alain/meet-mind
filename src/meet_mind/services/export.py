"""Export of a processed meeting to provider-neutral JSON and CSV.

The formats mirror MeetMind's own task model and know nothing about Asana, Trello or Jira.
Importing into those tools needs a per-tool column mapping, which belongs in a separate adapter.
"""

import csv
import io
from typing import Any

from meet_mind.models.meeting import Meeting
from meet_mind.models.task import Evidence, ExtractionResult, Task

# Column order of the CSV; also what the README documents.
CSV_COLUMNS = [
    "task_id",
    "title",
    "description",
    "assignee",
    "assignee_status",
    "due_date",
    "due_date_status",
    "priority",
    "review_status",
    "needs_review",
    "evidence_quotes",
    "evidence_timestamps",
]

# Quotes and timestamps of a task are packed into one cell; this separates the entries.
_JOINER = " | "

# Characters that make a spreadsheet read a cell as a formula.
_FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")


def _safe_cell(value: str) -> str:
    """Neutralise spreadsheet formulas in text that came from a transcript.

    Task titles and quotes are derived from untrusted input. A cell such as
    ``=HYPERLINK("http://evil", "click")`` would run when the file is opened in Excel or
    Sheets, so such cells are prefixed with an apostrophe, which spreadsheets treat as "this is
    plain text". Only text columns are passed through here; dates and flags never start with
    these characters.
    """
    return f"'{value}" if value.startswith(_FORMULA_TRIGGERS) else value


def _format_time(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def _evidence_cells(evidence: list[Evidence]) -> tuple[str, str]:
    """Join the quotes and start times of a task's evidence into one cell each.

    The two cells have the same number of entries and stay aligned, so the n-th timestamp
    belongs to the n-th quote (empty when the source had no timestamps).
    """
    quotes = _JOINER.join(item.quote.replace("\n", " ") for item in evidence)
    times = _JOINER.join(_format_time(item.start) for item in evidence)
    return _safe_cell(quotes), times


def task_to_row(task: Task) -> dict[str, str]:
    quotes, times = _evidence_cells(task.evidence)
    return {
        "task_id": str(task.id),
        "title": _safe_cell(task.title),
        "description": _safe_cell(task.description or ""),
        "assignee": _safe_cell(task.assignee or ""),
        "assignee_status": task.assignee_status.value,
        "due_date": task.due_date.isoformat() if task.due_date else "",
        "due_date_status": task.due_date_status.value,
        "priority": task.priority.value,
        "review_status": task.review_status.value,
        "needs_review": "true" if task.needs_review else "false",
        "evidence_quotes": quotes,
        "evidence_timestamps": times,
    }


def export_csv(tasks: list[Task]) -> str:
    """One row per task. UTF-8, comma separated, RFC 4180 quoting, header always present."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(task_to_row(task) for task in tasks)
    return buffer.getvalue()


def export_json(meeting: Meeting, results: ExtractionResult) -> dict[str, Any]:
    """The meeting's metadata and its full current result, including edits made by people."""
    return {
        "meeting": {
            "id": str(meeting.id),
            "title": meeting.title,
            "meeting_date": meeting.meeting_date.isoformat() if meeting.meeting_date else None,
            "participants": meeting.participants,
            "status": meeting.status.value,
        },
        **results.model_dump(mode="json"),
    }
