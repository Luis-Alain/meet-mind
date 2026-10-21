import csv
import io
from datetime import date

from meet_mind.models.common import FieldStatus
from meet_mind.models.meeting import Meeting
from meet_mind.models.task import Evidence, ExtractionResult, Task
from meet_mind.services.export import CSV_COLUMNS, export_csv, export_json


def _task(**kwargs) -> Task:
    defaults = {
        "title": "Cerrar auth",
        "evidence": [Evidence(quote="yo me encargo", segment_ids=["s0001"])],
    }
    return Task(**{**defaults, **kwargs})


def _read(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def test_csv_has_documented_header_even_without_tasks():
    assert export_csv([]).splitlines()[0].split(",") == CSV_COLUMNS
    assert _read(export_csv([])) == []


def test_csv_round_trips_with_standard_parser():
    task = _task(
        description='Con coma, y "comillas"\ny salto de línea',
        assignee="Luis",
        assignee_status=FieldStatus.EXPLICIT,
        due_date=date(2026, 10, 23),
        due_date_status=FieldStatus.INFERRED,
        needs_review=True,
    )
    [row] = _read(export_csv([task]))
    assert row["task_id"] == str(task.id)
    assert row["description"] == task.description  # quoting survives commas, quotes, newlines
    assert row["due_date"] == "2026-10-23" and row["needs_review"] == "true"
    assert row["assignee_status"] == "explicit" and row["priority"] == "unknown"


def test_empty_fields_are_empty_strings_not_none():
    [row] = _read(export_csv([_task()]))
    assert row["assignee"] == "" and row["due_date"] == "" and row["description"] == ""
    assert row["needs_review"] == "false"


def test_accents_and_non_latin_text_survive_utf8():
    text = export_csv([_task(title="Revisión del diseño — 日本語")])
    assert _read(text)[0]["title"] == "Revisión del diseño — 日本語"
    assert text.encode("utf-8").decode("utf-8") == text


def test_evidence_quotes_and_timestamps_stay_aligned():
    task = _task(
        evidence=[
            Evidence(quote="primera", segment_ids=["s1"], start=65),
            Evidence(quote="segunda", segment_ids=["s2"]),
            Evidence(quote="tercera", segment_ids=["s3"], start=3725),
        ]
    )
    [row] = _read(export_csv([task]))
    assert row["evidence_quotes"].split(" | ") == ["primera", "segunda", "tercera"]
    assert row["evidence_timestamps"].split(" | ") == ["00:01:05", "", "01:02:05"]


def test_formula_cells_are_neutralised():
    task = _task(
        title='=HYPERLINK("http://evil.test","click")',
        description="+cmd",
        assignee="@user",
        assignee_status=FieldStatus.EXPLICIT,
        evidence=[Evidence(quote="-2+3", segment_ids=["s1"])],
    )
    [row] = _read(export_csv([task]))
    for column in ("title", "description", "assignee", "evidence_quotes"):
        assert row[column].startswith("'"), column
    assert row["title"][1:] == task.title  # the original text is still there


def test_json_contains_meeting_metadata_and_every_section():
    meeting = Meeting(title="Sync", meeting_date=date(2026, 10, 19), participants=["Ana"])
    payload = export_json(meeting, ExtractionResult(summary="Resumen", tasks=[_task()]))
    assert payload["meeting"] == {
        "id": str(meeting.id),
        "title": "Sync",
        "meeting_date": "2026-10-19",
        "participants": ["Ana"],
        "status": "created",
    }
    assert payload["summary"] == "Resumen"
    assert payload["tasks"][0]["id"]  # task ids are included so exports can be matched later
    assert set(payload) >= {"decisions", "risks", "insights", "open_questions"}
