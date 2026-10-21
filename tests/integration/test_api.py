import io
from datetime import date
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from meet_mind.agent.factory import ModelNotConfiguredError
from meet_mind.config import Settings
from meet_mind.main import create_app
from meet_mind.models.common import FieldStatus
from meet_mind.models.task import Decision, Evidence, ExtractionResult, Task
from meet_mind.services.decider import DeciderClient

TRANSCRIPT = (
    "Ana: Necesitamos cerrar la integración de autenticación.\n"
    "Luis: Yo me encargo, la tengo lista el viernes."
)


class FakeAgent:
    """Stands in for the Bedrock agent and returns one fixed, well-formed extraction."""

    def __call__(self, prompt, structured_output_model=None):
        result = ExtractionResult(
            summary="Se acordó cerrar la autenticación.",
            tasks=[
                Task(
                    title="Cerrar la integración de autenticación",
                    assignee="Luis",
                    assignee_status=FieldStatus.EXPLICIT,
                    due_date=date(2026, 10, 23),
                    due_date_raw="el viernes",
                    due_date_status=FieldStatus.INFERRED,
                    evidence=[
                        Evidence(
                            quote="Yo me encargo, la tengo lista el viernes", segment_ids=["s0002"]
                        )
                    ],
                ),
                Task(  # quote that is not in the transcript -> flagged by validation
                    title="Tarea dudosa",
                    evidence=[Evidence(quote="esto nadie lo dijo", segment_ids=["s0001"])],
                ),
            ],
            decisions=[
                Decision(
                    text="Cerrar autenticación",
                    evidence=[Evidence(quote="cerrar la integración", segment_ids=["s0001"])],
                )
            ],
        )
        return SimpleNamespace(structured_output=result, metrics=None)


def _decider_body(task_type="task"):
    return {
        "model": "decider-test",
        "latency_ms": 3.0,
        "answers": {
            "statement_type": {
                "type": "choice",
                "choice": task_type,
                "probabilities": {task_type: 0.97},
                "confidence": 0.97,
            },
            "explicit_commitment": {"type": "noul", "noul": 0.95},
            "evidence_sufficient": {"type": "noul", "noul": 0.95},
        },
    }


def _decider(handler=None) -> DeciderClient:
    handler = handler or (lambda request: httpx.Response(200, json=_decider_body()))
    return DeciderClient("http://decider.test", transport=httpx.MockTransport(handler))


def _client(tmp_path, *, agent_factory=FakeAgent, decider=None, **settings) -> TestClient:
    options = {"aws_bedrock_model_id": "test-model", **settings}
    config = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path / 'api.db'}", **options)
    return TestClient(
        create_app(config, agent_factory=agent_factory, decider=decider or _decider())
    )


@pytest.fixture
def client(tmp_path):
    with _client(tmp_path) as test_client:
        yield test_client


def _create(client: TestClient, **overrides) -> str:
    payload = {
        "title": "Sync semanal",
        "meeting_date": "2026-10-19",
        "participants": ["Ana", "Luis"],
        "transcript": TRANSCRIPT,
        **overrides,
    }
    response = client.post("/api/v1/meetings", json=payload)
    assert response.status_code == 201, response.text
    return response.json()["meeting_id"]


# --- health ---------------------------------------------------------------------------------


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    body = client.get("/health/ready").json()
    assert body["status"] == "ok"
    assert body["checks"] == {"database": "ok", "bedrock_model": "ok", "decider": "ok"}


def test_ready_degrades_when_decider_is_down_and_fails_without_a_model(tmp_path):
    def down(request):
        raise httpx.ConnectError("refused")

    with _client(tmp_path, decider=_decider(down)) as degraded:
        response = degraded.get("/health/ready")
        assert response.status_code == 200 and response.json()["status"] == "degraded"

    with _client(tmp_path, aws_bedrock_model_id="") as unavailable:
        response = unavailable.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["checks"]["bedrock_model"] == "not_configured"


# --- creating meetings ----------------------------------------------------------------------


def test_create_and_get_unprocessed_meeting(client):
    meeting_id = _create(client)
    body = client.get(f"/api/v1/meetings/{meeting_id}").json()
    assert body["status"] == "created" and body["segments_count"] == 2
    assert body["results"] is None
    assert client.get(f"/api/v1/meetings/{meeting_id}/tasks").json() == []


@pytest.mark.parametrize(
    "payload",
    [
        {"title": "x", "transcript": "   "},
        {"title": "", "transcript": "Ana: hola"},
        {"title": "x"},
        {"title": "x", "transcript": "Ana: hola", "source_format": "docx"},
    ],
)
def test_invalid_meeting_payloads_are_rejected(client, payload):
    assert client.post("/api/v1/meetings", json=payload).status_code == 422


def test_transcript_over_the_limit_is_413(tmp_path):
    with _client(tmp_path, max_transcript_chars=20) as small:
        response = small.post(
            "/api/v1/meetings", json={"title": "x", "transcript": "Ana: " + "a" * 50}
        )
        assert response.status_code == 413


def test_unknown_meeting_is_404_and_bad_uuid_is_422(client):
    assert client.get("/api/v1/meetings/00000000-0000-0000-0000-000000000000").status_code == 404
    assert client.get("/api/v1/meetings/not-a-uuid").status_code == 422
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.post(f"/api/v1/meetings/{missing}/process").status_code == 404


# --- uploads --------------------------------------------------------------------------------


def _upload(client, name, content: bytes, **data):
    return client.post(
        "/api/v1/meetings/upload",
        files={"file": (name, io.BytesIO(content), "text/plain")},
        data={"title": "Subida", "participants": "Ana, Luis", **data},
    )


def test_upload_vtt_keeps_timestamps(client):
    vtt = b"WEBVTT\n\n00:00:01.000 --> 00:00:03.000\n<v Ana>Hola equipo</v>\n"
    response = _upload(client, "reunion.vtt", vtt)
    assert response.status_code == 201 and response.json()["segments_count"] == 1
    meeting = client.get(f"/api/v1/meetings/{response.json()['meeting_id']}").json()
    assert meeting["source_format"] == "vtt" and meeting["participants"] == ["Ana", "Luis"]


def test_upload_rejects_bad_type_encoding_and_empty_files(client):
    assert _upload(client, "notes.pdf", b"x").status_code == 415
    assert _upload(client, "a.txt", b"\xff\xfe\x00bad").status_code == 422
    assert _upload(client, "a.txt", b"  \n ").status_code == 422


def test_upload_too_large_is_413(tmp_path):
    with _client(tmp_path, max_transcript_chars=10) as small:
        assert _upload(small, "a.txt", b"Ana: " + b"a" * 100).status_code == 413


# --- processing -----------------------------------------------------------------------------


def test_process_stores_results_and_flags_doubtful_tasks(client):
    meeting_id = _create(client)
    response = client.post(f"/api/v1/meetings/{meeting_id}/process")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "completed" and body["tasks_count"] == 2
    assert body["decider_enabled"] is True
    assert body["review_required_count"] == 1  # only the task with an invented quote

    detail = client.get(f"/api/v1/meetings/{meeting_id}").json()
    assert detail["status"] == "completed"
    assert detail["results"]["summary"] == "Se acordó cerrar la autenticación."
    assert len(detail["results"]["decisions"]) == 1

    tasks = client.get(f"/api/v1/meetings/{meeting_id}/tasks").json()
    assert [t["needs_review"] for t in tasks] == [False, True]
    assert all(t["id"] for t in tasks)


def test_processing_twice_needs_force(client):
    meeting_id = _create(client)
    assert client.post(f"/api/v1/meetings/{meeting_id}/process").status_code == 200
    assert client.post(f"/api/v1/meetings/{meeting_id}/process").status_code == 409
    assert client.post(f"/api/v1/meetings/{meeting_id}/process?force=true").status_code == 200


def test_decider_down_does_not_block_processing(tmp_path):
    def down(request):
        raise httpx.ConnectError("refused")

    with _client(tmp_path, decider=_decider(down)) as degraded:
        meeting_id = _create(degraded)
        body = degraded.post(f"/api/v1/meetings/{meeting_id}/process").json()
        assert body["tasks_count"] == 2 and body["decider_enabled"] is False


def test_decider_disabled_by_configuration(tmp_path):
    with _client(tmp_path, decider_enabled=False) as off:
        meeting_id = _create(off)
        assert off.post(f"/api/v1/meetings/{meeting_id}/process").json()["decider_enabled"] is False


def test_model_failure_is_502_without_leaking_details_and_can_be_retried(tmp_path):
    class Broken:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("secret-arn:aws:bedrock:123456789012")

    with _client(tmp_path, agent_factory=Broken) as failing:
        meeting_id = _create(failing)
        response = failing.post(f"/api/v1/meetings/{meeting_id}/process")
        assert response.status_code == 502 and "secret" not in response.text
        assert failing.get(f"/api/v1/meetings/{meeting_id}").json()["status"] == "failed"
        # A failed meeting can be processed again without force.
        assert failing.post(f"/api/v1/meetings/{meeting_id}/process").status_code == 502


def test_missing_model_configuration_is_503(tmp_path):
    def unconfigured():
        raise ModelNotConfiguredError("AWS_BEDROCK_MODEL_ID is not configured")

    with _client(tmp_path, agent_factory=unconfigured) as app:
        meeting_id = _create(app)
        assert app.post(f"/api/v1/meetings/{meeting_id}/process").status_code == 503


# --- editing tasks --------------------------------------------------------------------------


def _first_task_id(client) -> str:
    meeting_id = _create(client)
    client.post(f"/api/v1/meetings/{meeting_id}/process")
    return client.get(f"/api/v1/meetings/{meeting_id}/tasks").json()[0]["id"]


def test_patch_task_persists_and_marks_user_edit(client):
    task_id = _first_task_id(client)
    response = client.patch(
        f"/api/v1/tasks/{task_id}", json={"assignee": "Ana", "priority": "high"}
    )
    assert response.status_code == 200
    task = response.json()
    assert task["assignee"] == "Ana" and task["assignee_status"] == "explicit"
    assert task["priority"] == "high"
    assert task["review_status"] == "edited" and task["provenance"] == "user"
    assert task["evidence"], "evidence must survive an edit"


def test_patch_null_clears_and_omitted_fields_are_kept(client):
    task_id = _first_task_id(client)
    task = client.patch(f"/api/v1/tasks/{task_id}", json={"assignee": None}).json()
    assert task["assignee"] is None and task["assignee_status"] == "unknown"
    assert task["title"] == "Cerrar la integración de autenticación"
    assert task["due_date"] == "2026-10-23"


@pytest.mark.parametrize(
    "payload",
    [{}, {"title": ""}, {"title": None}, {"priority": "urgent"}, {"evidence": []}, {"nope": 1}],
)
def test_invalid_patches_are_422(client, payload):
    task_id = _first_task_id(client)
    assert client.patch(f"/api/v1/tasks/{task_id}", json=payload).status_code == 422


def test_patch_unknown_task_is_404(client):
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.patch(f"/api/v1/tasks/{missing}", json={"title": "x"}).status_code == 404


def test_data_survives_an_application_restart(tmp_path):
    with _client(tmp_path) as first:
        meeting_id = _create(first)
        first.post(f"/api/v1/meetings/{meeting_id}/process")
        task_id = first.get(f"/api/v1/meetings/{meeting_id}/tasks").json()[0]["id"]
        first.patch(f"/api/v1/tasks/{task_id}", json={"title": "Título editado"})

    with _client(tmp_path) as second:
        tasks = second.get(f"/api/v1/meetings/{meeting_id}/tasks").json()
        assert tasks[0]["title"] == "Título editado"
