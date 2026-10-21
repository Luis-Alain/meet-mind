import pytest

from meet_mind.config import get_settings
from meet_mind.models.task import Evidence, Task
from meet_mind.services.decider import DeciderClient

pytestmark = pytest.mark.decider


def test_real_server_classifies_a_clear_commitment():
    settings = get_settings()
    client = DeciderClient(settings.decider_base_url, settings.decider_confidence_threshold)
    assert client.is_available()
    task = Task(
        title="Cerrar la integración de autenticación",
        evidence=[
            Evidence(
                quote="Yo me encargo de cerrar la integración de autenticación el viernes.",
                segment_ids=["s0001"],
                speaker="Ana",
            )
        ],
    )
    outcome = client.classify_task(task)
    assert outcome.error is None
    assert outcome.statement_type == "task"
    assert outcome.latency_ms and outcome.model
