from pathlib import Path

import pytest

from meet_mind.agent.factory import make_agent_factory
from meet_mind.config import get_settings
from meet_mind.models.common import SourceFormat
from meet_mind.models.meeting import Meeting
from meet_mind.services.extraction import extract_meeting
from meet_mind.services.ingestion import parse_transcript

pytestmark = pytest.mark.bedrock

FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample.txt"


def test_real_extraction_has_traceable_evidence():
    segments = parse_transcript(FIXTURE.read_text(), SourceFormat.TXT)
    meeting = Meeting(title="Sync", participants=["Ana", "Luis"], segments=segments)
    outcome = extract_meeting(meeting, make_agent_factory(get_settings()))
    ids = {s.id for s in segments}
    assert outcome.result.tasks
    for task in outcome.result.tasks:
        assert task.evidence
        assert all(sid in ids for e in task.evidence for sid in e.segment_ids)
