from meet_mind.models.meeting import Meeting
from meet_mind.services.extraction import AgentFactory, ExtractionOutcome, extract_meeting


def process_meeting(meeting: Meeting, agent_factory: AgentFactory) -> ExtractionOutcome:
    return extract_meeting(meeting, agent_factory)
