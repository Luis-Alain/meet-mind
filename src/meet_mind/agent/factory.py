from collections.abc import Callable

from strands import Agent
from strands.models import BedrockModel

from meet_mind.agent.prompts import EXTRACTION_SYSTEM_PROMPT
from meet_mind.config import Settings

AgentFactory = Callable[[], Agent]


def build_extraction_agent(settings: Settings) -> Agent:
    """Create a fresh, stateless extraction agent (no tools: the LLM can execute nothing)."""
    if not settings.aws_bedrock_model_id:
        raise RuntimeError("AWS_BEDROCK_MODEL_ID is not configured")
    model = BedrockModel(
        model_id=settings.aws_bedrock_model_id,
        region_name=settings.aws_region,
        temperature=0.0,
    )
    return Agent(
        model=model,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        tools=[],
        callback_handler=None,
    )


def make_agent_factory(settings: Settings) -> AgentFactory:
    return lambda: build_extraction_agent(settings)
