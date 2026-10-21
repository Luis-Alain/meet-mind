from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    aws_region: str = "us-east-1"
    aws_profile: str | None = None
    aws_bedrock_model_id: str = ""

    decider_enabled: bool = True
    decider_base_url: str = "http://127.0.0.1:8001"
    # A cold server (model weights not yet in memory) can take several seconds to answer once.
    decider_timeout_seconds: float = Field(default=30.0, gt=0)
    decider_model_id: str = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
    # Experimental: calibrate with evals, not a guarantee of correctness.
    decider_confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    database_url: str = "sqlite:///./meetmind.db"
    max_transcript_chars: int = Field(default=100_000, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
