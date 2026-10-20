from meet_mind.config import Settings


def test_defaults(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    settings = Settings(_env_file=None)
    assert settings.aws_region == "us-east-1"
    assert settings.decider_confidence_threshold == 0.85
    assert settings.max_transcript_chars == 100_000


def test_env_override(monkeypatch):
    monkeypatch.setenv("MAX_TRANSCRIPT_CHARS", "500")
    assert Settings(_env_file=None).max_transcript_chars == 500
