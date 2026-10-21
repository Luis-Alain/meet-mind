from pathlib import Path

import pytest

from meet_mind.models.common import SourceFormat
from meet_mind.services.ingestion import (
    TranscriptTooLongError,
    parse_srt,
    parse_transcript,
    parse_txt,
    parse_vtt,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_txt_speakers_and_ids():
    segments = parse_txt((FIXTURES / "sample.txt").read_text())
    assert [s.id for s in segments] == ["s0001", "s0002", "s0003", "s0004"]
    assert segments[0].speaker == "Ana"
    assert segments[1].text.startswith("Yo me encargo")
    assert segments[0].start is None


def test_txt_without_speaker():
    segments = parse_txt("Una línea sin hablante\n\nOtra más")
    assert [s.speaker for s in segments] == [None, None]
    assert len(segments) == 2


def test_vtt_voice_tags_and_times():
    segments = parse_vtt((FIXTURES / "sample.vtt").read_text())
    assert len(segments) == 2
    assert segments[0].speaker == "Ana"
    assert segments[0].start == 1.0 and segments[0].end == 4.5
    assert "<v" not in segments[1].text


def test_srt_speaker_prefix_and_minutes():
    segments = parse_srt((FIXTURES / "sample.srt").read_text())
    assert segments[1].speaker == "Luis"
    assert segments[1].start == 65.0


def test_max_chars_enforced():
    with pytest.raises(TranscriptTooLongError):
        parse_transcript("x" * 11, SourceFormat.TEXT, max_chars=10)
