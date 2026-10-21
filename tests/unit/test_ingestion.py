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


MEET_EXPORT = """GOOGLE MEET TRANSCRIPT
Meeting: Sprint Sync
Date: October 21, 2026
Participants: Carlos, Sofía

[00:00:15] Carlos: Buenos días.
[00:01:05] Sofía: Hola, Carlos.
Seguimos con el segundo punto.
[1:02:03] Carlos: Cerramos aquí.
"""


def test_meeting_export_header_is_skipped_and_timestamps_become_start_times():
    segments = parse_txt(MEET_EXPORT)
    assert [s.speaker for s in segments] == ["Carlos", "Sofía", "Carlos"]
    assert [s.start for s in segments] == [15.0, 65.0, 3723.0]
    assert segments[0].text == "Buenos días."
    assert not any("Participants" in s.text or s.speaker == "Meeting" for s in segments)


def test_untimestamped_line_continues_the_previous_utterance():
    segments = parse_txt(MEET_EXPORT)
    assert segments[1].text == "Hola, Carlos. Seguimos con el segundo punto."


def test_parenthesised_timestamps_are_accepted():
    segments = parse_txt("(00:00:05) Ana: hola\n(00:00:09) Luis: adiós")
    assert [s.start for s in segments] == [5.0, 9.0]


def test_plain_notes_keep_one_segment_per_line():
    segments = parse_txt("Meeting: no es una cabecera aquí\nAna: hola")
    assert [s.speaker for s in segments] == ["Meeting", "Ana"]
    assert all(s.start is None for s in segments)


def test_bundled_dataset_transcripts_parse_cleanly():
    for name in ("standard_transcript.txt", "ambiguous_transcript.txt"):
        segments = parse_txt((FIXTURES.parent / "datasets" / name).read_text())
        assert segments and all(s.speaker and s.start is not None for s in segments), name
        assert {s.speaker for s in segments} <= {"Carlos", "Sofía", "Luis", "María"}
