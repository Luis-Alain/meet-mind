"""Turns raw transcript files into a uniform list of segments.

Every format (plain text, WebVTT, SRT) ends up as ``TranscriptSegment`` objects with a stable
id such as ``s0001``. Those ids are what the rest of the system uses to point back at the
exact place in the transcript where an extracted item comes from.
"""

import re

from meet_mind.models.common import SourceFormat
from meet_mind.models.meeting import TranscriptSegment

# Matches a cue timing line such as ``00:01:05,000 --> 00:01:08,000``. SRT uses a comma for the
# milliseconds, VTT a dot, and VTT may omit the hours.
_TIMECODE = re.compile(
    r"(?P<start>(?:\d+:)?\d{1,2}:\d{2}[.,]\d{3})\s*-->\s*(?P<end>(?:\d+:)?\d{1,2}:\d{2}[.,]\d{3})"
)
# WebVTT marks the speaker with a voice tag: ``<v Ana>text</v>``.
_VTT_VOICE = re.compile(r"^<v\s+([^>]+)>(.*)$", re.DOTALL)
# Any other inline markup (italics, colours, timestamps) that must not reach the model.
_TAG = re.compile(r"</?[^>]+>")
# ``Name: text`` at the start of a line. The name must start with a letter and stay short, so a
# sentence that merely contains a colon is not mistaken for a speaker label.
_SPEAKER_LINE = re.compile(r"^([^\W\d_][\w .'-]{0,40}?):\s+(\S.*)$")


# the transcription exceeds the limit.
class TranscriptTooLongError(ValueError):
    pass


# The format is not supported
class UnsupportedFormatError(ValueError):
    pass


# converts a timestamp into total seconds.
def _seconds(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


# generate a four digit identifier for each segment.
def _segment_id(index: int) -> str:
    return f"s{index:04d}"


# normalize the content before analize
def _normalize(content: str) -> str:
    return content.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")


# processes .txt transcripts, where each line represents a segment of conversation.
def parse_txt(content: str) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for line in _normalize(content).split("\n"):
        line = line.strip()
        if not line:
            continue
        match = _SPEAKER_LINE.match(line)
        speaker, text = (match.group(1).strip(), match.group(2).strip()) if match else (None, line)
        segment_id = _segment_id(len(segments) + 1)
        segments.append(TranscriptSegment(id=segment_id, text=text, speaker=speaker))
    return segments


# Shared parser for VTT and SRT: both are blank-line separated cue blocks made of an optional cue
# number, a timing line and one or more text lines.
def _parse_timed_blocks(content: str) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for block in re.split(r"\n\s*\n", _normalize(content).strip()):
        lines = [line for line in block.split("\n") if line.strip()]
        timing_idx = next((i for i, line in enumerate(lines) if _TIMECODE.search(line)), None)
        if timing_idx is None:
            continue  # WEBVTT header, NOTE blocks, etc.
        timing = _TIMECODE.search(lines[timing_idx])
        if timing is None:
            continue
        text = " ".join(line.strip() for line in lines[timing_idx + 1 :])
        # The speaker can come from a VTT voice tag or from a ``Name:`` prefix in the text.
        speaker = None
        voice = _VTT_VOICE.match(text)
        if voice:
            speaker, text = voice.group(1).strip(), voice.group(2)
        text = _TAG.sub("", text).strip()
        if not speaker:
            match = _SPEAKER_LINE.match(text)
            if match:
                speaker, text = match.group(1).strip(), match.group(2).strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                id=_segment_id(len(segments) + 1),
                text=text,
                speaker=speaker,
                start=_seconds(timing.group("start")),
                end=_seconds(timing.group("end")),
            )
        )
    return segments


def parse_vtt(content: str) -> list[TranscriptSegment]:
    return _parse_timed_blocks(content)


def parse_srt(content: str) -> list[TranscriptSegment]:
    return _parse_timed_blocks(content)


# Single entry point: enforces the size limit before parsing, so oversized input is rejected
# cheaply, and then picks the parser that matches the declared format.
def parse_transcript(
    content: str, source_format: SourceFormat, max_chars: int | None = None
) -> list[TranscriptSegment]:
    if max_chars is not None and len(content) > max_chars:
        raise TranscriptTooLongError(f"Transcript exceeds {max_chars} characters")
    parsers = {
        SourceFormat.TEXT: parse_txt,
        SourceFormat.TXT: parse_txt,
        SourceFormat.VTT: parse_vtt,
        SourceFormat.SRT: parse_srt,
    }
    try:
        parser = parsers[source_format]
    except KeyError:
        raise UnsupportedFormatError(source_format) from None
    return parser(content)
