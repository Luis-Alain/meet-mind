import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from strands.types.exceptions import StructuredOutputException

from meet_mind.agent.prompts import CHUNK_PROMPT_TEMPLATE
from meet_mind.models.meeting import Meeting, TranscriptSegment
from meet_mind.models.task import Evidence, ExtractionResult

logger = logging.getLogger(__name__)

AgentFactory = Callable[[], Any]


@dataclass
class ExtractionMetrics:
    chunks: int = 0
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    dropped_items: int = 0
    failed_chunks: int = 0


@dataclass
class ExtractionOutcome:
    result: ExtractionResult
    metrics: ExtractionMetrics = field(default_factory=ExtractionMetrics)


def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total = int(seconds)
    return f" @{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def format_segment(segment: TranscriptSegment) -> str:
    speaker = segment.speaker or "?"
    text = segment.text.replace("</transcript>", "")
    return f"[{segment.id}] {speaker}{_fmt_time(segment.start)}: {text}"


def chunk_segments(
    segments: list[TranscriptSegment], max_chars: int = 12_000, overlap: int = 2
) -> list[list[TranscriptSegment]]:
    """Split segments into chunks of at most max_chars, repeating `overlap` segments."""
    chunks: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    size = 0
    for segment in segments:
        length = len(format_segment(segment)) + 1
        if current and size + length > max_chars:
            chunks.append(current)
            current = current[-overlap:] if overlap else []
            size = sum(len(format_segment(s)) + 1 for s in current)
        current.append(segment)
        size += length
    if current:
        chunks.append(current)
    return chunks


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def _enrich_evidence(
    evidence: list[Evidence], by_id: dict[str, TranscriptSegment]
) -> list[Evidence]:
    """Keep only evidence pointing at real segments and fill speaker/start from them."""
    kept: list[Evidence] = []
    for item in evidence:
        ids = [sid for sid in item.segment_ids if sid in by_id]
        if not ids:
            continue
        first = by_id[ids[0]]
        kept.append(
            item.model_copy(
                update={
                    "segment_ids": ids,
                    "speaker": item.speaker or first.speaker,
                    "start": item.start if item.start is not None else first.start,
                }
            )
        )
    return kept


def _sanitize(
    result: ExtractionResult, by_id: dict[str, TranscriptSegment], metrics: ExtractionMetrics
) -> ExtractionResult:
    """Drop items whose evidence does not point at the transcript."""
    updates: dict[str, Any] = {}
    for name in ("tasks", "decisions", "risks", "insights", "open_questions"):
        kept = []
        for item in getattr(result, name):
            evidence = _enrich_evidence(item.evidence, by_id)
            if evidence:
                kept.append(item.model_copy(update={"evidence": evidence}))
            else:
                metrics.dropped_items += 1
        updates[name] = kept
    return result.model_copy(update=updates)


def _merge_evidence(a: list[Evidence], b: list[Evidence]) -> list[Evidence]:
    seen = {(e.quote, tuple(e.segment_ids)) for e in a}
    return a + [e for e in b if (e.quote, tuple(e.segment_ids)) not in seen]


def consolidate(results: list[ExtractionResult]) -> ExtractionResult:
    """Merge per-chunk results, deduplicating by normalized title/text."""
    merged = ExtractionResult(summary=" ".join(r.summary for r in results if r.summary).strip())
    for name, key in (
        ("tasks", "title"),
        ("decisions", "text"),
        ("risks", "text"),
        ("insights", "text"),
        ("open_questions", "text"),
    ):
        index: dict[str, int] = {}
        items: list = []
        for result in results:
            for item in getattr(result, name):
                norm = _normalize(getattr(item, key))
                if norm in index:
                    existing = items[index[norm]]
                    items[index[norm]] = existing.model_copy(
                        update={"evidence": _merge_evidence(existing.evidence, item.evidence)}
                    )
                else:
                    index[norm] = len(items)
                    items.append(item)
        setattr(merged, name, items)
    return merged


def _chunk_prompt(meeting: Meeting, chunk: list[TranscriptSegment], position: tuple[int, int]):
    index, total = position
    note = (
        f"This is part {index} of {total} of the transcript; extract only what appears here.\n"
        if total > 1
        else ""
    )
    return CHUNK_PROMPT_TEMPLATE.format(
        title=meeting.title,
        meeting_date=meeting.meeting_date.isoformat() if meeting.meeting_date else "unknown",
        participants=", ".join(meeting.participants) or "unknown",
        chunk_note=note,
        transcript="\n".join(format_segment(s) for s in chunk),
    )


def _add_usage(metrics: ExtractionMetrics, agent_result: Any) -> None:
    usage = getattr(getattr(agent_result, "metrics", None), "accumulated_usage", None) or {}
    metrics.input_tokens += usage.get("inputTokens", 0)
    metrics.output_tokens += usage.get("outputTokens", 0)


def extract_meeting(
    meeting: Meeting,
    agent_factory: AgentFactory,
    *,
    max_chunk_chars: int = 12_000,
    overlap: int = 2,
    max_attempts: int = 2,
) -> ExtractionOutcome:
    """Run LLM extraction over the meeting transcript, chunking long inputs."""
    metrics = ExtractionMetrics()
    by_id = {s.id: s for s in meeting.segments}
    chunks = chunk_segments(meeting.segments, max_chunk_chars, overlap)
    metrics.chunks = len(chunks)
    partials: list[ExtractionResult] = []

    for number, chunk in enumerate(chunks, start=1):
        prompt = _chunk_prompt(meeting, chunk, (number, len(chunks)))
        for attempt in range(1, max_attempts + 1):
            metrics.attempts += 1
            started = time.perf_counter()
            try:
                agent_result = agent_factory()(prompt, structured_output_model=ExtractionResult)
            except StructuredOutputException:
                logger.warning("Invalid structured output (chunk %d, attempt %d)", number, attempt)
                continue
            finally:
                metrics.latency_ms += (time.perf_counter() - started) * 1000
            _add_usage(metrics, agent_result)
            partials.append(_sanitize(agent_result.structured_output, by_id, metrics))
            break
        else:
            metrics.failed_chunks += 1

    if chunks and metrics.failed_chunks == len(chunks):
        raise RuntimeError("Extraction failed: no chunk produced valid structured output")
    return ExtractionOutcome(result=consolidate(partials), metrics=metrics)
