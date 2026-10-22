"""Glue between MeetMind and the Strands Evals SDK: turns label files into SDK cases and the
real pipeline into SDK task functions."""

from dataclasses import dataclass, field
from pathlib import Path

from strands_evals import Case, eval_task

from evals.schema import REPO_ROOT, FragmentLabels, MeetingLabels
from meet_mind.agent.workflow import ProcessingOutcome, process_meeting, review_with_decider
from meet_mind.models.common import SourceFormat
from meet_mind.models.meeting import Meeting
from meet_mind.models.task import Evidence, Task
from meet_mind.services.decider import DeciderClient
from meet_mind.services.extraction import AgentFactory
from meet_mind.services.ingestion import parse_transcript
from meet_mind.services.validation import quote_is_verified

MODE_BASELINE = "baseline"  # extraction + deterministic validation only
MODE_HYBRID = "hybrid"  # the same, plus Decider's second opinion on each task


# --- meetings -------------------------------------------------------------------------------


def build_meeting_cases(labels: list[MeetingLabels]) -> list[Case]:
    """One SDK case per labelled meeting.

    The case input is only what the real system receives (the transcript and its metadata); the
    labels travel separately as the expected output, so the system can never see the answers.
    """
    cases = []
    for item in labels:
        text = (REPO_ROOT / item.transcript_file).read_text(encoding="utf-8")
        cases.append(
            Case(
                name=item.meeting,
                input={
                    "title": item.title,
                    "meeting_date": item.meeting_date.isoformat(),
                    "participants": item.participants,
                    "transcript": text,
                },
                expected_output=item.model_dump(mode="json"),
                metadata={"status": item.status.value},
            )
        )
    return cases


def _meeting_from_input(case_input: dict) -> Meeting:
    from datetime import date

    return Meeting(
        title=case_input["title"],
        meeting_date=date.fromisoformat(case_input["meeting_date"]),
        participants=case_input["participants"],
        source_format=SourceFormat.TXT,
        segments=parse_transcript(case_input["transcript"], SourceFormat.TXT),
    )


def _task_dict(task: Task, segments: dict) -> dict:
    return {
        "title": task.title,
        "assignee": task.assignee,
        "due_date": task.due_date.isoformat() if task.due_date else None,
        "evidence_quotes": [e.quote for e in task.evidence],
        "needs_review": task.needs_review,
        "review_reasons": task.review_reasons,
        "evidence_total": len(task.evidence),
        "evidence_unverified": sum(not quote_is_verified(e, segments) for e in task.evidence),
    }


def _call_dict(outcome) -> dict | None:
    if outcome is None:
        return None
    return {
        "statement_type": outcome.statement_type,
        "confidence": outcome.statement_type_confidence,
        "explicit_commitment": outcome.explicit_commitment,
        "evidence_sufficient": outcome.evidence_sufficient,
        "latency_ms": outcome.latency_ms,
        "abstained": outcome.abstained,
        "error": outcome.error,
        "model": outcome.model,
    }


@dataclass
class MeetingRunner:
    """Runs the pipeline once per meeting and serves it to both evaluation modes.

    Decider only adds review flags after extraction and validation have finished, so "without
    Decider" and "with Decider" are two readings of the same extraction. Extracting once keeps the
    model's run-to-run variation out of the comparison (and halves the cost).
    """

    agent_factory: AgentFactory
    decider: DeciderClient | None = None
    _extractions: dict[str, tuple[Meeting, ProcessingOutcome]] = field(default_factory=dict)
    outputs: dict[str, dict[str, dict]] = field(default_factory=dict)  # mode -> meeting -> output

    def _extraction(self, name: str, case_input: dict) -> tuple[Meeting, ProcessingOutcome]:
        if name not in self._extractions:
            meeting = _meeting_from_input(case_input)
            self._extractions[name] = (meeting, process_meeting(meeting, self.agent_factory, None))
        return self._extractions[name]

    def task_for(self, mode: str):
        """Build the SDK task function for one mode."""
        if mode == MODE_HYBRID and self.decider is None:
            raise ValueError("hybrid mode needs a Decider client")

        @eval_task()
        def run(case: Case) -> dict:
            meeting, outcome = self._extraction(case.name, case.input)
            tasks = outcome.extraction.result.tasks
            calls: list[dict | None] = []
            available = True
            if mode == MODE_HYBRID:
                tasks, decisions, available = review_with_decider(tasks, self.decider)
                calls = [_call_dict(d) for d in decisions]

            segments = {s.id: s for s in meeting.segments}
            metrics = outcome.extraction.metrics
            output = {
                "mode": mode,
                "tasks": [_task_dict(t, segments) for t in tasks],
                "usage": {
                    "input_tokens": metrics.input_tokens,
                    "output_tokens": metrics.output_tokens,
                    "extraction_latency_ms": metrics.latency_ms,
                },
                "decider": {"used": mode == MODE_HYBRID, "available": available, "calls": calls},
            }
            self.outputs.setdefault(mode, {})[case.name] = output
            return {"output": output}

        return run


# --- single sentences (Decider on its own) --------------------------------------------------


def build_fragment_cases(labels: list[FragmentLabels]) -> list[Case]:
    return [
        Case(
            name=item.id,
            input={
                "speaker": item.speaker,
                "quote": item.quote,
                "candidate_title": item.candidate_title or item.quote,
            },
            expected_output=item.model_dump(mode="json"),
            metadata={"meeting": item.meeting, "note": item.note},
        )
        for item in labels
    ]


def fragment_task(decider: DeciderClient, outputs: dict[str, dict]):
    """SDK task that asks Decider about one sentence, exactly as the pipeline would ask about a
    task whose evidence is that sentence."""

    @eval_task()
    def run(case: Case) -> dict:
        data = case.input
        candidate = Task(
            title=data["candidate_title"],
            evidence=[
                Evidence(quote=data["quote"], segment_ids=["fragment"], speaker=data["speaker"])
            ],
        )
        call = decider.classify_task(candidate)
        output = _call_dict(call)
        outputs[case.name] = output
        return {"output": output}

    return run


def resolve_path(name: str | Path) -> Path:
    path = Path(name)
    return path if path.is_absolute() else REPO_ROOT / path
