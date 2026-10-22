import json
import re
from types import SimpleNamespace

import httpx
import pytest
from evals.aggregate import aggregate_decider, aggregate_pipeline
from evals.evaluators import (
    AssigneeEvaluator,
    DeciderTypeEvaluator,
    DueDateEvaluator,
    ReviewFlagEvaluator,
    TaskExtractionEvaluator,
    TrapEvaluator,
)
from evals.pipeline import MeetingRunner
from evals.run_decider_eval import run_decider_eval
from evals.run_pipeline_eval import run_pipeline_eval
from evals.schema import (
    DATASETS_DIR,
    REPO_ROOT,
    load_fragment_labels,
    load_meeting_labels,
)
from strands_evals import Case, Experiment, eval_task

from meet_mind.config import Settings
from meet_mind.models.common import FieldStatus
from meet_mind.models.task import Evidence, ExtractionResult, Task
from meet_mind.services.decider import DeciderClient
from meet_mind.services.validation import normalize


def _gold(quote, assignee="Luis", due=None):
    return {"evidence_quote": quote, "assignee": assignee, "due_date": due, "commitment": "firm"}


def _pred(quote, assignee="Luis", due=None, flagged=False, unverified=0):
    return {
        "title": "t",
        "assignee": assignee,
        "due_date": due,
        "evidence_quotes": [quote],
        "needs_review": flagged,
        "review_reasons": [],
        "evidence_total": 1,
        "evidence_unverified": unverified,
    }


def _output(tasks, calls=()):
    return {
        "tasks": tasks,
        "usage": {"input_tokens": 100, "output_tokens": 50, "extraction_latency_ms": 1000.0},
        "decider": {"used": bool(calls), "available": True, "calls": list(calls)},
    }


# --- the shipped label files ------------------------------------------------------------------


def test_dev_labels_load_only_as_drafts():
    with pytest.raises(ValueError, match="no usable labels"):
        load_meeting_labels(DATASETS_DIR / "meetings_dev.jsonl")
    assert len(load_meeting_labels(DATASETS_DIR / "meetings_dev.jsonl", allow_draft=True)) == 2
    assert load_fragment_labels(DATASETS_DIR / "decider_dev.jsonl", allow_draft=True)


@pytest.mark.parametrize("name", ["meetings_test.jsonl", "decider_test.jsonl"])
def test_templates_are_never_usable(name):
    loader = load_meeting_labels if name.startswith("meetings") else load_fragment_labels
    with pytest.raises(ValueError, match="no usable labels"):
        loader(DATASETS_DIR / name, allow_draft=True)


def test_every_label_quote_really_appears_in_its_transcript():
    for labels in load_meeting_labels(DATASETS_DIR / "meetings_dev.jsonl", allow_draft=True):
        transcript = normalize((REPO_ROOT / labels.transcript_file).read_text())
        quotes = [t.evidence_quote for t in labels.tasks] + labels.not_task_quotes
        for quote in quotes:
            assert normalize(quote) in transcript, (labels.meeting, quote)
    for fragment in load_fragment_labels(DATASETS_DIR / "decider_dev.jsonl", allow_draft=True):
        transcript = normalize(
            (REPO_ROOT / f"tests/datasets/{fragment.meeting}_transcript.txt").read_text()
        )
        assert normalize(fragment.quote) in transcript, fragment.id


def test_reviewed_fragment_with_empty_labels_is_rejected(tmp_path):
    row = {"id": "x", "meeting": "m", "status": "reviewed", "speaker": "A", "quote": "hola"}
    path = tmp_path / "f.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="empty labels"):
        load_fragment_labels(path)


# --- evaluators, run through the real SDK ------------------------------------------------------


def _scores(evaluator, expected, output):
    @eval_task()
    def task(case):
        return {"output": output}

    case = Case(name="c", input={}, expected_output=expected)
    report = Experiment(cases=[case], evaluators=[evaluator]).run_evaluations(task)
    return report.detailed_results[0][0]


def test_meeting_evaluators_score_one_wrong_task():
    expected = {
        "tasks": [_gold("cerrar la integración de autenticación", "Luis", "2026-10-23")],
        "not_task_quotes": ["almorzar juntos mañana"],
    }
    output = _output(
        [
            _pred("cerrar la integración de autenticación", "Ana", "2026-10-24"),
            _pred("almorzar juntos mañana", flagged=False),
        ]
    )
    f1 = _scores(TaskExtractionEvaluator(), expected, output)
    assert f1.score == pytest.approx(2 / 3) and not f1.test_pass  # 1 found, 1 invented, 0 missed
    assert _scores(AssigneeEvaluator(), expected, output).score == 0.0
    assert _scores(DueDateEvaluator(), expected, output).score == 0.0
    assert _scores(TrapEvaluator(), expected, output).score == 0.0
    flags = _scores(ReviewFlagEvaluator(), expected, output)
    assert flags.score == 0.0 and not flags.test_pass  # two wrong tasks, none flagged


def test_flagging_the_wrong_tasks_passes_the_review_evaluator():
    expected = {"tasks": [], "not_task_quotes": []}
    output = _output([_pred("algo inventado por el modelo", flagged=True)])
    assert _scores(ReviewFlagEvaluator(), expected, output).test_pass


def test_empty_meeting_with_no_predictions_is_a_success():
    expected = {"tasks": [], "not_task_quotes": []}
    assert _scores(TaskExtractionEvaluator(), expected, _output([])).score == 1.0


def test_decider_type_evaluator():
    expected = {"statement_type": "task", "note": None}
    ok = {"error": None, "statement_type": "task", "confidence": 0.9, "abstained": False}
    assert _scores(DeciderTypeEvaluator(), expected, ok).test_pass
    failed = {**ok, "error": "decider unreachable"}
    assert _scores(DeciderTypeEvaluator(), expected, failed).score == 0.0


# --- corpus aggregation -------------------------------------------------------------------------


def test_aggregate_pipeline_counts_by_hand():
    expected = {
        "tasks": [_gold("quote one here", "Luis", "2026-10-23"), _gold("quote two there", "Ana")],
        "not_task_quotes": ["small talk about coffee"],
    }
    output = _output(
        [
            _pred("quote one here", "Luis", "2026-10-23"),  # right
            _pred("quote two there", "Marta", flagged=True),  # wrong assignee, flagged
            _pred("small talk about coffee", flagged=False, unverified=1),  # invented, silent
        ],
        calls=[
            {"abstained": True, "error": None, "latency_ms": 100.0},
            {"abstained": False, "error": None, "latency_ms": 300.0},
            None,
        ],
    )
    result = aggregate_pipeline([(expected, output)])
    assert result["tasks"]["tp"] == 2 and result["tasks"]["fp"] == 1 and result["tasks"]["fn"] == 0
    assert (
        result["fields"]["assignee_accuracy"] == 0.5
        and result["fields"]["due_date_accuracy"] == 1.0
    )
    assert result["traps"] == {"hits": 1, "total": 1}
    review = result["review"]
    assert (review["wrong"], review["caught"], review["silent_errors"]) == (2, 1, 1)
    assert review["false_alarms"] == 0 and review["flag_recall"] == 0.5
    assert result["evidence"]["unsupported_rate"] == pytest.approx(1 / 3)
    assert result["decider"]["calls"] == 2 and result["decider"]["abstention_rate"] == 0.5
    assert result["decider"]["latency_ms_p50"] == 200.0


def _call(stype, confidence, commit=0.9, enough=0.9, error=None, abstained=False):
    return {
        "statement_type": stype,
        "confidence": confidence,
        "explicit_commitment": commit,
        "evidence_sufficient": enough,
        "latency_ms": 100.0,
        "abstained": abstained,
        "error": error,
    }


def test_aggregate_decider_ignores_failed_calls():
    label = {"statement_type": "task", "explicit_commitment": True, "evidence_sufficient": True}
    other = {"statement_type": "other", "explicit_commitment": False, "evidence_sufficient": False}
    rows = [
        (label, _call("task", 0.9)),
        (label, _call("other", 0.5, commit=0.2, enough=0.3)),  # wrong type, wrong answers
        (other, _call("other", 0.8, commit=0.1, enough=0.1)),
        (label, _call(None, None, error="unreachable", abstained=True)),
    ]
    result = aggregate_decider(rows)
    assert result["errors"] == 1 and result["fragments"] == 4
    assert result["statement_type"]["accuracy"] == pytest.approx(2 / 3)
    assert result["explicit_commitment"]["accuracy"] == pytest.approx(2 / 3)
    assert result["statement_type"]["confusion_matrix"]["task"]["other"] == 1
    assert result["statement_type"]["coverage_accuracy"][-1]["threshold"] == 0.9


# --- whole runs with a fake model and a fake Decider ---------------------------------------------


class ScriptedAgent:
    """Extracts exactly the dev labels of whichever meeting the prompt is about, with the
    segment ids read back from the prompt, so the run exercises the real validation."""

    def __call__(self, prompt, structured_output_model=None):
        labels = {
            m.title: m
            for m in load_meeting_labels(DATASETS_DIR / "meetings_dev.jsonl", allow_draft=True)
        }
        title = re.search(r"Meeting title: (.+)", prompt).group(1)
        segments = re.findall(r"\[(s\d+)\] .*?(?: @\d\d:\d\d:\d\d)?: (.*)", prompt)
        tasks = []
        for number, gold in enumerate(labels[title].tasks, 1):
            hit = next(
                i
                for i, (_, text) in enumerate(segments)
                if normalize(gold.evidence_quote) in normalize(text)
            )
            # Like the real model, also cite the reply that follows a request, so a person who is
            # only addressed ("I need you to...") is backed by the segment where they answer.
            ids = [sid for sid, _ in segments[hit : hit + 2]]
            tasks.append(
                Task(
                    title=f"task {number}",
                    assignee=gold.assignee,
                    assignee_status=FieldStatus.EXPLICIT,
                    evidence=[Evidence(quote=gold.evidence_quote, segment_ids=ids)],
                )
            )
        return SimpleNamespace(structured_output=ExtractionResult(tasks=tasks), metrics=None)


def _decider(task_type="task", confidence=0.95):
    body = {
        "model": "decider-test",
        "latency_ms": 5.0,
        "answers": {
            "statement_type": {
                "type": "choice",
                "choice": task_type,
                "probabilities": {},
                "confidence": confidence,
            },
            "explicit_commitment": {"type": "noul", "noul": 0.9},
            "evidence_sufficient": {"type": "noul", "noul": 0.9},
        },
    }
    return DeciderClient(
        "http://decider.test",
        0.6,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)),
    )


def test_pipeline_run_writes_reports_and_compares_modes(tmp_path):
    settings = Settings(_env_file=None, aws_bedrock_model_id="test-model")
    path = DATASETS_DIR / "meetings_dev.jsonl"
    labels = load_meeting_labels(path, allow_draft=True)
    runner = MeetingRunner(ScriptedAgent, _decider(task_type="suggestion"))

    result = run_pipeline_eval(labels, runner, settings, ["baseline", "hybrid"], tmp_path, [path])

    out = result["out_dir"]
    assert {p.name for p in out.iterdir()} >= {
        "baseline.report.json",
        "hybrid.report.json",
        "metrics.json",
        "predictions.json",
        "summary.md",
    }
    baseline, hybrid = result["metrics"]["baseline"], result["metrics"]["hybrid"]
    assert baseline["tasks"]["f1"] == 1.0 and hybrid["tasks"]["f1"] == 1.0  # same extraction
    # The scripted agent finds every task but leaves the due dates empty, so all 7 are wrong.
    assert baseline["review"]["wrong"] == 7 and baseline["review"]["flagged"] == 0
    assert baseline["review"]["silent_errors"] == 7  # nothing flags them without Decider
    # This fake Decider doubts every task, which here happens to catch all the errors.
    assert hybrid["review"]["flagged"] == hybrid["review"]["predicted_tasks"] == 7
    assert hybrid["review"]["caught"] == 7 and hybrid["review"]["silent_errors"] == 0
    assert hybrid["review"]["false_alarms"] == 0
    assert baseline["decider"]["calls"] == 0 and hybrid["decider"]["calls"] == 7
    metadata = json.loads((out / "metrics.json").read_text())["metadata"]
    assert (
        metadata["bedrock_model_id"] == "test-model"
        and "meetings_dev.jsonl" in metadata["datasets"]
    )
    assert "Silent errors" in (out / "summary.md").read_text()


def test_each_meeting_is_extracted_only_once_for_both_modes(tmp_path):
    calls = []

    class Counting(ScriptedAgent):
        def __call__(self, prompt, structured_output_model=None):
            calls.append(1)
            return super().__call__(prompt, structured_output_model)

    settings = Settings(_env_file=None)
    path = DATASETS_DIR / "meetings_dev.jsonl"
    labels = load_meeting_labels(path, allow_draft=True)
    runner = MeetingRunner(Counting, _decider())
    run_pipeline_eval(labels, runner, settings, ["baseline", "hybrid"], tmp_path, [path])
    assert len(calls) == len(labels)


def test_decider_run_writes_reports(tmp_path):
    settings = Settings(_env_file=None)
    path = DATASETS_DIR / "decider_dev.jsonl"
    labels = load_fragment_labels(path, allow_draft=True)
    result = run_decider_eval(labels, _decider(), settings, tmp_path, [path])
    assert result["metrics"]["fragments"] == len(labels) and result["metrics"]["errors"] == 0
    summary = (result["out_dir"] / "summary.md").read_text()
    assert "Confusion matrix" in summary and "Choosing the confidence threshold" in summary
