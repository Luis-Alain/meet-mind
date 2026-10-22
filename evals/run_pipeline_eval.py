"""Evaluate the extraction pipeline on labelled meetings, with and without Decider.

    uv run python -m evals.run_pipeline_eval --split dev

Needs AWS credentials (the model extracts) and, for the "hybrid" mode, a running Decider server.
Each meeting is extracted once and read twice, so the comparison isolates what Decider changes.
"""

import argparse
from pathlib import Path

from strands_evals import Experiment

from evals.aggregate import aggregate_pipeline
from evals.evaluators import (
    AssigneeEvaluator,
    DueDateEvaluator,
    ReviewFlagEvaluator,
    TaskExtractionEvaluator,
    TrapEvaluator,
)
from evals.pipeline import MODE_BASELINE, MODE_HYBRID, MeetingRunner, build_meeting_cases
from evals.reporting import new_run_dir, num, pct, run_metadata, save_json
from evals.schema import DATASETS_DIR, REPO_ROOT, MeetingLabels, load_meeting_labels
from meet_mind.config import Settings, get_settings


def _evaluators() -> list:
    return [
        TaskExtractionEvaluator(),
        AssigneeEvaluator(),
        DueDateEvaluator(),
        TrapEvaluator(),
        ReviewFlagEvaluator(),
    ]


def render_summary(metadata: dict, metrics: dict[str, dict], per_case: dict[str, dict]) -> str:
    """A Markdown report: a side-by-side table of the modes, then the detail per meeting."""
    modes = list(metrics)
    lines = ["# Pipeline evaluation", ""]
    lines += [f"- **{key}:** {value}" for key, value in metadata.items() if key != "packages"]
    lines += ["", "## Results", "", "| Metric | " + " | ".join(modes) + " |"]
    lines += ["|---|" + "---|" * len(modes)]

    def row(label, getter, fmt=pct):
        lines.append(f"| {label} | " + " | ".join(fmt(getter(metrics[m])) for m in modes) + " |")

    row("Task precision", lambda m: m["tasks"]["precision"])
    row("Task recall", lambda m: m["tasks"]["recall"])
    row("Task F1", lambda m: m["tasks"]["f1"])
    row("Assignee accuracy (found tasks)", lambda m: m["fields"]["assignee_accuracy"])
    row("Due-date accuracy (found tasks)", lambda m: m["fields"]["due_date_accuracy"])
    row("Traps fallen for", lambda m: f"{m['traps']['hits']}/{m['traps']['total']}", str)
    row("Quotes not found in transcript", lambda m: m["evidence"]["unsupported_rate"])
    lines.append("| *Review flags* | " + " | ".join("" for _ in modes) + " |")
    row(
        "Tasks flagged / predicted",
        lambda m: f"{m['review']['flagged']}/{m['review']['predicted_tasks']}",
        str,
    )
    row("Wrong tasks", lambda m: m["review"]["wrong"], str)
    row("Wrong tasks that were flagged", lambda m: m["review"]["caught"], str)
    row("Silent errors (wrong, not flagged)", lambda m: m["review"]["silent_errors"], str)
    row("False alarms (right, flagged)", lambda m: m["review"]["false_alarms"], str)
    row("Decider abstention rate", lambda m: m["decider"]["abstention_rate"])
    row(
        "Decider latency p50 / p95 (ms)",
        lambda m: f"{num(m['decider']['latency_ms_p50'])} / {num(m['decider']['latency_ms_p95'])}",
        str,
    )
    row(
        "Tokens in / out",
        lambda m: f"{m['cost']['input_tokens']} / {m['cost']['output_tokens']}",
        str,
    )

    lines += [
        "",
        "## Per meeting (task F1)",
        "",
        "| Meeting | " + " | ".join(modes) + " | Missed | Invented |",
    ]
    lines += ["|---|" + "---|" * (len(modes) + 2)]
    for name, info in per_case.items():
        lines.append(
            f"| {name} | "
            + " | ".join(pct(info["f1"][m]) for m in modes)
            + f" | {'; '.join(info['missed']) or '-'} | {'; '.join(info['invented']) or '-'} |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "- Counts matter more than percentages when there are only a handful of meetings: one task "
        "changes a percentage a lot.",
        "- *Silent errors* are the costly ones: wrong tasks that reached the user without a flag.",
        "- Baseline and hybrid use the same extraction, so any difference between them comes "
        "from Decider alone.",
    ]
    return "\n".join(lines) + "\n"


def run_pipeline_eval(
    labels: list[MeetingLabels],
    runner: MeetingRunner,
    settings: Settings,
    modes: list[str],
    out_root: Path,
    dataset_paths: list[Path],
) -> dict:
    """Run every mode over the labelled meetings and write the report files. Returns the metrics."""
    cases = build_meeting_cases(labels)
    out_dir = new_run_dir(out_root, "pipeline")
    expected = {case.name: case.expected_output for case in cases}

    metrics: dict[str, dict] = {}
    per_case: dict[str, dict] = {}
    for mode in modes:
        report = Experiment(cases=cases, evaluators=_evaluators()).run_evaluations(
            runner.task_for(mode)
        )
        report.to_file(str(out_dir / f"{mode}.report.json"))
        outputs = runner.outputs[mode]
        metrics[mode] = aggregate_pipeline([(expected[name], outputs[name]) for name in expected])

        # Per-meeting detail comes straight from the SDK's evaluator results.
        for row, details in zip(report.cases, report.detailed_results, strict=True):
            if row["evaluator"] != "TaskExtractionEvaluator":
                continue
            info = per_case.setdefault(row["name"], {"f1": {}, "missed": [], "invented": []})
            info["f1"][mode] = details[0].score
            if mode == modes[0]:
                info["missed"] = details[0].metadata["missed"]
                info["invented"] = details[0].metadata["invented"]

    metadata = run_metadata(settings, dataset_paths, modes=modes, meetings=len(cases))
    save_json(out_dir / "metrics.json", {"metadata": metadata, "metrics": metrics})
    save_json(out_dir / "predictions.json", runner.outputs)
    (out_dir / "summary.md").write_text(render_summary(metadata, metrics, per_case))
    return {"out_dir": out_dir, "metrics": metrics, "per_case": per_case}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument(
        "--allow-draft", action="store_true", help="also use labels not yet reviewed"
    )
    parser.add_argument("--modes", nargs="+", default=[MODE_BASELINE, MODE_HYBRID])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "evals" / "reports")
    args = parser.parse_args()

    from meet_mind.agent.factory import make_agent_factory
    from meet_mind.services.decider import DeciderClient

    settings = get_settings()
    path = DATASETS_DIR / f"meetings_{args.split}.jsonl"
    labels = load_meeting_labels(path, allow_draft=args.allow_draft)

    decider = None
    if MODE_HYBRID in args.modes:
        decider = DeciderClient(
            settings.decider_base_url,
            settings.decider_confidence_threshold,
            timeout=settings.decider_timeout_seconds,
        )
        if not decider.is_available():
            raise SystemExit(f"Decider is not reachable at {settings.decider_base_url}")

    if args.split == "test":
        print("TEST split: report these numbers, but do not tune prompts or thresholds on them.")
    runner = MeetingRunner(make_agent_factory(settings), decider)
    result = run_pipeline_eval(labels, runner, settings, args.modes, args.out, [path])
    print((result["out_dir"] / "summary.md").read_text())
    print(f"Files written to {result['out_dir']}")


if __name__ == "__main__":
    main()
