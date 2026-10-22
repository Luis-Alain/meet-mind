"""Evaluate Decider on its own, one sentence at a time.

    uv run python -m evals.run_decider_eval --split dev

Needs a running Decider server. No language model is involved, so a run is free and quick.
"""

import argparse
from pathlib import Path

from strands_evals import Experiment

from evals.aggregate import STATEMENT_TYPES, aggregate_decider
from evals.evaluators import (
    DeciderCommitmentEvaluator,
    DeciderEvidenceEvaluator,
    DeciderTypeEvaluator,
)
from evals.pipeline import build_fragment_cases, fragment_task
from evals.reporting import new_run_dir, num, pct, run_metadata, save_json
from evals.schema import DATASETS_DIR, REPO_ROOT, FragmentLabels, load_fragment_labels
from meet_mind.config import Settings, get_settings
from meet_mind.services.decider import DeciderClient


def render_summary(metadata: dict, m: dict) -> str:
    t = m["statement_type"]
    lines = ["# Decider evaluation", ""]
    lines += [f"- **{key}:** {value}" for key, value in metadata.items() if key != "packages"]
    lines += [
        "",
        "## Results",
        "",
        f"- Fragments: {m['fragments']} "
        f"({m['errors']} failed calls, left out of the figures below)",
        f"- Statement type accuracy: {pct(t['accuracy'])} | macro F1: {pct(t['macro_f1'])} "
        f"| ECE: {num(t['ece'], 3)}",
        f"- Explicit commitment accuracy: {pct(m['explicit_commitment']['accuracy'])} "
        f"(Brier {num(m['explicit_commitment']['brier'], 3)})",
        f"- Evidence sufficient accuracy: {pct(m['evidence_sufficient']['accuracy'])} "
        f"(Brier {num(m['evidence_sufficient']['brier'], 3)})",
        f"- Abstention rate at the configured threshold: {pct(m['abstention_rate'])}",
        f"- Latency p50 / p95: {num(m['latency_ms_p50'])} / {num(m['latency_ms_p95'])} ms",
        "",
        "## Statement type per class",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for label, s in t["per_class"].items():
        cells = [pct(s["precision"]), pct(s["recall"]), pct(s["f1"]), str(s["support"])]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Confusion matrix (rows: true, columns: predicted)",
        "",
        "| | " + " | ".join(STATEMENT_TYPES) + " |",
    ]
    lines += ["|---|" + "---|" * len(STATEMENT_TYPES)]
    for true, row in t["confusion_matrix"].items():
        lines.append(f"| **{true}** | " + " | ".join(str(row[p]) for p in STATEMENT_TYPES) + " |")

    lines += [
        "",
        "## Choosing the confidence threshold",
        "",
        "If only answers at or above a confidence were trusted, what would be left?",
        "",
        "| Threshold | Answered | Coverage | Accuracy of those answers |",
        "|---|---|---|---|",
    ]
    for point in t["coverage_accuracy"]:
        cells = [f"{point['threshold']:.1f}", str(point["answered"])]
        cells += [pct(point["coverage"]), pct(point["accuracy"])]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def run_decider_eval(
    labels: list[FragmentLabels],
    decider: DeciderClient,
    settings: Settings,
    out_root: Path,
    dataset_paths: list[Path],
) -> dict:
    cases = build_fragment_cases(labels)
    out_dir = new_run_dir(out_root, "decider")
    outputs: dict[str, dict] = {}
    evaluators = [DeciderTypeEvaluator(), DeciderCommitmentEvaluator(), DeciderEvidenceEvaluator()]
    report = Experiment(cases=cases, evaluators=evaluators).run_evaluations(
        fragment_task(decider, outputs)
    )
    report.to_file(str(out_dir / "report.json"))

    metrics = aggregate_decider([(case.expected_output, outputs[case.name]) for case in cases])
    metadata = run_metadata(settings, dataset_paths, fragments=len(cases))
    save_json(out_dir / "metrics.json", {"metadata": metadata, "metrics": metrics})
    save_json(out_dir / "predictions.json", outputs)
    (out_dir / "summary.md").write_text(render_summary(metadata, metrics))
    return {"out_dir": out_dir, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument(
        "--allow-draft", action="store_true", help="also use labels not yet reviewed"
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "evals" / "reports")
    args = parser.parse_args()

    settings = get_settings()
    path = DATASETS_DIR / f"decider_{args.split}.jsonl"
    labels = load_fragment_labels(path, allow_draft=args.allow_draft)
    decider = DeciderClient(
        settings.decider_base_url,
        settings.decider_confidence_threshold,
        timeout=settings.decider_timeout_seconds,
    )
    if not decider.is_available():
        raise SystemExit(f"Decider is not reachable at {settings.decider_base_url}")

    if args.split == "test":
        print("TEST split: report these numbers, but do not tune prompts or thresholds on them.")
    result = run_decider_eval(labels, decider, settings, args.out, [path])
    print((result["out_dir"] / "summary.md").read_text())
    print(f"Files written to {result['out_dir']}")


if __name__ == "__main__":
    main()
