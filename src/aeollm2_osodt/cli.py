"""Command line entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .aggregate import aggregate_osodt
from .artifacts import atomic_write_json, sha256_file
from .config import load_judge_config
from .constants import SCHEMA_VERSION
from .metrics import evaluate_predictions, metrics_by_scope
from .submission import read_predictions, write_submission
from .pipeline import run_pipeline
from .rubric import load_rubric


def _run_command(args: argparse.Namespace) -> None:
    config = load_judge_config(args.config)
    run_pipeline(
        method=args.method,
        config=config,
        report_root=args.report_root,
        rubric_dir=args.rubric_dir,
        prompt_scales=args.prompt_scales,
        output_dir=args.output_dir,
        odt_order=args.order,
        preload_responses_from=args.preload_responses_from,
    )


def _aggregate_command(args: argparse.Namespace) -> None:
    paths = aggregate_osodt(args.original, args.reverse, args.output_dir)
    atomic_write_json(
        args.output_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "method": "offline-osodt-aggregation",
            "inputs": {
                "original": {
                    "path": str(args.original),
                    "sha256": sha256_file(args.original),
                },
                "reverse": {
                    "path": str(args.reverse),
                    "sha256": sha256_file(args.reverse),
                },
            },
        },
    )
    print(f"OSODT predictions: {paths[0]}")
    print(f"Symmetric criterion scores: {paths[1]}")
    print(f"Order-effect diagnostics: {paths[2]}")
    print(f"Aggregation manifest: {args.output_dir / 'manifest.json'}")


def _evaluate_command(args: argparse.Namespace) -> None:
    predictions = read_predictions(args.predictions)
    labels = pd.read_csv(args.labels)
    question_ids = sorted(int(value) for value in labels["questionId"].unique())
    weights = {
        question_id: load_rubric(args.rubric_dir, question_id).dimension_weights
        for question_id in question_ids
    }
    summary, details = evaluate_predictions(predictions, labels, weights)
    scoped = metrics_by_scope(summary)
    print(scoped.to_string(index=False))
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_json(output_dir / "summary.json", orient="records", indent=2)
        details.to_csv(output_dir / "per_question.csv", index=False)
        scoped.to_csv(output_dir / "metrics_by_scope.csv", index=False)


def _export_command(args: argparse.Namespace) -> None:
    write_submission(read_predictions(args.predictions), args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aeollm2-osodt")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run Joint, ODT, or OSODT inference")
    run.add_argument("--method", choices=["joint", "odt", "osodt"], required=True)
    run.add_argument("--order", choices=["original", "reverse"], default="original")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--report-root", type=Path, required=True)
    run.add_argument("--rubric-dir", type=Path, required=True)
    run.add_argument("--prompt-scales", type=Path, default=Path("prompts/prompt_scales.json"))
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument(
        "--preload-responses-from", type=Path,
        help="Reuse verified matching raw responses from a prior run's JSONL cache",
    )
    run.set_defaults(func=_run_command)

    aggregate = subparsers.add_parser(
        "aggregate-osodt",
        help="Build OSODT artifacts offline from original/reverse criterion JSONL files",
    )
    aggregate.add_argument("--original", type=Path, required=True)
    aggregate.add_argument("--reverse", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.set_defaults(func=_aggregate_command)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate predictions against human labels")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--labels", type=Path, required=True)
    evaluate.add_argument("--rubric-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, default=None)
    evaluate.set_defaults(func=_evaluate_command)

    export = subparsers.add_parser("export-submission", help="Export a headerless seven-column task file")
    export.add_argument("--predictions", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.set_defaults(func=_export_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
