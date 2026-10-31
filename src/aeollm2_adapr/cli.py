"""AdapR command-line entry point."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .pipeline import run_adapr


def _load_config(path: Path) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_answer_map(
    report_root: Path, question_ids: list[int]
) -> tuple[dict[int, str], dict[str, dict]]:
    """Scan report_root for .docx files and build question/answer maps.

    Expects report_root/{Report{qid}|Q{qid}}/*.docx or report_root/**/*.docx.
    Falls back to a flat scan if no subdirectory structure is found.
    """
    from datasets import load_dataset

    ds = load_dataset("THUIR/AEOLLM", "aeollm_2")
    question_map: dict[int, str] = {}
    answer_id_to_qid: dict[str, int] = {}

    for split_name in ["train", "test"]:
        for row in ds[split_name]:
            qid = int(row["questionId"])
            if qid in question_ids:
                question_map[qid] = str(row["question"])
                answer_id_to_qid[str(row["answerId"])] = qid

    answer_map: dict[str, dict] = {}
    for docx_path in sorted(report_root.rglob("*.docx")):
        stem = docx_path.stem
        if stem in answer_id_to_qid:
            qid = answer_id_to_qid[stem]
            answer_map[stem] = {
                "questionId": qid,
                "taskId": 0,
                "docx_path": docx_path,
            }

    return question_map, answer_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AdapR: Adaptive Criterion Auto-Rubric scoring for NTCIR-19 AEOLLM-2"
    )
    parser.add_argument("--config", required=True, help="YAML config file path")
    parser.add_argument(
        "--report-root",
        required=True,
        type=Path,
        help="Root directory containing .docx report files",
    )
    parser.add_argument(
        "--question-ids",
        nargs="+",
        type=int,
        help="Question IDs to score (default: all found under report-root)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output TSV file path",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache/adapr"),
        help="Root directory for caching LLM outputs (default: cache/adapr)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent document workers (default: 4)",
    )
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    model_cfg = cfg.get("model", {})
    gen_cfg = cfg.get("generation", {})
    extra_body = cfg.get("extra_body", {})

    api_key_env = model_cfg.get("api_key_env", "DEEPSEEK_API_KEY")
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise SystemExit(f"Environment variable {api_key_env!r} is not set.")

    report_root = Path(args.report_root)
    qids = args.question_ids or []

    if not qids:
        import re
        for d in sorted(report_root.iterdir()):
            m = re.search(r"(\d+)", d.name)
            if m and d.is_dir():
                qids.append(int(m.group(1)))

    question_map, answer_map = _build_answer_map(report_root, qids)

    if not answer_map:
        raise SystemExit(
            f"No matching .docx files found under {report_root} for question IDs {qids}."
        )

    print(
        f"[AdapR] {len(question_map)} questions, {len(answer_map)} documents, "
        f"model={model_cfg.get('id')}"
    )

    run_adapr(
        report_root=report_root,
        question_map=question_map,
        answer_map=answer_map,
        output_path=Path(args.output),
        api_key=api_key,
        api_base=model_cfg.get("base_url", "https://api.deepseek.com"),
        model=model_cfg.get("id", "deepseek-v4-pro"),
        extra_body=extra_body,
        cache_root=Path(args.cache_dir),
        workers=args.workers,
    )
