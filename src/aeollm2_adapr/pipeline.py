"""AdapR end-to-end scoring pipeline."""

from __future__ import annotations

import concurrent.futures
import csv
import os
from pathlib import Path

import docx

from openai import OpenAI

from .criterion_gen import AutoRCriterionGenerator, IFChecklistGenerator, QuestionTypeClassifier
from .scorer import AdapRScorer

DIMENSIONS = ("comprehensiveness", "insight", "instruction_following", "readability")


def _read_docx(path: Path) -> str:
    doc = docx.Document(str(path))
    parts: list[str] = []
    for block in doc.element.body:
        tag = block.tag.split("}")[-1]
        if tag == "p":
            import docx.oxml.ns as ns
            from docx.text.paragraph import Paragraph
            para = Paragraph(block, doc)
            text = para.text.strip()
            if text:
                parts.append(text)
        elif tag == "tbl":
            from docx.table import Table
            tbl = Table(block, doc)
            for row in tbl.rows:
                row_text = "\t".join(c.text.strip() for c in row.cells)
                if row_text.strip():
                    parts.append(row_text)
    return "\n".join(parts)


def run_adapr(
    *,
    report_root: Path,
    question_map: dict[int, str],
    answer_map: dict[str, dict],
    output_path: Path,
    api_key: str,
    api_base: str = "https://api.deepseek.com",
    model: str = "deepseek-v4-pro",
    extra_body: dict | None = None,
    cache_root: Path,
    workers: int = 4,
) -> None:
    """Run the full AdapR pipeline and write a TSV prediction file.

    Args:
        report_root: Directory containing per-question subdirectories of .docx files.
        question_map: {question_id: question_text}
        answer_map: {answer_id: {"questionId": int, "taskId": int, "docx_path": Path}}
        output_path: Where to write the TSV predictions.
        api_key: DeepSeek (or compatible) API key.
        api_base: API base URL.
        model: Model ID.
        extra_body: Extra request body fields (e.g. thinking disable).
        cache_root: Root directory for all caches.
        workers: Concurrent document workers.
    """
    client = OpenAI(api_key=api_key, base_url=api_base)

    criterion_gen = AutoRCriterionGenerator(
        client, model, cache_dir=cache_root / "criteria"
    )
    classifier = QuestionTypeClassifier(
        client, model, cache_dir=cache_root / "qtypes"
    )
    checklist_gen = IFChecklistGenerator(
        client, model, cache_dir=cache_root / "checklists"
    )
    scorer = AdapRScorer(
        client,
        model,
        cache_dir=cache_root / "scores",
        extra_body=extra_body or {},
    )

    # Step 0: preprocessing — generate per-question artifacts
    criteria_map: dict[int, dict] = {}
    qtype_map: dict[int, str] = {}
    checklist_map: dict[int, list] = {}

    for qid, question in sorted(question_map.items()):
        print(f"[Prep] Q{qid}: generating criteria ...")
        criteria_map[qid] = criterion_gen.generate(question, qid)
        qtype_map[qid] = classifier.classify(question, qid)
        if qtype_map[qid] == "A":
            print(f"[Prep] Q{qid}: Type A — generating IF checklist ...")
            checklist_map[qid] = checklist_gen.generate(question, qid)
        else:
            print(f"[Prep] Q{qid}: Type B — using AutoR criteria for IF")

    # Step 1-4: score each document
    rows: list[dict] = []
    lock = concurrent.futures.ThreadPoolExecutor._initializer  # noqa: used below

    def score_one(answer_id: str, meta: dict) -> dict | None:
        qid = meta["questionId"]
        docx_path = meta["docx_path"]
        try:
            doc_text = _read_docx(Path(docx_path))
        except Exception as e:
            print(f"[Error] {answer_id}: read failed — {e}")
            return None
        if_checklist = checklist_map.get(qid)
        try:
            scores = scorer.score(
                doc_text,
                question_map[qid],
                criteria_map[qid],
                answer_id,
                if_checklist=if_checklist,
            )
        except Exception as e:
            print(f"[Error] {answer_id}: scoring failed — {e}")
            return None
        return {
            "taskId": meta.get("taskId", 0),
            "questionId": qid,
            "answerID": answer_id,
            **{f"{d}_score": scores[d] for d in DIMENSIONS},
        }

    items = list(answer_map.items())
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(score_one, aid, meta): aid for aid, meta in items
        }
        for fut in concurrent.futures.as_completed(futures):
            aid = futures[fut]
            result = fut.result()
            if result:
                rows.append(result)
                print(
                    f"[Done] {aid} → "
                    + " ".join(
                        f"{d[:2].upper()}={result[f'{d}_score']:.2f}"
                        for d in DIMENSIONS
                    )
                )

    # Write output TSV
    rows.sort(key=lambda r: (r["questionId"], r["answerID"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["taskId", "questionId", "answerID"] + [f"{d}_score" for d in DIMENSIONS]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[AdapR] Done. {len(rows)} documents → {output_path}")
