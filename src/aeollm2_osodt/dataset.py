"""Split-agnostic AEOLLM-2 document discovery."""

from __future__ import annotations

import re
from pathlib import Path

from .document import DocumentRecord, read_docx_with_tables

_REPORT_RE = re.compile(r"report(\d+)$", re.I)


def discover_documents(report_root: str | Path) -> list[DocumentRecord]:
    """Discover ReportN/*.docx files without hard-coding train/test/reserve splits."""
    root = Path(report_root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    records: list[DocumentRecord] = []
    for report_dir in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not report_dir.is_dir():
            continue
        match = _REPORT_RE.fullmatch(report_dir.name)
        if not match:
            continue
        question_id = int(match.group(1))
        for path in sorted(report_dir.glob("*.docx"), key=lambda p: p.stem):
            records.append(read_docx_with_tables(path, question_id=question_id))

    if not records:
        raise ValueError(f"no ReportN/*.docx documents found under {root}")

    keys = [(record.question_id, record.answer_id) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate (question_id, answer_id) document keys")
    return records
