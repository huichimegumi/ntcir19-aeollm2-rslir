"""Faithful DOCX reading with paragraphs and tables preserved in document order."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from docx import Document as WordDocument
from docx.table import Table
from docx.text.paragraph import Paragraph

TABLE_START = "[TABLE_START]"
TABLE_END = "[TABLE_END]"


@dataclass(frozen=True)
class DocumentRecord:
    question_id: int
    answer_id: str
    path: Path
    text: str
    paragraph_count: int
    table_count: int
    table_row_count: int


def _normalize_table_cell_text(text: str) -> str:
    """Flatten one table cell while preserving paragraph boundaries."""
    parts = [
        part.strip()
        for part in str(text).replace("\r", "\n").split("\n")
        if part.strip()
    ]
    return " <br> ".join(parts).replace("\t", " ")


def _serialize_table(table: Table) -> tuple[str, int]:
    """Serialize a Word table as compact TSV with explicit boundary markers.

    python-docx repeats the same underlying XML cell for merged cells. The
    seen_cells set keeps the corresponding column placeholder while avoiding
    duplicated cell text.
    """
    lines = [TABLE_START]
    seen_cells: set[object] = set()
    row_count = 0

    for row in table.rows:
        row_count += 1
        cells: list[str] = []
        for cell in row.cells:
            cell_key = cell._tc
            if cell_key in seen_cells:
                cells.append("")
                continue
            seen_cells.add(cell_key)
            cells.append(_normalize_table_cell_text(cell.text))
        lines.append("\t".join(cells).rstrip())

    lines.append(TABLE_END)
    return "\n".join(lines), row_count


def read_docx_with_tables(
    path: str | Path,
    *,
    question_id: int | None = None,
    answer_id: str | None = None,
) -> DocumentRecord:
    """Read top-level paragraphs and real Word tables in their original order."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if question_id is None:
        raise ValueError("question_id is required when building a DocumentRecord")

    document = WordDocument(str(path))
    blocks: list[str] = []
    paragraph_count = 0
    table_count = 0
    table_row_count = 0

    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            paragraph_count += 1
            # Preserve empty paragraphs and newline behavior from the controlled
            # paper reader; tables are the only intentional content addition.
            blocks.append(block.text)
        elif isinstance(block, Table):
            table_count += 1
            serialized, rows = _serialize_table(block)
            table_row_count += rows
            blocks.append(serialized)

    text = "\n".join(blocks)
    if not text.strip():
        raise ValueError(f"DOCX contains no readable text or tables: {path}")

    return DocumentRecord(
        question_id=int(question_id),
        answer_id=str(answer_id or path.stem),
        path=path,
        text=text,
        paragraph_count=paragraph_count,
        table_count=table_count,
        table_row_count=table_row_count,
    )
