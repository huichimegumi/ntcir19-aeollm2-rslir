"""ODT and OSODT utilities for AEOLLM-2."""

from .aggregate import aggregate_osodt
from .constants import DIMENSIONS, PROMPT_REVISION
from .document import DocumentRecord, read_docx_with_tables
from .rubric import Criterion, Rubric, load_rubric
from .scoring import weighted_score

__all__ = [
    "Criterion",
    "DIMENSIONS",
    "DocumentRecord",
    "PROMPT_REVISION",
    "Rubric",
    "aggregate_osodt",
    "load_rubric",
    "read_docx_with_tables",
    "weighted_score",
]
