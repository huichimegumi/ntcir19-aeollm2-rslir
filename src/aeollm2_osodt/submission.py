"""Read internal TSVs and AEOLLM-2 seven-column, whitespace-separated task files."""

from pathlib import Path

import pandas as pd

from .constants import DIMENSIONS
from .metrics import _normalize_score_columns, _validate_frame

COLUMNS = ["taskId", "questionId", "answerId", *[f"{d}_score" for d in DIMENSIONS]]


def read_predictions(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    with path.open(encoding="utf-8-sig") as handle:
        first = handle.readline().split()
    if not first:
        raise ValueError("prediction file is empty")
    header = first == COLUMNS
    frame = pd.read_csv(path, sep=r"\s+", header=0 if header else None,
                        names=None if header else COLUMNS, encoding="utf-8-sig",
                        float_precision="round_trip")
    if list(frame.columns) != COLUMNS or frame.index.nlevels != 1 or not isinstance(frame.index, pd.RangeIndex):
        raise ValueError("predictions require exactly seven columns")
    checked = _normalize_score_columns(frame.copy())
    _validate_frame("predictions", checked)
    if frame["answerId"].str.contains(r"\s").any():
        raise ValueError("answerId must not contain whitespace")
    return frame


def write_submission(frame: pd.DataFrame, path: str | Path) -> None:
    checked = _normalize_score_columns(frame.copy())
    if "taskId" not in checked:
        checked["taskId"] = 0
    _validate_frame("predictions", checked)
    if checked["answerId"].str.contains(r"\s").any():
        raise ValueError("answerId must not contain whitespace")
    exported = checked.rename(columns={d: f"{d}_score" for d in DIMENSIONS})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exported.sort_values(["questionId", "answerId"])[COLUMNS].to_csv(path, sep="\t", index=False, header=False)
