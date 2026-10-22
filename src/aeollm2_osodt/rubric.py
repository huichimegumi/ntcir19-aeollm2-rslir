"""Canonical AEOLLM-2 rubric representation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .constants import DIMENSIONS


@dataclass(frozen=True)
class Criterion:
    criterion_id: str
    original_index: int
    text: str
    explanation: str
    weight: float
    weight_display: str | None = None

    @property
    def prompt_weight(self) -> str:
        return self.weight_display if self.weight_display is not None else str(self.weight)


@dataclass(frozen=True)
class Rubric:
    question_id: int
    prompt: str
    dimension_weights: Mapping[str, float]
    criteria: Mapping[str, tuple[Criterion, ...]]


def _rubric_path(rubric_dir: Path, question_id: int) -> Path:
    candidates = (
        rubric_dir / f"criterion{question_id}.json",
        rubric_dir / f"criteria{question_id}.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Rubric not found for question {question_id}: {candidates}")


def load_rubric(rubric_dir: str | Path, question_id: int) -> Rubric:
    path = _rubric_path(Path(rubric_dir), int(question_id))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("id", question_id)) != int(question_id):
        raise ValueError(f"Rubric question ID mismatch in {path}")

    raw_criteria = payload.get("criterions")
    if not isinstance(raw_criteria, dict):
        raise ValueError(f"Rubric has no criterions object: {path}")

    criteria: dict[str, tuple[Criterion, ...]] = {}
    for dimension in DIMENSIONS:
        items = raw_criteria.get(dimension)
        if not isinstance(items, list) or not items:
            raise ValueError(f"Rubric {path} has no criteria for {dimension}")
        normalized: list[Criterion] = []
        for index, item in enumerate(items, 1):
            if not isinstance(item, dict):
                raise ValueError(f"Invalid criterion at {dimension}[{index}] in {path}")
            raw_weight = item.get("weight", 0.0)
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid criterion weight at {dimension}[{index}] in {path}: {raw_weight!r}"
                ) from exc
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"Invalid criterion weight at {dimension}[{index}] in {path}")
            normalized.append(
                Criterion(
                    criterion_id=f"{dimension}_{index}",
                    original_index=index,
                    text=str(item.get("criterion", "")),
                    explanation=str(item.get("explanation", "")),
                    weight=weight,
                    weight_display=str(raw_weight),
                )
            )
        if sum(item.weight for item in normalized) <= 0:
            raise ValueError(f"Rubric {path} has non-positive total criterion weight for {dimension}")
        criteria[dimension] = tuple(normalized)

    raw_dimension_weights = payload.get("dimension_weight")
    if not isinstance(raw_dimension_weights, dict):
        raise ValueError(f"Rubric has no dimension_weight object: {path}")
    missing_weights = [dimension for dimension in DIMENSIONS if dimension not in raw_dimension_weights]
    if missing_weights:
        raise ValueError(f"Rubric {path} is missing dimension weights: {missing_weights}")

    dimension_weights: dict[str, float] = {}
    for dimension in DIMENSIONS:
        raw_weight = raw_dimension_weights[dimension]
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid dimension weight for {dimension} in {path}: {raw_weight!r}"
            ) from exc
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid dimension weight for {dimension} in {path}: {weight}")
        dimension_weights[dimension] = weight
    if sum(dimension_weights.values()) <= 0:
        raise ValueError(f"Rubric {path} has non-positive total dimension weight")

    return Rubric(
        question_id=int(payload.get("id", question_id)),
        prompt=str(payload.get("prompt", "")),
        dimension_weights=dimension_weights,
        criteria=criteria,
    )
