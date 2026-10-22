"""ODT criterion identity mapping and dimension aggregation."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..prompting import criteria_in_prompt_order
from ..rubric import Rubric
from ..scoring import weighted_score


@dataclass(frozen=True)
class CriterionScore:
    criterion_id: str
    original_index: int
    prompt_position: int
    criterion: str
    weight: float
    score: float


@dataclass(frozen=True)
class DimensionScore:
    dimension: str
    criterion_order: str
    criteria: tuple[CriterionScore, ...]
    dim_score: float


def map_prompt_scores_to_canonical(
    rubric: Rubric,
    dimension: str,
    prompt_scores: list[float],
    *,
    criterion_order: str,
) -> DimensionScore:
    """Map positional model output back to stable original criterion identities."""
    prompt_criteria = criteria_in_prompt_order(rubric, dimension, criterion_order)
    if len(prompt_criteria) != len(prompt_scores):
        raise ValueError(
            f"{dimension}: expected {len(prompt_criteria)} prompt scores, got {len(prompt_scores)}"
        )

    by_id: dict[str, CriterionScore] = {}
    for prompt_position, (criterion, score) in enumerate(zip(prompt_criteria, prompt_scores), 1):
        if isinstance(score, bool) or not math.isfinite(float(score)) or not 0 <= float(score) <= 10:
            raise ValueError(f"{criterion.criterion_id}: score outside [0, 10]: {score!r}")
        normalized = CriterionScore(
            criterion_id=criterion.criterion_id,
            original_index=criterion.original_index,
            prompt_position=prompt_position,
            criterion=criterion.text,
            weight=float(criterion.weight),
            score=round(float(score), 4),
        )
        if normalized.criterion_id in by_id:
            raise ValueError(f"duplicate criterion ID: {normalized.criterion_id}")
        by_id[normalized.criterion_id] = normalized

    canonical = tuple(sorted(by_id.values(), key=lambda item: item.original_index))
    expected_ids = [f"{dimension}_{index}" for index in range(1, len(canonical) + 1)]
    actual_ids = [item.criterion_id for item in canonical]
    if actual_ids != expected_ids:
        raise ValueError(f"non-canonical IDs for {dimension}: {actual_ids}")

    dim_score = weighted_score(
        rubric.criteria[dimension],
        [item.score for item in canonical],
    )
    return DimensionScore(
        dimension=dimension,
        criterion_order=criterion_order,
        criteria=canonical,
        dim_score=dim_score,
    )
