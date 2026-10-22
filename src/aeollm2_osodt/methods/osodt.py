"""Strict criterion-aligned averaging for OSODT."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .odt import CriterionScore, DimensionScore


@dataclass(frozen=True)
class SymmetricCriterionScore:
    criterion_id: str
    original_index: int
    criterion: str
    weight: float
    original_score: float
    reverse_score: float
    score: float


@dataclass(frozen=True)
class SymmetricDimensionScore:
    dimension: str
    criteria: tuple[SymmetricCriterionScore, ...]
    original_dim_score: float
    reverse_dim_score: float
    dim_score: float


def average_order_records(
    original: DimensionScore,
    reverse: DimensionScore,
) -> SymmetricDimensionScore:
    if original.dimension != reverse.dimension:
        raise ValueError("dimension mismatch between original and reverse records")
    if original.criterion_order != "original" or reverse.criterion_order != "reverse":
        raise ValueError("OSODT requires original and reverse ODT records")

    left = {item.criterion_id: item for item in original.criteria}
    right = {item.criterion_id: item for item in reverse.criteria}
    if set(left) != set(right):
        raise ValueError("criterion IDs differ between original and reverse records")

    averaged: list[SymmetricCriterionScore] = []
    for criterion_id in sorted(left, key=lambda key: left[key].original_index):
        a = left[criterion_id]
        b = right[criterion_id]
        if a.original_index != b.original_index:
            raise ValueError(f"{criterion_id}: original_index mismatch")
        if a.criterion != b.criterion:
            raise ValueError(f"{criterion_id}: criterion text mismatch")
        if not math.isclose(a.weight, b.weight, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{criterion_id}: criterion weight mismatch")
        mean_score = round((float(a.score) + float(b.score)) / 2.0, 4)
        averaged.append(
            SymmetricCriterionScore(
                criterion_id=criterion_id,
                original_index=a.original_index,
                criterion=a.criterion,
                weight=a.weight,
                original_score=float(a.score),
                reverse_score=float(b.score),
                score=mean_score,
            )
        )

    total_weight = sum(item.weight for item in averaged)
    if total_weight <= 0:
        symmetric = sum(item.score for item in averaged) / len(averaged)
    else:
        symmetric = sum(item.weight * item.score for item in averaged) / total_weight

    return SymmetricDimensionScore(
        dimension=original.dimension,
        criteria=tuple(averaged),
        original_dim_score=original.dim_score,
        reverse_dim_score=reverse.dim_score,
        dim_score=round(float(symmetric), 4),
    )
