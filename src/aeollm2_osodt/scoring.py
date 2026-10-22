"""Pure scoring utilities shared by Joint, ODT, and OSODT."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .rubric import Criterion


def _validate_score(value: float, *, context: str) -> float:
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 10.0:
        raise ValueError(f"{context}: score outside [0, 10]: {value!r}")
    return score


def weighted_score(criteria: Sequence[Criterion], scores: Sequence[float]) -> float:
    """Aggregate criterion scores with official criterion weights."""
    if len(criteria) != len(scores):
        raise ValueError(
            f"criterion/score length mismatch: {len(criteria)} criteria vs {len(scores)} scores"
        )
    if not criteria:
        raise ValueError("cannot aggregate an empty criterion list")

    values = [
        _validate_score(score, context=criterion.criterion_id)
        for criterion, score in zip(criteria, scores)
    ]
    weights = [float(criterion.weight) for criterion in criteria]
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("criterion weights must be finite and non-negative")

    total_weight = sum(weights)
    if total_weight <= 0:
        result = sum(values) / len(values)
    else:
        result = sum(weight * score for weight, score in zip(weights, values)) / total_weight
    return round(float(result), 4)
