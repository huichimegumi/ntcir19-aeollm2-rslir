"""Offline OSODT aggregation from canonical original/reverse criterion JSONL files."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import atomic_write_json
from .constants import DIMENSIONS, SCHEMA_VERSION
from .methods.odt import CriterionScore, DimensionScore
from .methods.osodt import SymmetricDimensionScore, average_order_records


def _record_key(record: dict[str, Any]) -> tuple[int, str]:
    try:
        return int(record["questionId"]), str(record["answerId"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("criterion record must contain questionId and answerId") from exc


def _finite_score(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context}: boolean is not a score")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: non-numeric score {value!r}") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 10.0:
        raise ValueError(f"{context}: score outside [0, 10]: {score}")
    return score


def _finite_weight(value: Any, context: str) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: non-numeric weight {value!r}") from exc
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(f"{context}: invalid weight {weight}")
    return weight


def _parse_dimension(
    payload: Any,
    *,
    dimension: str,
    expected_order: str,
    context: str,
) -> DimensionScore:
    if not isinstance(payload, dict):
        raise ValueError(f"{context}: dimension payload must be an object")
    dimension_order = payload.get("criterionOrder", expected_order)
    if dimension_order != expected_order:
        raise ValueError(
            f"{context}: criterionOrder must be {expected_order!r}, got {dimension_order!r}"
        )
    raw_criteria = payload.get("criteria")
    if not isinstance(raw_criteria, list) or not raw_criteria:
        raise ValueError(f"{context}: criteria must be a non-empty list")

    criteria: list[CriterionScore] = []
    seen_ids: set[str] = set()
    for position, item in enumerate(raw_criteria, 1):
        if not isinstance(item, dict):
            raise ValueError(f"{context}: criterion {position} must be an object")
        criterion_id = str(item.get("criterionId", ""))
        if not criterion_id or criterion_id in seen_ids:
            raise ValueError(f"{context}: duplicate or missing criterionId {criterion_id!r}")
        seen_ids.add(criterion_id)
        criterion_context = f"{context}/{criterion_id}"
        try:
            original_index = int(item["originalIndex"])
            prompt_position = int(item["promptPosition"])
            for field in ("originalIndex", "promptPosition"):
                if isinstance(item[field], bool) or float(item[field]) != int(item[field]):
                    raise ValueError("non-integral criterion index")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{criterion_context}: originalIndex and promptPosition must be integers"
            ) from exc
        criteria.append(
            CriterionScore(
                criterion_id=criterion_id,
                original_index=original_index,
                prompt_position=prompt_position,
                criterion=str(item.get("criterion", "")),
                weight=_finite_weight(item.get("weight"), criterion_context),
                score=_finite_score(item.get("score"), criterion_context),
            )
        )

    criteria.sort(key=lambda item: item.original_index)
    expected_ids = [f"{dimension}_{index}" for index in range(1, len(criteria) + 1)]
    actual_ids = [item.criterion_id for item in criteria]
    if actual_ids != expected_ids:
        raise ValueError(f"{context}: expected canonical IDs {expected_ids}, got {actual_ids}")
    expected_indices = list(range(1, len(criteria) + 1))
    if [item.original_index for item in criteria] != expected_indices:
        raise ValueError(f"{context}: originalIndex values are not canonical")

    prompt_positions = [item.prompt_position for item in criteria]
    expected_positions = (
        expected_indices if expected_order == "original" else list(reversed(expected_indices))
    )
    if prompt_positions != expected_positions:
        raise ValueError(
            f"{context}: promptPosition values do not match {expected_order} presentation order"
        )

    total_weight = sum(item.weight for item in criteria)
    dim_score = (
        sum(item.score for item in criteria) / len(criteria)
        if total_weight <= 0
        else sum(item.weight * item.score for item in criteria) / total_weight
    )
    calculated = round(float(dim_score), 4)
    if "dim_score" in payload:
        historical = _finite_score(payload["dim_score"], f"{context}/dim_score")
        if not math.isclose(calculated, historical, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"{context}: stored dim_score {historical} does not match reaggregation {calculated}"
            )

    return DimensionScore(
        dimension=dimension,
        criterion_order=expected_order,
        criteria=tuple(criteria),
        dim_score=calculated,
    )


def load_order_records(
    path: str | Path,
    *,
    expected_order: str,
) -> dict[tuple[int, str], dict[str, Any]]:
    """Load and strictly validate one canonical original/reverse criterion JSONL file."""
    if expected_order not in {"original", "reverse"}:
        raise ValueError("expected_order must be original or reverse")
    path = Path(path)
    records: dict[tuple[int, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: record must be an object")
            if record.get("criterionOrder") != expected_order:
                raise ValueError(
                    f"{path}:{line_no}: criterionOrder must be {expected_order!r}"
                )
            key = _record_key(record)
            if key in records:
                raise ValueError(f"{path}:{line_no}: duplicate document key {key}")
            dimensions = record.get("dimensions")
            if not isinstance(dimensions, dict) or set(dimensions) != set(DIMENSIONS):
                raise ValueError(f"{path}:{line_no}: dimensions must be exactly {list(DIMENSIONS)}")
            parsed_dimensions = {
                dimension: _parse_dimension(
                    dimensions[dimension],
                    dimension=dimension,
                    expected_order=expected_order,
                    context=f"{path.name}:{line_no}/{key}/{dimension}",
                )
                for dimension in DIMENSIONS
            }
            records[key] = {
                "judgeModel": record.get("judgeModel"),
                "dimensions": parsed_dimensions,
            }
    if not records:
        raise ValueError(f"no criterion records found in {path}")
    return records


def _criterion_payload(item) -> dict[str, Any]:
    return {
        "criterionId": item.criterion_id,
        "originalIndex": item.original_index,
        "criterion": item.criterion,
        "weight": item.weight,
        "originalScore": item.original_score,
        "reverseScore": item.reverse_score,
        "score": item.score,
    }


def _dimension_payload(score: SymmetricDimensionScore) -> dict[str, Any]:
    return {
        "criterionOrder": "symmetric",
        "criteria": [_criterion_payload(item) for item in score.criteria],
        "original_dim_score": score.original_dim_score,
        "reverse_dim_score": score.reverse_dim_score,
        "dim_score": score.dim_score,
    }


def aggregate_osodt(
    original_path: str | Path,
    reverse_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Average matched original/reverse criterion records and write OSODT artifacts."""
    original = load_order_records(original_path, expected_order="original")
    reverse = load_order_records(reverse_path, expected_order="reverse")
    if set(original) != set(reverse):
        only_original = sorted(set(original) - set(reverse))
        only_reverse = sorted(set(reverse) - set(original))
        raise ValueError(
            "document sets differ; refusing partial OSODT aggregation: "
            f"only_original={only_original[:10]}, only_reverse={only_reverse[:10]}"
        )

    output_dir = Path(output_dir)
    symmetric_records: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    criterion_deltas: list[float] = []
    dimension_deltas: list[float] = []

    for question_id, answer_id in sorted(original):
        left = original[(question_id, answer_id)]
        right = reverse[(question_id, answer_id)]
        if left["judgeModel"] != right["judgeModel"]:
            raise ValueError(
                f"{(question_id, answer_id)}: judge model mismatch: "
                f"{left['judgeModel']!r} vs {right['judgeModel']!r}"
            )

        dimensions: dict[str, SymmetricDimensionScore] = {}
        for dimension in DIMENSIONS:
            symmetric = average_order_records(
                left["dimensions"][dimension], right["dimensions"][dimension]
            )
            dimensions[dimension] = symmetric
            dimension_deltas.append(abs(symmetric.original_dim_score - symmetric.reverse_dim_score))
            criterion_deltas.extend(
                abs(item.original_score - item.reverse_score) for item in symmetric.criteria
            )

        symmetric_records.append(
            {
                "schemaVersion": SCHEMA_VERSION,
                "criterionOrder": "symmetric",
                "questionId": question_id,
                "answerId": answer_id,
                "judgeModel": left["judgeModel"],
                "dimensions": {
                    dimension: _dimension_payload(dimensions[dimension])
                    for dimension in DIMENSIONS
                },
            }
        )
        prediction_rows.append(
            {
                "taskId": 0,
                "questionId": question_id,
                "answerId": answer_id,
                **{f"{dimension}_score": dimensions[dimension].dim_score for dimension in DIMENSIONS},
            }
        )

    criterion_path = output_dir / "criterion_scores" / "symmetric.jsonl"
    prediction_path = output_dir / "predictions" / "osodt.tsv"
    diagnostic_path = output_dir / "diagnostics" / "order_effect.json"
    criterion_path.parent.mkdir(parents=True, exist_ok=True)
    with criterion_path.open("w", encoding="utf-8") as handle:
        for record in symmetric_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(prediction_path, sep="\t", index=False)
    atomic_write_json(
        diagnostic_path,
        {
            "documents": len(symmetric_records),
            "criterionComparisons": len(criterion_deltas),
            "dimensionComparisons": len(dimension_deltas),
            "meanAbsoluteCriterionOrderDifference": round(
                sum(criterion_deltas) / len(criterion_deltas), 6
            ),
            "meanAbsoluteDimensionOrderDifference": round(
                sum(dimension_deltas) / len(dimension_deltas), 6
            ),
        },
    )
    return prediction_path, criterion_path, diagnostic_path
