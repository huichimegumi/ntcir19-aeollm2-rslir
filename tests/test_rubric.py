import json
from pathlib import Path

import pytest

from aeollm2_osodt.constants import DIMENSIONS
from aeollm2_osodt.rubric import load_rubric


def _payload():
    return {
        "id": 1,
        "prompt": "  task text with intentional spaces  ",
        "dimension_weight": {dimension: 0.25 for dimension in DIMENSIONS},
        "criterions": {
            dimension: [
                {
                    "criterion": "  criterion text  ",
                    "explanation": "  explanation text  ",
                    "weight": 1.0,
                }
            ]
            for dimension in DIMENSIONS
        },
    }


def _write(tmp_path: Path, payload: dict):
    path = tmp_path / "criterion1.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_rubric_loader_preserves_prompt_text_and_weight_representation(tmp_path: Path):
    _write(tmp_path, _payload())
    rubric = load_rubric(tmp_path, 1)
    criterion = rubric.criteria["insight"][0]
    assert rubric.prompt == "  task text with intentional spaces  "
    assert criterion.text == "  criterion text  "
    assert criterion.explanation == "  explanation text  "
    assert criterion.weight == 1.0
    assert criterion.prompt_weight == "1.0"


def test_rubric_loader_requires_all_official_dimension_weights(tmp_path: Path):
    payload = _payload()
    del payload["dimension_weight"]["readability"]
    _write(tmp_path, payload)
    with pytest.raises(ValueError, match="missing dimension weights"):
        load_rubric(tmp_path, 1)


def test_rubric_loader_rejects_nonpositive_criterion_weight_total(tmp_path: Path):
    payload = _payload()
    payload["criterions"]["insight"][0]["weight"] = 0.0
    _write(tmp_path, payload)
    with pytest.raises(ValueError, match="non-positive total criterion weight"):
        load_rubric(tmp_path, 1)
