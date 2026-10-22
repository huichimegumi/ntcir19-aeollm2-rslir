import json
from pathlib import Path

import pandas as pd
import pytest

from aeollm2_osodt.aggregate import aggregate_osodt
from aeollm2_osodt.constants import DIMENSIONS


def _record(order: str, *, model: str = "judge", score_shift: float = 0.0):
    dimensions = {}
    for dimension in DIMENSIONS:
        criteria = []
        scores = [6.0 + score_shift, 8.0 + score_shift]
        prompt_positions = [1, 2] if order == "original" else [2, 1]
        for index, (score, prompt_position) in enumerate(zip(scores, prompt_positions), 1):
            criteria.append(
                {
                    "criterionId": f"{dimension}_{index}",
                    "originalIndex": index,
                    "promptPosition": prompt_position,
                    "criterion": f"{dimension} criterion {index}",
                    "weight": 0.5,
                    "score": score,
                }
            )
        dimensions[dimension] = {
            "criterionOrder": order,
            "criteria": criteria,
            "dim_score": 7.0 + score_shift,
        }
    return {
        "schemaVersion": 1,
        "criterionOrder": order,
        "questionId": 1,
        "answerId": "Doc_001",
        "judgeModel": model,
        "dimensions": dimensions,
    }


def _write(path: Path, record: dict):
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


def test_offline_aggregate_writes_symmetric_jsonl_predictions_and_diagnostics(tmp_path: Path):
    original = tmp_path / "original.jsonl"
    reverse = tmp_path / "reverse.jsonl"
    _write(original, _record("original", score_shift=0.0))
    _write(reverse, _record("reverse", score_shift=1.0))

    prediction_path, criterion_path, diagnostic_path = aggregate_osodt(
        original, reverse, tmp_path / "out"
    )

    predictions = pd.read_csv(prediction_path, sep="\t")
    assert predictions.loc[0, "comprehensiveness_score"] == 7.5
    symmetric = json.loads(criterion_path.read_text(encoding="utf-8").splitlines()[0])
    criterion = symmetric["dimensions"]["insight"]["criteria"][0]
    assert criterion["criterionId"] == "insight_1"
    assert criterion["originalScore"] == 6.0
    assert criterion["reverseScore"] == 7.0
    assert criterion["score"] == 6.5
    diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert diagnostics["documents"] == 1
    assert diagnostics["meanAbsoluteCriterionOrderDifference"] == 1.0


def test_offline_aggregate_rejects_judge_model_mismatch(tmp_path: Path):
    original = tmp_path / "original.jsonl"
    reverse = tmp_path / "reverse.jsonl"
    _write(original, _record("original", model="a"))
    _write(reverse, _record("reverse", model="b"))

    with pytest.raises(ValueError, match="judge model mismatch"):
        aggregate_osodt(original, reverse, tmp_path / "out")


def test_offline_aggregate_rejects_reverse_prompt_position_corruption(tmp_path: Path):
    original = tmp_path / "original.jsonl"
    reverse = tmp_path / "reverse.jsonl"
    left = _record("original")
    right = _record("reverse")
    right["dimensions"]["insight"]["criteria"][0]["promptPosition"] = 1
    _write(original, left)
    _write(reverse, right)

    with pytest.raises(ValueError, match="promptPosition"):
        aggregate_osodt(original, reverse, tmp_path / "out")


@pytest.mark.parametrize('corrupt', ['duplicate_id','missing_id','weight','text','nonfinite','fractional_index','dimension_score'])
def test_aggregate_rejects_criterion_corruption(tmp_path,corrupt):
    import copy
    original,reverse=tmp_path/'original.jsonl',tmp_path/'reverse.jsonl'
    left,right=_record('original'),_record('reverse')
    criteria=right['dimensions']['insight']['criteria']
    if corrupt=='duplicate_id': criteria[1]['criterionId']=criteria[0]['criterionId']
    if corrupt=='missing_id': criteria.pop()
    if corrupt=='weight': criteria[0]['weight']=0.6
    if corrupt=='text': criteria[0]['criterion']='changed'
    if corrupt=='nonfinite': criteria[0]['score']=float('nan')
    if corrupt=='fractional_index': criteria[0]['originalIndex']=1.5
    if corrupt=='dimension_score': right['dimensions']['insight']['dim_score']=7.0001
    _write(original,left);_write(reverse,right)
    with pytest.raises(ValueError): aggregate_osodt(original,reverse,tmp_path/'out')


def test_record_serialization_order_does_not_change_alignment(tmp_path):
    original,reverse=tmp_path/'original.jsonl',tmp_path/'reverse.jsonl'
    left,right=_record('original'),_record('reverse',score_shift=1)
    for dim in right['dimensions'].values(): dim['criteria'].reverse()
    _write(original,left);_write(reverse,right)
    pred,_,_=aggregate_osodt(original,reverse,tmp_path/'out')
    assert pd.read_csv(pred,sep='\t').loc[0,'insight_score']==7.5
