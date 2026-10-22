from aeollm2_osodt.methods.odt import map_prompt_scores_to_canonical
from aeollm2_osodt.methods.osodt import average_order_records
from aeollm2_osodt.rubric import Criterion, Rubric


def _rubric():
    criteria = tuple(
        Criterion(f"insight_{i}", i, f"C{i}", "", weight)
        for i, weight in enumerate((0.2, 0.3, 0.5), 1)
    )
    return Rubric(
        question_id=1,
        prompt="task",
        dimension_weights={"comprehensiveness": 1, "insight": 1, "instruction_following": 1, "readability": 1},
        criteria={
            "comprehensiveness": criteria,
            "insight": criteria,
            "instruction_following": criteria,
            "readability": criteria,
        },
    )


def test_reverse_scores_are_mapped_back_to_canonical_ids():
    rubric = _rubric()
    record = map_prompt_scores_to_canonical(
        rubric,
        "insight",
        [8.0, 7.0, 6.0],
        criterion_order="reverse",
    )
    assert [item.criterion_id for item in record.criteria] == ["insight_1", "insight_2", "insight_3"]
    assert [item.score for item in record.criteria] == [6.0, 7.0, 8.0]
    assert [item.prompt_position for item in record.criteria] == [3, 2, 1]


def test_osodt_averages_same_criterion_identity_not_array_position():
    rubric = _rubric()
    original = map_prompt_scores_to_canonical(
        rubric, "insight", [2.0, 4.0, 8.0], criterion_order="original"
    )
    reverse = map_prompt_scores_to_canonical(
        rubric, "insight", [6.0, 10.0, 4.0], criterion_order="reverse"
    )
    symmetric = average_order_records(original, reverse)
    assert [item.score for item in symmetric.criteria] == [3.0, 7.0, 7.0]
    assert symmetric.dim_score == 6.2


def test_historical_rounding_occurs_before_weighted_aggregation():
    record = map_prompt_scores_to_canonical(_rubric(), 'insight', [1.23445, 2.34565, 7.89015], criterion_order='original')
    expected = [round(v,4) for v in [1.23445,2.34565,7.89015]]
    assert [c.score for c in record.criteria] == expected
    assert record.dim_score == round(sum(w*s for w,s in zip([0.2,0.3,0.5],expected)),4)


def test_invalid_raw_score_is_rejected_before_rounding():
    import pytest
    for value in [10.00001, -0.00001, float('nan'), float('inf'), True]:
        with pytest.raises(ValueError):
            map_prompt_scores_to_canonical(_rubric(), 'insight', [value,2,3], criterion_order='original')
