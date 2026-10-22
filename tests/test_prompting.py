from aeollm2_osodt.constants import BEGIN_MARKER, END_MARKER
from aeollm2_osodt.prompting import build_odt_messages, criteria_in_prompt_order
from aeollm2_osodt.rubric import Criterion, Rubric


def _rubric():
    criteria = tuple(
        Criterion(
            f"insight_{i}",
            i,
            f"Criterion {i}",
            f"Explanation {i}",
            1.0,
            "1.0",
        )
        for i in range(1, 4)
    )
    return Rubric(
        question_id=1,
        prompt="Evaluate the report.",
        dimension_weights={
            "comprehensiveness": 1,
            "insight": 1,
            "instruction_following": 1,
            "readability": 1,
        },
        criteria={
            "comprehensiveness": criteria,
            "insight": criteria,
            "instruction_following": criteria,
            "readability": criteria,
        },
    )


def test_reverse_changes_only_criterion_presentation_order():
    rubric = _rubric()
    assert [c.criterion_id for c in criteria_in_prompt_order(rubric, "insight", "original")] == [
        "insight_1", "insight_2", "insight_3"
    ]
    assert [c.criterion_id for c in criteria_in_prompt_order(rubric, "insight", "reverse")] == [
        "insight_3", "insight_2", "insight_1"
    ]


def test_paper_v1_odt_system_message_matches_formal_criterion_order_runner():
    system, _ = build_odt_messages(
        "document",
        _rubric(),
        {"insight": "anchor"},
        "insight",
    )
    assert system["content"] == (
        f"只允许返回 JSON。输出必须以单独一行的 {BEGIN_MARKER} 开始，"
        f"并以单独一行的 {END_MARKER} 结束。"
        "请先在内部完成证据检查和理由判断后再打分，但不要输出 reasoning。"
        "只返回当前单一维度的数字分数数组，不要输出解释或其他文字。"
    )


def test_paper_v1_odt_prompt_contains_markers_full_document_and_raw_weight_text():
    system, user = build_odt_messages(
        "FULL DOCUMENT CONTENT",
        _rubric(),
        {"insight": "ANCHOR"},
        "insight",
        criterion_order="reverse",
    )
    assert BEGIN_MARKER in system["content"]
    assert END_MARKER in user["content"]
    assert "FULL DOCUMENT CONTENT" in user["content"]
    assert user["content"].index("Criterion 3") < user["content"].index("Criterion 1")
    assert "（权重 1.0）" in user["content"]
