"""Canonical Joint and ODT prompt construction for the paper protocol."""

from __future__ import annotations

import json
from pathlib import Path

from .constants import BEGIN_MARKER, DIMENSIONS, END_MARKER
from .rubric import Criterion, Rubric

DIMENSION_STRATEGIES = {
    "comprehensiveness": """维度专属评审策略：覆盖性检查。

本维度评估文档是否系统覆盖任务和具体准则要求的关键方面。
打分前，请在内部按以下路径检查每条准则：
1. 这条准则要求覆盖哪些关键方面？
2. 文档是否明确覆盖了这些方面？
3. 相关内容只是简单提到、部分展开、基本覆盖，还是充分覆盖？
4. 是否存在会明显降低分数的关键遗漏？

常见误判提醒：不要因为文章篇幅长、章节多、术语多，或信息量大但偏离任务，就给全面性高分。高分必须来自任务关键方面被系统、实质地覆盖。""",
    "insight": """维度专属评审策略：深度与洞察检查。

本维度评估文档的分析深度和判断价值。请在内部区分以下层次：
1. 事实罗列。
2. 分类整理或信息组织。
3. 对关系、机制、原因、权衡或影响的分析。
4. 有支撑的批判性、综合性、前瞻性或独到洞察。

评审时请检查：文档是否解释了“为什么”和“如何”？是否超越资料汇总？结论是否有前文支撑？是否识别机制、趋势、矛盾、局限或未来影响？

常见误判提醒：不要把信息丰富、语言学术化、引用多或术语复杂直接等同于有洞察。本维度应严格评分。""",
    "instruction_following": """维度专属评审策略：硬性要求遵循检查。

本维度只评估文档对用户显式任务要求的遵循程度。请把原始任务和当前准则当作验收清单。
打分前，请在内部检查：
1. 任务明确提出了哪些要求？
2. 要求的范围、对象、数量、格式、理论维度、输出结构和重点是否满足？
3. 是否存在遗漏、偏离、数量错误或跑题内容？

常见误判提醒：内容质量好、表达流畅或分析深入，不能弥补没有遵守指令的问题。本维度只评价任务要求的完成度，不评价观点质量或文字水平。""",
    "readability": """维度专属评审策略：阅读体验与表达结构检查。

本维度从读者体验和表达清晰度出发评分，并与内容覆盖度、分析深度解耦。
打分前，请在内部检查：
1. 结构是否清楚、易于跟随？
2. 标题、段落、列表和表格是否帮助理解？
3. 语言是否准确、简洁、连贯，并符合任务所需的专业程度？
4. 是否存在冗余、逻辑跳跃、无支撑偏见、逻辑谬误，或增加阅读负担的表达方式？

常见误判提醒：不要因为内容全面或观点深刻就给可读性高分；也不要因为观点一般就自动降低可读性。文章可以“内容一般但清楚易读”，也可以“内容丰富但组织混乱”。""",
}


def load_prompt_scales(path: str | Path) -> dict[str, str]:
    """Load scoring anchors without rewriting their text."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    dimension_scales = payload.get("dim_scoring_scales")
    readability = payload.get("readability_rubrics")
    if not isinstance(dimension_scales, dict) or not isinstance(readability, dict):
        raise ValueError("prompt_scales must contain dim_scoring_scales and readability_rubrics")

    scales = {
        dimension: str(dimension_scales.get(dimension, ""))
        for dimension in DIMENSIONS
    }
    scales["readability"] = "\n\n".join(str(value) for value in readability.values())
    if any(not scales[dimension].strip() for dimension in DIMENSIONS):
        raise ValueError("prompt_scales does not define every evaluation dimension")
    return scales


def _criterion_lines(criteria: tuple[Criterion, ...]) -> str:
    return "\n".join(
        f"{position}. 准则：{criterion.text}（权重 {criterion.prompt_weight}）\n"
        f"   说明：{criterion.explanation}"
        for position, criterion in enumerate(criteria, 1)
    )


def criteria_in_prompt_order(rubric: Rubric, dimension: str, order: str) -> tuple[Criterion, ...]:
    if dimension not in DIMENSIONS:
        raise ValueError(f"unknown dimension: {dimension}")
    if order not in {"original", "reverse"}:
        raise ValueError("criterion order must be 'original' or 'reverse'")
    criteria = rubric.criteria[dimension]
    return criteria if order == "original" else tuple(reversed(criteria))


def _system_prompt(scope: str, prompt_protocol: str = "paper-v1") -> str:
    if prompt_protocol == "e9-schema-v1":
        return (
            "只允许返回严格 JSON 对象。请先在内部完成证据检查和理由判断后再打分，"
            f"但不要输出 reasoning。只返回{scope}的数字分数数组，不要输出解释或其他文字。"
        )
    if prompt_protocol != "paper-v1":
        raise ValueError(f"unknown prompt protocol: {prompt_protocol}")
    return (
        f"只允许返回 JSON。输出必须以单独一行的 {BEGIN_MARKER} 开始，"
        f"并以单独一行的 {END_MARKER} 结束。"
        "请先在内部完成证据检查和理由判断后再打分，但不要输出 reasoning。"
        f"只返回{scope}的数字分数数组，不要输出解释或其他文字。"
    )


def build_odt_messages(
    document_text: str,
    rubric: Rubric,
    scales: dict[str, str],
    dimension: str,
    *,
    criterion_order: str = "original",
    prompt_protocol: str = "paper-v1",
) -> tuple[dict[str, str], dict[str, str]]:
    criteria = criteria_in_prompt_order(rubric, dimension, criterion_order)
    criterion_gap = "\n\n" if prompt_protocol == "paper-v1" else "\n"
    prompt = (
        f"你是一名专门评估 `{dimension}` 维度的报告评审专家。\n\n"
        "【本维度评审目标】\n"
        f"请只评价 `{dimension}` 这一维度。除非当前准则明确要求，否则不要把其他维度的表现混入本维度评分。\n\n"
        "【本维度评审流程策略】\n"
        f"{DIMENSION_STRATEGIES[dimension]}\n\n"
        "【评分锚点】\n"
        f"{scales[dimension]}\n\n"
        "【当前任务】\n"
        f"{rubric.prompt}\n\n"
        "【待评估文档】\n"
        "（文档开始）\n"
        f"{document_text}\n"
        "（文档结束）\n\n"
        "【本维度具体准则】\n"
        f"{_criterion_lines(criteria)}{criterion_gap}"
        "【内部评审步骤】\n"
        "1. 请在内部根据每条准则检查文档证据。\n"
        "2. 请在内部根据本维度策略识别遗漏、浅层处理、指令遵循问题或表达问题。\n"
        "3. 按上方准则顺序，为每条准则给出一个 0-10 分（可小数）。\n"
        "4. 不要输出 reasoning、证据、准则名称、Markdown 或任何额外文字。\n\n"
    )
    if prompt_protocol == "paper-v1":
        prompt += (
            "请严格按如下 JSON 结构输出：\n"
            "{\n"
            f'  "{dimension}": [0-10, 0-10, ...]\n'
            "}\n\n"
            f"数组长度必须严格等于本维度准则数量：{len(criteria)}。\n"
            f"输出必须以单独一行的 {BEGIN_MARKER} 开始，并以单独一行的 {END_MARKER} 结束。"
        )
    else:
        prompt += (
            f"请严格输出 JSON 对象：键为 `{dimension}`，值为数字数组。"
            f"数组长度必须严格等于本维度准则数量：{len(criteria)}。"
        )
    return (
        {"role": "system", "content": _system_prompt("当前单一维度", prompt_protocol)},
        {"role": "user", "content": prompt},
    )


def build_joint_messages(
    document_text: str,
    rubric: Rubric,
    scales: dict[str, str],
    *,
    prompt_protocol: str = "paper-v1",
) -> tuple[dict[str, str], dict[str, str]]:
    sections: list[str] = []
    counts: dict[str, int] = {}
    for dimension in DIMENSIONS:
        items = rubric.criteria[dimension]
        counts[dimension] = len(items)
        criteria = "\n".join(
            f"- 准则: {item.text}（权重 {item.prompt_weight}）\n  - 说明: {item.explanation}"
            for item in items
        )
        sections.append(
            f"{'=' * 60}\n{scales[dimension]}\n"
            f"以下是该维度下的 {len(items)} 项具体准则，请按锚点综合加权评分：\n{criteria}"
        )
    count_text = "，".join(f"{dimension}={counts[dimension]}" for dimension in DIMENSIONS)
    joined_sections = "\n\n".join(sections)
    prompt = (
        "你是一位专业公正的报告评审专家。请阅读以下完整文档，根据【评分标准】"
        "对四个维度进行 0-10 分打分，不得评分虚高。\n\n"
        f"【待评测文档】\n(文档开始)\n{document_text}\n(文档结束)\n\n"
        f"【评分标准】\n{joined_sections}\n\n"
        "请按每个维度中的每条具体准则（criterion）进行独立打分（0-10）。\n"
        "请先在内部仔细思考每条准则对应的依据与理由，再给出分数。\n"
        "评分遵循比较规则：首先假设基线分数为5-6分，然后根据文档内容和评分规则进行分数档次调整。\n"
        "最终只返回分数数组。\n"
        f"每个数组长度必须严格等于该维度准则数：{count_text}。\n"
        "输出严格 JSON 对象，键为 comprehensiveness、insight、instruction_following、readability，"
        "值为对应的数字数组。不要输出 reason、解释、分析、Markdown、代码块或其他文字。\n"
        f"输出必须以单独一行的 {BEGIN_MARKER} 开始，并以单独一行的 {END_MARKER} 结束。"
    )
    if prompt_protocol == "e9-schema-v1":
        prompt = prompt.rsplit("\n输出必须以单独一行的 ", 1)[0]
    return (
        {"role": "system", "content": _system_prompt("四个维度", prompt_protocol)},
        {"role": "user", "content": prompt},
    )
