"""Original Chinese SRAD prompts retained from the formal-run implementation."""

from __future__ import annotations

FULLTEXT_DIMS = ["readability"]

READABILITY_RUBRICS = {}

DIM_SCORING_SCALES = {}

BEGIN_MARKER = "###BEGIN_JSON###"

END_MARKER = "###END_JSON###"

DIMENSION_STRATEGIES = {'readability': '维度专属评审策略：阅读体验与表达结构检查。\n\n本维度从读者体验和表达清晰度出发评分，并与内容覆盖度、分析深度解耦。\n打分前，请在内部检查：\n1. 结构是否清楚、易于跟随？\n2. 标题、段落、列表和表格是否帮助理解？\n3. 语言是否准确、简洁、连贯，并符合任务所需的专业程度？\n4. 是否存在冗余、逻辑跳跃、无支撑偏见、逻辑谬误，或增加阅读负担的表达方式？\n\n常见误判提醒：不要因为内容全面或观点深刻就给可读性高分；也不要因为观点一般就自动降低可读性。文章可以“内容一般但清楚易读”，也可以“内容丰富但组织混乱”。'}

def construct_single_dimension_prompt(doc_text: str, rubric: dict, dim: str) -> str:
    if dim not in FULLTEXT_DIMS:
        raise ValueError(f"Unknown dimension: {dim}")

    criterions = rubric.get("criterions", {})
    items = criterions.get(dim, [])
    dim_scale = DIM_SCORING_SCALES.get(dim, "")
    strategy = DIMENSION_STRATEGIES.get(dim, "")

    criteria_text = ""
    for idx, item in enumerate(items, 1):
        criteria_text += f"{idx}. 准则：{item.get('criterion', '')}（权重 {item.get('weight', 0)}）\n"
        criteria_text += f"   说明：{item.get('explanation', '')}\n"

    prompt = (
        f"你是一名专门评估 `{dim}` 维度的报告评审专家。\n\n"
        "【本维度评审目标】\n"
        f"请只评价 `{dim}` 这一维度。除非当前准则明确要求，否则不要把其他维度的表现混入本维度评分。\n\n"
        "【本维度评审流程策略】\n"
        f"{strategy}\n\n"
        "【评分锚点】\n"
        f"{dim_scale}\n\n"
        "【当前任务】\n"
        f"{rubric.get('prompt', '')}\n\n"
        "【待评估文档】\n"
        "（文档开始）\n"
        f"{doc_text}\n"
        "（文档结束）\n\n"
        "【本维度具体准则】\n"
        f"{criteria_text}\n"
        "【内部评审步骤】\n"
        "1. 请在内部根据每条准则检查文档证据。\n"
        "2. 请在内部根据本维度策略识别遗漏、浅层处理、指令遵循问题或表达问题。\n"
        "3. 按上方准则顺序，为每条准则给出一个 0-10 分（可小数）。\n"
        "4. 不要输出 reasoning、证据、准则名称、Markdown 或任何额外文字。\n\n"
        "请严格按如下 JSON 结构输出：\n"
        "{\n"
        f'  "{dim}": [0-10, 0-10, ...]\n'
        "}\n\n"
        "数组长度必须严格等于本维度准则数量："
        f"{len(items)}。\n"
        f"输出必须以单独一行的 {BEGIN_MARKER} 开始，并以单独一行的 {END_MARKER} 结束。"
    )
    return prompt

def construct_paragraph_readability_prompt(rubric: dict, blocks: list[tuple[int, str]], total_blocks: int) -> str:
    block_text = ""
    for idx, text in blocks:
        block_text += f"\n[文章部分 {idx}/{total_blocks}]\n{text}\n"

    dim_scale = DIM_SCORING_SCALES.get("readability", "")
    strategy = DIMENSION_STRATEGIES.get("readability", "")

    return (
        "你是一名专门评估文章部分级可读性的报告评审专家。\n\n"
        "【评审目标】\n"
        "请只评价每个文章部分自身的临时可读性分数 s_i。这个分数不是最终全文分数，"
        "不要考虑该部分所在位置的权重，位置权重会由程序另行计算。\n\n"
        "【当前任务】\n"
        f"{rubric.get('prompt', '')}\n\n"
        "【可读性评分锚点】\n"
        f"{dim_scale}\n\n"
        "【评审策略】\n"
        f"{strategy}\n\n"
        "【部分级评分范围】\n"
        "对每个文章部分，请关注：表达是否清楚、该部分内部组织是否顺畅、"
        "上下文衔接是否自然、信息密度是否造成阅读负担、是否有明显歧义、偏见或夸张断言。\n"
        "不要因为该部分内容全面、洞察深或严格遵循任务就自动给高可读性分。\n\n"
        "【待评估文章部分】\n"
        f"{block_text}\n\n"
        "请严格按如下 JSON 结构输出，必须覆盖上面出现的所有 index：\n"
        "{\n"
        '  "paragraph_scores": [\n'
        '    {"index": 1, "score": 0-10},\n'
        '    {"index": 2, "score": 0-10}\n'
        "  ]\n"
        "}\n\n"
        "只输出数字分数，不要输出 reason、解释、Markdown 或任何额外文字。\n"
        f"输出必须以单独一行的 {BEGIN_MARKER} 开始，并以单独一行的 {END_MARKER} 结束。"
    )

def construct_fullcontext_readability_prompt(doc_text: str, rubric: dict, blocks: list[tuple[int, str]]) -> str:
    block_text = ""
    total_blocks = len(blocks)
    for idx, text in blocks:
        block_text += f"\n[文章部分 {idx}/{total_blocks}]\n{text}\n"

    dim_scale = DIM_SCORING_SCALES.get("readability", "")
    strategy = DIMENSION_STRATEGIES.get("readability", "")
    required_indices = ", ".join(str(idx) for idx, _ in blocks)

    return (
        "你是一名专门评估全文上下文中文章部分级可读性的报告评审专家。\n\n"
        "【评审目标】\n"
        "请先阅读完整文档，再基于全文结构和上下文，为每个文章部分给出临时可读性分数 s_i。"
        "这个分数不是最终全文分数，不要考虑部分位置权重，位置权重会由程序另行计算。\n\n"
        "【当前任务】\n"
        f"{rubric.get('prompt', '')}\n\n"
        "【可读性评分锚点】\n"
        f"{dim_scale}\n\n"
        "【评审策略】\n"
        f"{strategy}\n\n"
        "【完整文档】\n"
        "（文档开始）\n"
        f"{doc_text}\n"
        "（文档结束）\n\n"
        "【需要逐一评分的文章部分】\n"
        f"{block_text}\n\n"
        "【评分要求】\n"
        "1. 必须在理解全文结构、前后衔接和部分功能后，为每个文章部分评分。\n"
        "2. 分数只评价可读性：表达清晰度、部分内部组织、结构衔接、冗余程度、歧义、偏见和阅读负担。\n"
        "3. 不要因为内容全面、洞察深或指令遵循好就自动给高可读性分。\n"
        "4. 必须覆盖所有原始 index，禁止重新编号，禁止遗漏。\n"
        f"5. 必须返回这些 index：{required_indices}。\n\n"
        "请严格按如下 JSON 结构输出：\n"
        "{\n"
        '  "paragraph_scores": [\n'
        '    {"index": 原始index, "score": 0-10}\n'
        "  ]\n"
        "}\n\n"
        "只输出数字分数，不要输出 reason、解释、Markdown 或任何额外文字。\n"
        f"输出必须以单独一行的 {BEGIN_MARKER} 开始，并以单独一行的 {END_MARKER} 结束。"
    )

def construct_global_readability_prompt(doc_text: str, rubric: dict) -> str:
    prompt = construct_single_dimension_prompt(doc_text, rubric, "readability")
    return (
        prompt
        + "\n\n【全文结构校正说明】\n"
        "这一次评分只用于全文结构校正，请从整篇文章角度关注逻辑结构、表达形式、偏见检查。"
        "不要考虑段落位置权重，也不要输出段落级分数。"
    )

def construct_sequential_global_readability_prompt(
    reference_doc_text: str,
    reference_doc_id: str,
    reference_global_score: float,
    current_doc_text: str,
    current_doc_id: str,
    rubric: dict,
) -> str:
    """Build a free-calibration Global prompt using the previous document as reference."""
    items = rubric.get("criterions", {}).get("readability", [])
    dim_scale = DIM_SCORING_SCALES.get("readability", "")
    strategy = DIMENSION_STRATEGIES.get("readability", "")

    criteria_text = ""
    for idx, item in enumerate(items, 1):
        criteria_text += (
            f"{idx}. 准则：{item.get('criterion', '')}"
            f"（权重 {item.get('weight', 0)}）\n"
            f"   说明：{item.get('explanation', '')}\n"
        )

    return (
        "你是一名专门评估长篇报告全文可读性的严格评审专家。\n\n"
        "【评审目标】\n"
        "请只评价当前文档的 Global Readability。你还会看到同一任务下上一篇已经评估过的"
        "参考文档及其 Global Readability 分数。参考文档和分数只用于帮助你保持跨文档"
        "评分尺度的一致性；它们不是标准答案，也不是当前文档应模仿的写作模板。\n\n"
        "【当前任务】\n"
        f"{rubric.get('prompt', '')}\n\n"
        "【可读性评审策略】\n"
        f"{strategy}\n\n"
        "【评分锚点】\n"
        f"{dim_scale}\n\n"
        "【本维度具体准则】\n"
        f"{criteria_text}\n"
        "【参考文档信息】\n"
        f"Reference document ID: {reference_doc_id}\n"
        f"Reference Global Readability score: {float(reference_global_score):.4f}\n\n"
        "[REFERENCE_DOCUMENT_START]\n"
        f"{reference_doc_text}\n"
        "[REFERENCE_DOCUMENT_END]\n\n"
        "【当前待评文档】\n"
        f"Current document ID: {current_doc_id}\n\n"
        "[CURRENT_DOCUMENT_START]\n"
        f"{current_doc_text}\n"
        "[CURRENT_DOCUMENT_END]\n\n"
        "【内部评审步骤】\n"
        "1. 先根据 rubric 独立检查当前文档在每条可读性准则上的实际表现。\n"
        "2. 再比较参考文档和当前文档的整体阅读体验，用参考分数校准本次评分尺度。\n"
        "3. 当前文档可以高于、低于或等于参考分数；不存在预设的最低或最高分差。\n"
        "4. 如果两篇文档质量接近，可以给出非常接近甚至相同的分数；如果差异明显，"
        "可以自由给出更大的分差。\n"
        "5. 不要机械复制参考分数，也不要为了制造区分而强行拉开分数。\n"
        "6. 不要评价全面性、洞察或指令遵循，不要输出 reasoning 或比较说明。\n"
        "7. 按上方准则顺序，只为当前文档给出 0-10 分（可小数）。\n\n"
        "请严格按如下 JSON 结构输出：\n"
        "{\n"
        '  "readability": [0-10, 0-10, ...]\n'
        "}\n\n"
        f"数组长度必须严格等于可读性准则数量：{len(items)}。\n"
        f"输出必须以单独一行的 {BEGIN_MARKER} 开始，并以单独一行的 {END_MARKER} 结束。"
    )

def _system_json_message() -> str:
    return (
        f"只允许返回 JSON。输出必须以单独一行的 {BEGIN_MARKER} 开始，"
        f"并以单独一行的 {END_MARKER} 结束。"
        "请先在内部完成证据检查和理由判断后再打分，但不要输出 reasoning。"
        "只返回用户要求的数字分数 JSON，不要输出解释或其他文字。"
    )
