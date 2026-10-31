"""AdapR per-dimension scoring with adaptive IF checklist."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

# Fixed rubric for readability (question-agnostic)
_READABILITY_CRITERIA = [
    {
        "criterion": "逻辑结构",
        "weight": 0.40,
        "explanation": "仅读标题能否预测内容？论点与论据支撑是否清晰？",
    },
    {
        "criterion": "表达形式",
        "weight": 0.35,
        "explanation": "语言是否专业？图表与正文是否融合？",
    },
    {
        "criterion": "偏见检查",
        "weight": 0.25,
        "explanation": "立场是否公正？有无逻辑谬误或来源偏见？",
    },
]


# ── Prompt builders ────────────────────────────────────────────────────────────

def _build_content_dim_prompt(
    doc_text: str,
    question: str,
    dim: str,
    criteria: list[dict[str, Any]],
) -> str:
    n = len(criteria)
    criteria_text = ""
    for i, item in enumerate(criteria, 1):
        anchors = item.get("anchors", {})
        anchor_text = (
            f"\n      Excellent (9-10): {anchors.get('excellent', '')}"
            f"\n      Good (7-8):      {anchors.get('good', '')}"
            f"\n      Pass (5-6):      {anchors.get('pass', '')}"
            f"\n      Fail (0-4):      {anchors.get('fail', '')}"
        ) if anchors else ""
        criteria_text += (
            f"\n{'='*50}\n"
            f"{i}. {item['criterion']}（权重{item['weight']}）\n"
            f"   {item.get('explanation', '')}"
            f"{anchor_text}\n"
        )

    return (
        f"你是一位专业公正的报告评审专家。请阅读以下完整文档，"
        f"只对【{dim}】这一个维度进行评分。\n\n"
        f"【研究问题】：\n{question}\n\n"
        "【待评测文档】：\n（文档开始）\n"
        f"{doc_text}\n"
        "（文档结束）\n\n"
        f"【{dim} 评分子标准（共{n}条）】：{criteria_text}\n"
        f"请对上述{n}条子标准各自独立打分（0-10分）。\n\n"
        f"严格按以下 JSON 格式输出，不要有任何其他内容：\n"
        f'{{"{dim}": [分数1, 分数2, ...]}}'
    )


def _build_if_checklist_prompt(
    doc_text: str,
    question: str,
    checklist: list[dict[str, Any]],
) -> str:
    n = len(checklist)
    items_text = "".join(
        f"  {i}. {item['requirement']}（权重{item['weight']}）\n"
        for i, item in enumerate(checklist, 1)
    )
    return (
        "你是一位专业公正的报告评审专家。请阅读以下完整文档，"
        "只对【instruction_following（指令遵循）】这一个维度进行评分。\n\n"
        "【instruction_following（指令遵循）评分说明】\n"
        "逐条核查文档是否满足研究问题的每项具体要求：\n"
        "  9-10分：该要求被完整、清晰地覆盖，有充分内容支撑\n"
        "  7-8分：大体覆盖，有少量遗漏或论述不够充分\n"
        "  5-6分：部分覆盖，存在明显遗漏\n"
        "  0-4分：基本未覆盖或严重偏离该要求\n\n"
        f"【研究问题】：\n{question}\n\n"
        "【待评测文档】：\n（文档开始）\n"
        f"{doc_text}\n"
        "（文档结束）\n\n"
        f"【instruction_following 需求清单（共{n}条）】：\n"
        f"{items_text}\n"
        "请对每条需求逐一核查并打分（0-10分），再按权重加权。\n\n"
        "严格按以下 JSON 格式输出，不要有任何其他内容：\n"
        '{"instruction_following": [分数1, 分数2, ...]}'
    )


def _build_readability_prompt(doc_text: str) -> str:
    criteria_text = "".join(
        f"\n{'='*50}\n{i}. {item['criterion']}（权重{item['weight']}）\n   {item['explanation']}\n"
        for i, item in enumerate(_READABILITY_CRITERIA, 1)
    )
    return (
        "你是一位专业公正的报告评审专家。请阅读以下完整文档，"
        "只对【readability（可读性）】这一个维度进行评分。\n\n"
        "【待评测文档】：\n（文档开始）\n"
        f"{doc_text}\n"
        "（文档结束）\n\n"
        f"【可读性评分标准（共3个子维度）】：{criteria_text}\n\n"
        "请对上述3个子维度各自独立打分（0-10分）。\n"
        "评分基准：5-6分为中等水平，根据实际内容上下浮动。\n\n"
        "严格按以下 JSON 格式输出，不要有任何其他内容：\n"
        '{"readability": [分数1, 分数2, 分数3]}'
    )


# ── Score parser ───────────────────────────────────────────────────────────────

def _parse_dim_score(
    text: str, dim: str, criteria: list[dict[str, Any]]
) -> float | None:
    clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

    def extract(data: Any) -> float | None:
        scores = data.get(dim)
        if scores is None:
            for k, v in data.items():
                if dim[:8].lower() in k.lower() or k.lower() in dim.lower():
                    scores = v
                    break
        if scores is None and len(data) == 1:
            scores = next(iter(data.values()))
        if not isinstance(scores, list) or len(scores) != len(criteria):
            return None
        weighted = sum(
            float(s) * float(criteria[i]["weight"])
            for i, s in enumerate(scores)
        )
        return round(max(0.0, min(10.0, weighted)), 4)

    for candidate in [clean, text]:
        try:
            return extract(json.loads(candidate))
        except Exception:
            pass
        m = re.search(r"\{.*\}", candidate, re.DOTALL)
        if m:
            try:
                return extract(json.loads(m.group()))
            except Exception:
                pass
    return None


# ── Per-document scorer ────────────────────────────────────────────────────────

class AdapRScorer:
    """Score one document across four dimensions using the AdapR protocol."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        cache_dir: Path | None = None,
        max_retries: int = 3,
        extra_body: dict | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.extra_body = extra_body or {}
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, answer_id: str, dim: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{answer_id}_{dim}.json"

    def _call_llm(self, prompt: str, answer_id: str, dim: str) -> str:
        cache_path = self._cache_path(answer_id, dim)
        if cache_path and cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

        kwargs: dict[str, Any] = {}
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=2000,
                    **kwargs,
                )
                text = resp.choices[0].message.content or ""
                if cache_path:
                    cache_path.write_text(text, encoding="utf-8")
                return text
            except Exception as e:
                print(f"[Score] {answer_id}/{dim} attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(2**attempt)
        return ""

    def score(
        self,
        doc_text: str,
        question: str,
        criteria: dict[str, list[dict]],
        answer_id: str,
        if_checklist: list[dict] | None = None,
    ) -> dict[str, float]:
        """Return {dim: score} for all four dimensions."""
        results: dict[str, float] = {}

        # Comprehensiveness
        c_items = criteria["comprehensiveness"]
        c_prompt = _build_content_dim_prompt(doc_text, question, "comprehensiveness", c_items)
        c_raw = self._call_llm(c_prompt, answer_id, "comprehensiveness")
        c_score = _parse_dim_score(c_raw, "comprehensiveness", c_items)
        results["comprehensiveness"] = c_score if c_score is not None else 5.0

        # Insight
        i_items = criteria["insight"]
        i_prompt = _build_content_dim_prompt(doc_text, question, "insight", i_items)
        i_raw = self._call_llm(i_prompt, answer_id, "insight")
        i_score = _parse_dim_score(i_raw, "insight", i_items)
        results["insight"] = i_score if i_score is not None else 5.0

        # Instruction following: checklist for Type A, criteria for Type B
        if if_checklist:
            if_prompt = _build_if_checklist_prompt(doc_text, question, if_checklist)
            if_items = if_checklist
        else:
            if_items = criteria["instruction_following"]
            if_prompt = _build_content_dim_prompt(
                doc_text, question, "instruction_following", if_items
            )
        if_raw = self._call_llm(if_prompt, answer_id, "instruction_following")
        if_score = _parse_dim_score(if_raw, "instruction_following", if_items)
        results["instruction_following"] = if_score if if_score is not None else 5.0

        # Readability (fixed rubric, question-agnostic)
        r_prompt = _build_readability_prompt(doc_text)
        r_raw = self._call_llm(r_prompt, answer_id, "readability")
        r_score = _parse_dim_score(r_raw, "readability", _READABILITY_CRITERIA)
        results["readability"] = r_score if r_score is not None else 5.0

        return results
