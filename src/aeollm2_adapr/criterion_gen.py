"""AutoR: LLM-based question-specific criterion generation for AdapR."""

from __future__ import annotations

import ast
import json
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

CONTENT_DIMS = ["comprehensiveness", "insight", "instruction_following"]

_DIM_DESCRIPTIONS = {
    "comprehensiveness": (
        "Comprehensiveness: whether the report covers all key angles and sub-topics "
        "relevant to the research question"
    ),
    "insight": (
        "Insight: whether the report goes beyond surface facts to provide deep analysis "
        "and original perspectives"
    ),
    "instruction_following": (
        "Instruction Following: whether the report accurately responds to the specific "
        "requirements stated in the research question"
    ),
}


# ── Criterion generation (AutoR) ──────────────────────────────────────────────

def _build_criterion_prompt(question: str) -> str:
    dim_list = "\n".join(f"- {k}: {v}" for k, v in _DIM_DESCRIPTIONS.items())
    return (
        "你是一位深度研究报告评审专家。请针对以下研究问题，为评审报告质量设计具体的评分子标准。\n\n"
        f"【研究问题】：\n{question}\n\n"
        "请为以下3个评分维度各设计3~5条具体子标准，每条子标准需要：\n"
        "1. 紧扣该研究问题的核心内容（不要泛泛而谈）\n"
        "2. 有明确的权重（同一维度所有权重之和 = 1.0）\n"
        "3. 有1~2句详细说明（解释该标准具体考察什么内容、判断依据是什么）\n"
        "4. 有4档评分锚点，每档1句话（15~25字），描述该标准在该得分段的具体表现特征：\n"
        "   - excellent（9-10分）：在该标准上表现卓越的具体特征\n"
        "   - good（7-8分）：表现较好但仍有提升空间的具体特征\n"
        "   - pass（5-6分）：基本达标但存在明显不足的具体特征\n"
        "   - fail（0-4分）：明显缺失或严重不达标的具体特征\n\n"
        f"评分维度：\n{dim_list}\n\n"
        "严格按以下 JSON 格式输出，不要有任何其他内容：\n"
        "{\n"
        '  "comprehensiveness": [\n'
        "    {\n"
        '      "criterion": "子标准描述",\n'
        '      "weight": 0.3,\n'
        '      "explanation": "2~3句详细说明",\n'
        '      "anchors": {\n'
        '        "excellent": "9-10: 优秀表现的具体描述",\n'
        '        "good": "7-8: 良好表现的具体描述",\n'
        '        "pass": "5-6: 及格表现的具体描述",\n'
        '        "fail": "0-4: 不及格表现的具体描述"\n'
        "      }\n"
        "    },\n"
        "    ...\n"
        "  ],\n"
        '  "insight": [...],\n'
        '  "instruction_following": [...]\n'
        "}"
    )


def _parse_criterion(text: str) -> dict[str, list[dict[str, Any]]] | None:
    clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

    def validate(data: Any) -> dict | None:
        if not isinstance(data, dict):
            return None
        for dim in CONTENT_DIMS:
            if dim not in data:
                return None
            items = data[dim]
            if not items or not isinstance(items, list):
                return None
            total_w = sum(float(it.get("weight", 0)) for it in items)
            if total_w <= 0:
                return None
            for it in items:
                it["weight"] = round(float(it["weight"]) / total_w, 4)
        return data

    for candidate in [clean, text]:
        try:
            return validate(json.loads(candidate))
        except Exception:
            pass
        m = re.search(r"\{.*\}", candidate, re.DOTALL)
        if m:
            try:
                return validate(json.loads(m.group()))
            except Exception:
                pass
        try:
            return validate(ast.literal_eval(candidate))
        except Exception:
            pass
    return None


class AutoRCriterionGenerator:
    """Generate question-specific scoring criteria for C, I, and IF dimensions."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        cache_dir: Path | None = None,
        max_retries: int = 3,
    ) -> None:
        self.client = client
        self.model = model
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, question_id: int) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"criterion_q{question_id}.json"

    def generate(self, question: str, question_id: int) -> dict[str, list[dict]]:
        cache_path = self._cache_path(question_id)
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        prompt = _build_criterion_prompt(question)
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=8000,
                )
                text = resp.choices[0].message.content or ""
                parsed = _parse_criterion(text)
                if parsed is not None:
                    if cache_path:
                        cache_path.write_text(
                            json.dumps(parsed, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    return parsed
                print(f"[AutoR] Q{question_id} parse failed, attempt {attempt + 1}")
            except Exception as e:
                print(f"[AutoR] Q{question_id} error attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)

        raise RuntimeError(f"AutoR criterion generation failed for Q{question_id}")


# ── Question type classification ───────────────────────────────────────────────

def _build_classify_prompt(question: str) -> str:
    return (
        "判断以下研究问题应使用哪种「指令遵循」评分策略：\n\n"
        "策略A（清单核查）：问题包含多个独立的、可逐条核查的子任务，\n"
        "各子任务有明确分界（例如"整合X，在此基础上预测Y，最后分析Z"）。\n"
        "核心判断：文档是否覆盖了各个子任务。\n\n"
        "策略B（质量判断）：问题主要要求对某主题进行综合研究或综述，\n"
        "没有明确的分步要求，文档的深度和专业性是关键区分因素。\n"
        "核心判断：文档对主题的把握深度和质量。\n\n"
        f"【研究问题】：\n{question}\n\n"
        '只输出 "A" 或 "B"，不要任何其他内容。'
    )


class QuestionTypeClassifier:
    """Classify a research question as Type A (checklist) or Type B (criteria)."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        cache_dir: Path | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, question_id: int) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"qtype_q{question_id}.json"

    def classify(self, question: str, question_id: int) -> str:
        """Return 'A' (checklist) or 'B' (criteria)."""
        cache_path = self._cache_path(question_id)
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))["type"]

        prompt = _build_classify_prompt(question)
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10,
                )
                raw = (resp.choices[0].message.content or "").strip().upper()
                result = "A" if raw.startswith("A") and "B" not in raw else "B"
                if cache_path:
                    cache_path.write_text(
                        json.dumps({"type": result}), encoding="utf-8"
                    )
                return result
            except Exception as e:
                print(f"[Classify] Q{question_id} attempt {attempt + 1}: {e}")
                if attempt < 2:
                    time.sleep(2**attempt)
        return "B"


# ── IF requirement checklist (Type A questions only) ───────────────────────────

def _build_checklist_prompt(question: str) -> str:
    return (
        "你是一位深度研究报告评审专家。请仔细阅读以下研究问题，将其拆解为具体的、可逐条核查的需求清单。\n\n"
        f"【研究问题】：\n{question}\n\n"
        "要求：\n"
        "1. 提取3~6条具体需求，每条需求应该是可以被"满足/未满足"的具体任务\n"
        "2. 每条需求要尽量独立，不要重叠\n"
        "3. 给每条需求分配权重（所有权重之和 = 1.0）\n"
        "4. 权重应反映该需求在整体问题中的重要性\n\n"
        "严格按以下 JSON 格式输出，不要有任何其他内容：\n"
        '{"checklist": [{"requirement": "具体需求描述（20字以内）", "weight": 0.3}, ...]}'
    )


def _parse_checklist(text: str) -> list[dict] | None:
    clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

    def validate(data: Any) -> list | None:
        if not isinstance(data, dict) or "checklist" not in data:
            return None
        items = data["checklist"]
        if not items or not isinstance(items, list) or len(items) < 2:
            return None
        total_w = sum(float(it.get("weight", 0)) for it in items)
        if total_w <= 0:
            return None
        for it in items:
            it["weight"] = round(float(it["weight"]) / total_w, 4)
        return items

    for candidate in [clean, text]:
        try:
            return validate(json.loads(candidate))
        except Exception:
            pass
        m = re.search(r"\{.*\}", candidate, re.DOTALL)
        if m:
            try:
                return validate(json.loads(m.group()))
            except Exception:
                pass
    return None


class IFChecklistGenerator:
    """Decompose a Type A question into a weighted requirement checklist for IF scoring."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        cache_dir: Path | None = None,
        max_retries: int = 3,
    ) -> None:
        self.client = client
        self.model = model
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, question_id: int) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"checklist_q{question_id}.json"

    def generate(self, question: str, question_id: int) -> list[dict]:
        cache_path = self._cache_path(question_id)
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        prompt = _build_checklist_prompt(question)
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1000,
                )
                text = resp.choices[0].message.content or ""
                checklist = _parse_checklist(text)
                if checklist is not None:
                    if cache_path:
                        cache_path.write_text(
                            json.dumps(checklist, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    return checklist
                print(f"[Checklist] Q{question_id} parse failed, attempt {attempt + 1}")
            except Exception as e:
                print(f"[Checklist] Q{question_id} error attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)

        return [{"requirement": "Address all specific requirements of the research question", "weight": 1.0}]
