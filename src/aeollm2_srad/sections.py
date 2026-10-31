"""Structural section splitting used by the final SRAD implementation."""

from __future__ import annotations

import re

MIN_SECTION_CHARS = 500

MAX_RULE_SECTIONS = 12

MIN_RULE_SECTIONS = 2

FALLBACK_SECTION_COUNT = 4

URL_STREAK_FOR_REFERENCES = 3

MAX_SECTION_CHAR_RATIO = 0.70

REFERENCE_HEADING_PATTERNS = [
    r"^\s*(key\s+)?citations?\s*$",
    r"^\s*(selected\s+)?references?\s*$",
    r"^\s*bibliography\s*$",
    r"^\s*(data\s+)?sources?\s*$",
    r"^\s*works\s+cited\s*$",
    r"^\s*参考文献\s*$",
    r"^\s*参考资料\s*$",
    r"^\s*文献来源\s*$",
    r"^\s*资料来源\s*$",
    r"^\s*引用来源\s*$",
    r"^\s*关键引用\s*$",
    r"^\s*主要引用\s*$",
    r"^\s*关键文献\s*$",
    r"^\s*主要文献\s*$",
]

def normalize_para(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())

def _looks_like_reference_heading(para: str) -> bool:
    text = normalize_para(para)
    normalized = re.sub(r"^[#*\-\s\d.、:：]+", "", text).strip()
    if any(re.search(pattern, normalized, flags=re.I) for pattern in REFERENCE_HEADING_PATTERNS):
        return True
    return False

def _looks_like_url_reference_para(para: str) -> bool:
    text = normalize_para(para)
    if not text:
        return False
    urls = re.findall(r"https?://\S+|www\.\S+", text, flags=re.I)
    if not urls:
        return False
    url_chars = sum(len(u) for u in urls)
    if len(urls) >= 2:
        return True
    if len(text) <= 220:
        return True
    return (url_chars / max(len(text), 1)) >= 0.25

def trim_reference_paragraphs(method, paragraphs: list[str]) -> tuple[list[str], int]:
    """Drop trailing references using the existing reference detectors."""
    if not paragraphs:
        return paragraphs, 0

    for idx, para in enumerate(paragraphs):
        if method._looks_like_reference_heading(para) or _looks_like_reference_heading(para):
            return paragraphs[:idx], len(paragraphs) - idx

    start = max(0, len(paragraphs) // 2)
    streak = 0
    streak_start = None
    for idx in range(start, len(paragraphs)):
        if method._looks_like_url_reference_block(paragraphs[idx]) or _looks_like_url_reference_para(paragraphs[idx]):
            if streak == 0:
                streak_start = idx
            streak += 1
            if streak >= URL_STREAK_FOR_REFERENCES and streak_start is not None:
                return paragraphs[:streak_start], len(paragraphs) - streak_start
        else:
            streak = 0
            streak_start = None

    return paragraphs, 0

def heading_candidates(paragraphs: list[str]) -> dict[str, list[int]]:
    patterns = {
        "chapter": [
            r"^第[一二三四五六七八九十百千万\d]+[章节部分篇]\s*[:：、.\-]?\s*\S+",
        ],
        "decimal_top_number": [
            r"^\d{1,2}\.0\s+\S+",
            r"^\d{1,2}\.\s+\S+",
        ],
        "chinese_number": [
            r"^[一二三四五六七八九十]+[、.．]\s*\S+",
        ],
        "arabic_number": [
            r"^\d{1,2}[、.．]\s+(?!\d)\S+",
            r"^\d{1,2}\s+[A-Z][A-Za-z].*",
        ],
        "roman_number": [
            r"^[IVXLCDM]{1,6}[、.．]\s+\S+",
        ],
        "common_heading": [
            r"^(摘要|引言|绪论|背景|研究背景|方法|研究方法|分析|结论|结语|建议|总结|附录)\s*$",
            r"^(Abstract|Introduction|Background|Methods?|Analysis|Results?|Discussion|Conclusion|Appendix)\s*$",
        ],
    }

    out: dict[str, list[int]] = {name: [] for name in patterns}
    for idx, para in enumerate(paragraphs):
        text = normalize_para(para)
        if not text:
            continue
        if len(text) > 90:
            continue
        if re.search(r"[。！？.!?]\s*$", text) and len(text) > 25:
            continue
        for name, pats in patterns.items():
            if any(re.search(pat, text, flags=re.I) for pat in pats):
                out[name].append(idx)
                break
    return out

def section_lengths(paragraphs: list[str], starts: list[int]) -> list[int]:
    lengths: list[int] = []
    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(paragraphs)
        text = "\n".join(paragraphs[start:end])
        lengths.append(len(text))
    return lengths

def _chinese_number_to_int(text: str) -> int | None:
    digits = {
        "零": 0,
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    text = str(text).strip()
    if not text:
        return None
    if text == "十":
        return 10
    if text.startswith("十"):
        tail = text[1:]
        return 10 + digits.get(tail, 0)
    if "十" in text:
        head, tail = text.split("十", 1)
        return digits.get(head, 0) * 10 + digits.get(tail, 0)
    return digits.get(text)

def _roman_to_int(text: str) -> int | None:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev = 0
    for ch in reversed(str(text).upper()):
        val = values.get(ch)
        if val is None:
            return None
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return total if total > 0 else None

def heading_number_sequence(kind: str, paragraphs: list[str], starts: list[int]) -> list[int]:
    out: list[int] = []
    for start in starts:
        text = normalize_para(paragraphs[start])
        num = None
        if kind == "chapter":
            m = re.search(r"^第([一二三四五六七八九十百千万\d]+)[章节部分篇]", text)
            if m:
                token = m.group(1)
                num = int(token) if token.isdigit() else _chinese_number_to_int(token)
        elif kind in {"decimal_top_number", "arabic_number"}:
            m = re.search(r"^(\d{1,2})(?:\.0|[、.．]|\s+)", text)
            if m:
                num = int(m.group(1))
        elif kind == "chinese_number":
            m = re.search(r"^([一二三四五六七八九十]+)[、.．]", text)
            if m:
                num = _chinese_number_to_int(m.group(1))
        elif kind == "roman_number":
            m = re.search(r"^([IVXLCDM]{1,6})[、.．]", text, flags=re.I)
            if m:
                num = _roman_to_int(m.group(1))
        if num is not None:
            out.append(num)
    return out

def numbered_headings_are_coherent(kind: str, paragraphs: list[str], starts: list[int]) -> bool:
    if kind == "common_heading":
        return True
    nums = heading_number_sequence(kind, paragraphs, starts)
    if len(nums) != len(starts):
        return True
    if len(nums) < 2:
        return True
    if nums != sorted(nums):
        return False
    gaps = [b - a for a, b in zip(nums, nums[1:])]
    return all(gap == 1 for gap in gaps)

def section_lengths_are_balanced(lengths: list[int]) -> bool:
    total = sum(lengths)
    if total <= 0:
        return False
    return (max(lengths) / total) <= MAX_SECTION_CHAR_RATIO

def choose_heading_starts(paragraphs: list[str]) -> tuple[str, list[int]]:
    candidates = heading_candidates(paragraphs)
    priority = ["chapter", "decimal_top_number", "chinese_number", "arabic_number", "roman_number", "common_heading"]

    for kind in priority:
        starts = candidates.get(kind, [])
        if not (MIN_RULE_SECTIONS <= len(starts) <= MAX_RULE_SECTIONS):
            continue
        if kind == "common_heading" and len(starts) < 3:
            continue
        lengths = section_lengths(paragraphs, starts)
        if len(starts) >= 3 and not section_lengths_are_balanced(lengths):
            continue
        if not numbered_headings_are_coherent(kind, paragraphs, starts):
            continue
        valid_lengths = [n for n in lengths if n >= MIN_SECTION_CHARS]
        if len(valid_lengths) >= max(1, int(len(lengths) * 0.6)):
            return kind, starts

    return "", []

def split_by_headings(paragraphs: list[str], starts: list[int]) -> list[dict]:
    sections: list[dict] = []
    preface = paragraphs[: starts[0]] if starts and starts[0] > 0 else []

    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(paragraphs)
        section_paras = paragraphs[start:end]
        display_start = start + 1
        if pos == 0 and preface:
            preface_len = len("\n".join(preface))
            if preface_len < MIN_SECTION_CHARS:
                section_paras = preface + section_paras
                display_start = 1
            else:
                sections.append(
                    {
                        "title": "Front Matter",
                        "start_para": 1,
                        "end_para": start,
                        "text": "\n".join(preface).strip(),
                    }
                )
        title = normalize_para(paragraphs[start])
        sections.append(
            {
                "title": title[:120],
                "start_para": display_start,
                "end_para": end,
                "text": "\n".join(section_paras).strip(),
            }
        )

    return [s for s in sections if s["text"]]

def split_by_length(paragraphs: list[str], target_sections: int = FALLBACK_SECTION_COUNT) -> list[dict]:
    total_chars = sum(len(p) for p in paragraphs)
    if total_chars <= 0:
        return []
    target = max(1, total_chars / max(1, target_sections))

    sections: list[dict] = []
    current: list[str] = []
    current_start = 0
    current_len = 0

    for idx, para in enumerate(paragraphs):
        if not current:
            current_start = idx
        current.append(para)
        current_len += len(para)
        remaining_paras = len(paragraphs) - idx - 1
        remaining_sections = target_sections - len(sections) - 1
        if current_len >= target and remaining_sections > 0 and remaining_paras > 0:
            sections.append(
                {
                    "title": f"Length Section {len(sections) + 1}",
                    "start_para": current_start + 1,
                    "end_para": idx + 1,
                    "text": "\n".join(current).strip(),
                }
            )
            current = []
            current_len = 0

    if current:
        sections.append(
            {
                "title": f"Length Section {len(sections) + 1}",
                "start_para": current_start + 1,
                "end_para": len(paragraphs),
                "text": "\n".join(current).strip(),
            }
        )

    return sections

def build_readability_sections(method, paragraphs: list[str]) -> tuple[list[dict], str, int]:
    normalized = [normalize_para(p) for p in paragraphs if normalize_para(p)]
    trimmed, dropped_refs = trim_reference_paragraphs(method, normalized)
    if not trimmed:
        return [], "empty", dropped_refs

    kind, starts = choose_heading_starts(trimmed)
    if starts:
        return split_by_headings(trimmed, starts), f"heading:{kind}", dropped_refs

    return split_by_length(trimmed, FALLBACK_SECTION_COUNT), "fallback:length", dropped_refs
