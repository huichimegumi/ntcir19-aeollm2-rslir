"""Strict response parsing for paper-v1 judge outputs."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from .constants import BEGIN_MARKER, END_MARKER


def _extract_payload_text(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("empty model response")

    marker_pattern = re.escape(BEGIN_MARKER) + r"\s*(.*?)\s*" + re.escape(END_MARKER)
    marker_match = re.search(marker_pattern, raw, flags=re.S)
    if marker_match:
        return marker_match.group(1).strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.S | re.I)
    if fence_match:
        return fence_match.group(1).strip()

    stripped = raw.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    raise ValueError("response is not marker-delimited JSON, fenced JSON, or a JSON object")


def parse_score_object(
    raw: str,
    expected_counts: Mapping[str, int],
    *,
    allow_comments: bool = False,
) -> dict[str, list[float]]:
    """Parse exactly the requested dimension arrays and validate all scores."""
    payload_text = _extract_payload_text(raw)
    if allow_comments:
        # The E9 parser accepted JSON with // comments. Preserve string contents
        # and all structural validation; never infer missing or malformed scores.
        clean = []
        in_string = escaped = False
        index = 0
        while index < len(payload_text):
            char = payload_text[index]
            if not in_string and payload_text[index:index + 2] == "//":
                newline = payload_text.find("\n", index)
                index = len(payload_text) if newline < 0 else newline
                continue
            clean.append(char)
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            index += 1
        payload_text = "".join(clean)
    try:
        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result
        payload = json.loads(payload_text, object_pairs_hook=unique_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON response: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("judge response must be a JSON object")
    if set(payload) != set(expected_counts):
        raise ValueError(
            f"response keys must be exactly {sorted(expected_counts)}, got {sorted(payload)}"
        )

    result: dict[str, list[float]] = {}
    for dimension, expected_count in expected_counts.items():
        values = payload.get(dimension)
        if not isinstance(values, list):
            raise ValueError(f"{dimension}: expected a score array")
        if len(values) != expected_count:
            raise ValueError(
                f"{dimension}: expected {expected_count} scores, got {len(values)}"
            )
        normalized: list[float] = []
        for index, value in enumerate(values, 1):
            if isinstance(value, bool):
                raise ValueError(f"{dimension}[{index}]: boolean is not a score")
            try:
                score = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{dimension}[{index}]: non-numeric score {value!r}") from exc
            if not 0.0 <= score <= 10.0:
                raise ValueError(f"{dimension}[{index}]: score outside [0, 10]: {score}")
            normalized.append(score)
        result[dimension] = normalized
    return result
