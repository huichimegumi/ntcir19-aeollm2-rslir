"""Tokenizer-native context preflight; incompatible templates fail closed."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _count_chat_tokens(
    tokenizer: Any, messages: Sequence[dict[str, str]], *, chat_template_kwargs: dict | None = None,
) -> int:
    try:
        encoded = tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=True, return_dict=True,
            **(chat_template_kwargs or {}),
        )
        input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        if hasattr(input_ids, "shape"):
            return int(input_ids.shape[-1])
        if input_ids and isinstance(input_ids[0], (list, tuple)):
            return len(input_ids[0])
        return len(input_ids)
    except (AttributeError, TypeError, ValueError, KeyError) as exc:
        raise ValueError("cannot count the server chat template; refusing approximate token preflight") from exc


def validate_request_contexts(
    requests, *, tokenizer_name: str, context_length: int, max_tokens: int,
    tokenizer_revision: str | None = None, chat_template_kwargs: dict | None = None,
) -> list[dict]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("tokenizer preflight is configured but transformers is not installed; "
                           "install the package with `pip install -e '.[tokenizer]'`") from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=tokenizer_revision)
    over_limit, rows = [], []
    for request in requests:
        prompt_tokens = _count_chat_tokens(tokenizer, request.messages, chat_template_kwargs=chat_template_kwargs)
        total = prompt_tokens + max_tokens
        rows.append({
            "request_id": request.request_id, "prompt_tokens": prompt_tokens,
            "reserved_output_tokens": max_tokens, "total_budget_tokens": total,
            "context_length": context_length, "within_context": total <= context_length,
        })
        if total > context_length:
            over_limit.append(f"{request.request_id}: prompt={prompt_tokens}, reserved_output={max_tokens}, total={total}")
    if over_limit:
        raise ValueError(f"refusing silent truncation: {len(over_limit)} requests exceed context_length={context_length}\n"
                         + "\n".join(over_limit[:20]))
    return rows
