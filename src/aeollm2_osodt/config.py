"""Runtime configuration for OpenAI-compatible judge backends."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class JudgeConfig:
    name: str
    model: str
    base_url: str
    api_key_env: str
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: float = 600.0
    max_retries: int = 3
    concurrency: int = 4
    tokenizer: str | None = None
    max_context_tokens: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    tokenizer_revision: str | None = None
    prompt_protocol: str = "paper-v1"

    @property
    def api_key(self) -> str:
        value = os.getenv(self.api_key_env)
        if not value:
            raise ValueError(f"environment variable {self.api_key_env!r} is not set")
        return value

    def validate(self) -> None:
        if self.prompt_protocol not in {"paper-v1", "e9-schema-v1"}:
            raise ValueError("prompt_protocol must be paper-v1 or e9-schema-v1")
        if not self.name.strip() or not self.model.strip() or not self.base_url.strip():
            raise ValueError("name, model, and base_url must be non-empty")
        if not self.api_key_env.strip():
            raise ValueError("api_key_env must be non-empty")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if min(self.max_tokens, self.max_retries, self.concurrency) <= 0:
            raise ValueError("max_tokens, max_retries, and concurrency must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if (self.tokenizer is None) != (self.max_context_tokens is None):
            raise ValueError(
                "tokenizer and max_context_tokens must either both be configured or both be omitted"
            )
        if self.max_context_tokens is not None and self.max_tokens >= self.max_context_tokens:
            raise ValueError("max_tokens must be smaller than max_context_tokens")


def load_judge_config(path: str | Path) -> JudgeConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")

    model = payload.get("model") or {}
    generation = payload.get("generation") or {}
    context = payload.get("context") or {}
    config = JudgeConfig(
        name=str(payload.get("name", "")).strip(),
        model=str(model.get("id", "")).strip(),
        base_url=str(model.get("base_url", "")).strip(),
        api_key_env=str(model.get("api_key_env", "")).strip(),
        temperature=float(generation.get("temperature", 0.0)),
        max_tokens=int(generation.get("max_tokens", 2048)),
        timeout_seconds=float(generation.get("timeout_seconds", 600.0)),
        max_retries=int(generation.get("max_retries", 3)),
        concurrency=int(generation.get("concurrency", 4)),
        tokenizer=(str(context.get("tokenizer")).strip() if context.get("tokenizer") else None),
        max_context_tokens=(
            int(context["max_context_tokens"])
            if context.get("max_context_tokens") is not None
            else None
        ),
        extra_body=dict(payload.get("extra_body") or {}),
        tokenizer_revision=context.get("tokenizer_revision"),
        prompt_protocol=str(payload.get("prompt_protocol", "paper-v1")),
    )
    config.validate()
    return config
