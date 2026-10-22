"""OpenAI-compatible judge client shared by local vLLM and remote APIs."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from .config import JudgeConfig


@dataclass(frozen=True)
class JudgeResponse:
    raw_response: str
    attempt_count: int
    latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None


class JudgeRequestError(RuntimeError):
    """Final judge failure with the last model output retained when available."""

    def __init__(
        self,
        message: str,
        *,
        attempt_count: int,
        raw_response: str | None = None,
    ) -> None:
        super().__init__(message)
        self.attempt_count = attempt_count
        self.raw_response = raw_response


class OpenAICompatibleJudge:
    def __init__(self, config: JudgeConfig):
        config.validate()
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=0,  # One shared HTTP/parse retry budget, owned by complete().
        )

    async def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        validator: Callable[[str], Any] | None = None,
    ) -> JudgeResponse:
        """Run one logical request, retrying provider errors and invalid model outputs."""
        last_error: Exception | None = None
        last_raw_response: str | None = None
        total_started = time.perf_counter()

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=list(messages),
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    extra_body=self.config.extra_body or None,
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("model returned an empty response")
                last_raw_response = content
                finish_reason = getattr(response.choices[0], "finish_reason", None)
                if finish_reason not in (None, "stop"):
                    raise ValueError(f"model did not finish normally: finish_reason={finish_reason}")
                if validator is not None:
                    validator(content)

                usage: Any = getattr(response, "usage", None)
                return JudgeResponse(
                    raw_response=content,
                    attempt_count=attempt,
                    latency_seconds=round(time.perf_counter() - total_started, 6),
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                )
            except Exception as exc:  # provider SDK and validation exceptions vary
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                await asyncio.sleep(min(2 ** (attempt - 1), 8))

        raise JudgeRequestError(
            f"judge request failed after {self.config.max_retries} attempts: {last_error}",
            attempt_count=self.config.max_retries,
            raw_response=last_raw_response,
        ) from last_error
