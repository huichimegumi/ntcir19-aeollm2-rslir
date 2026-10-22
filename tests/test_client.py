import asyncio
from types import SimpleNamespace

import pytest

import aeollm2_osodt.client as client_module
from aeollm2_osodt.client import JudgeRequestError, OpenAICompatibleJudge
from aeollm2_osodt.config import JudgeConfig


class _FakeCompletions:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        content = next(self.outputs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class _FakeAsyncOpenAI:
    outputs = []
    instance = None

    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=_FakeCompletions(list(type(self).outputs)))
        type(self).instance = self


def _config(max_retries=2):
    return JudgeConfig(
        name="fake",
        model="fake-model",
        base_url="http://fake/v1",
        api_key_env="FAKE_API_KEY",
        max_tokens=16,
        max_retries=max_retries,
        concurrency=1,
    )


def test_client_retries_when_validator_rejects_model_output(monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", "test")
    monkeypatch.setattr(client_module, "AsyncOpenAI", _FakeAsyncOpenAI)
    _FakeAsyncOpenAI.outputs = ["not-json", '{"score": 8}']

    judge = OpenAICompatibleJudge(_config(max_retries=2))

    def validator(raw):
        if not raw.startswith("{"):
            raise ValueError("invalid payload")

    response = asyncio.run(judge.complete([{"role": "user", "content": "x"}], validator=validator))
    assert response.raw_response == '{"score": 8}'
    assert response.attempt_count == 2
    assert _FakeAsyncOpenAI.instance.chat.completions.calls == 2


def test_client_preserves_last_raw_response_after_final_validation_failure(monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", "test")
    monkeypatch.setattr(client_module, "AsyncOpenAI", _FakeAsyncOpenAI)
    _FakeAsyncOpenAI.outputs = ["bad-1", "bad-2"]

    judge = OpenAICompatibleJudge(_config(max_retries=2))

    with pytest.raises(JudgeRequestError) as captured:
        asyncio.run(
            judge.complete(
                [{"role": "user", "content": "x"}],
                validator=lambda raw: (_ for _ in ()).throw(ValueError("invalid payload")),
            )
        )

    assert captured.value.attempt_count == 2
    assert captured.value.raw_response == "bad-2"
