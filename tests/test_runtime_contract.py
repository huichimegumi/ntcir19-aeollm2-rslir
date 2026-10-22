"""HTTP, tokenization, and full multi-document pipeline contracts."""
import asyncio
import json
import sys
from collections import UserDict
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI

import aeollm2_osodt.client as client_module
from aeollm2_osodt.client import JudgeRequestError, OpenAICompatibleJudge
from aeollm2_osodt.config import JudgeConfig
from aeollm2_osodt.parsing import parse_score_object
from aeollm2_osodt.preflight import _count_chat_tokens, validate_request_contexts


def config(attempts=3):
    return JudgeConfig(name='contract', model='deepseek-v4-flash', base_url='https://fake.example/v1',
                       api_key_env='CONTRACT_KEY', max_retries=attempts, max_tokens=2048,
                       extra_body={'thinking': {'type': 'disabled'}})


def completion(content, reason='stop'):
    return {'id': 'fixture', 'object': 'chat.completion', 'created': 0, 'model': 'deepseek-v4-flash',
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': reason}],
            'usage': {'prompt_tokens': 12, 'completion_tokens': 4, 'total_tokens': 16}}


@pytest.mark.parametrize('failure', ['429', 'timeout', 'empty', 'invalid', 'length', 'content_filter'])
def test_http_contract_and_shared_retry_budget(monkeypatch, failure):
    requests = []
    valid = '{"insight":[7.5,8]}'
    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            if failure == '429': return httpx.Response(429, json={'error': {'message': 'retry', 'type': 'rate_limit'}})
            if failure == 'timeout': raise httpx.ReadTimeout('fixture timeout', request=request)
            return httpx.Response(200, json=completion(
                '' if failure == 'empty' else 'bad' if failure == 'invalid' else valid,
                failure if failure in {'length', 'content_filter'} else 'stop'))
        return httpx.Response(200, json=completion(valid))
    def factory(**kwargs):
        assert kwargs['max_retries'] == 0
        return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async def no_sleep(_): pass
    monkeypatch.setattr(client_module, 'AsyncOpenAI', factory)
    monkeypatch.setattr(client_module.asyncio, 'sleep', no_sleep)
    monkeypatch.setenv('CONTRACT_KEY', 'fixture-only')
    messages = [{'role': 'user', 'content': 'fixture'}]
    async def run():
        judge = OpenAICompatibleJudge(config())
        try: return await judge.complete(messages, validator=lambda raw: parse_score_object(raw, {'insight': 2}))
        finally: await judge.client.close()
    result = asyncio.run(run())
    assert result.attempt_count == len(requests) == 2
    assert result.raw_response == valid
    assert result.prompt_tokens == 12
    for body in requests:
        assert body == {'model': 'deepseek-v4-flash', 'messages': messages, 'temperature': 0.0,
                        'max_tokens': 2048, 'thinking': {'type': 'disabled'}}


def test_mixed_http_and_parse_failures_do_not_multiply_sdk_retries(monkeypatch):
    attempts = []
    def handler(request):
        attempts.append(1)
        if len(attempts) == 1: return httpx.Response(429, json={'error': {'message': 'retry'}})
        if len(attempts) == 2: return httpx.Response(200, json=completion('bad-json'))
        raise httpx.ReadTimeout('timeout', request=request)
    monkeypatch.setenv('CONTRACT_KEY', 'fixture-only')
    monkeypatch.setattr(client_module, 'AsyncOpenAI', lambda **kw: AsyncOpenAI(
        **kw, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))))
    async def no_sleep(_): pass
    monkeypatch.setattr(client_module.asyncio, 'sleep', no_sleep)
    async def run():
        judge = OpenAICompatibleJudge(config())
        try:
            with pytest.raises(JudgeRequestError) as caught:
                await judge.complete([], validator=lambda raw: parse_score_object(raw, {'insight': 2}))
            assert caught.value.attempt_count == 3
            assert caught.value.raw_response == 'bad-json'
        finally: await judge.client.close()
    asyncio.run(run())
    assert len(attempts) == 3


def test_context_uses_native_template_revision_and_thinking_settings(monkeypatch):
    calls = []
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(kwargs)
            return UserDict({'input_ids': [1] * 8})
    def load(name, **kwargs):
        assert (name, kwargs) == ('fixture', {'revision': 'fixed-revision'})
        return Tokenizer()
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load)))
    request = SimpleNamespace(request_id='q1:doc:insight', messages=[{'role': 'user', 'content': 'x'}])
    kwargs = dict(tokenizer_name='fixture', tokenizer_revision='fixed-revision', max_tokens=2,
                  chat_template_kwargs={'enable_thinking': False})
    assert validate_request_contexts([request], context_length=10, **kwargs)[0]['within_context']
    assert calls[0]['enable_thinking'] is False
    with pytest.raises(ValueError, match='refusing silent truncation'):
        validate_request_contexts([request], context_length=9, **kwargs)


def test_context_incompatible_template_fails_instead_of_approximating():
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs): raise ValueError('no template')
        def encode(self, *args, **kwargs): pytest.fail('approximate fallback must not run')
    with pytest.raises(ValueError, match='refusing approximate'):
        _count_chat_tokens(Tokenizer(), [])


def test_context_missing_dependency_is_explicit(monkeypatch):
    monkeypatch.setitem(sys.modules, 'transformers', None)
    with pytest.raises(RuntimeError, match='transformers is not installed'):
        validate_request_contexts([], tokenizer_name='unused', context_length=10, max_tokens=2)
