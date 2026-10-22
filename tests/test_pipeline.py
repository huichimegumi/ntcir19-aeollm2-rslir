import json
import re
from pathlib import Path

import pandas as pd
import pytest
from docx import Document

import aeollm2_osodt.pipeline as pipeline
from aeollm2_osodt.client import JudgeResponse
from aeollm2_osodt.config import JudgeConfig
from aeollm2_osodt.constants import BEGIN_MARKER, DIMENSIONS, END_MARKER


class FakeJudge:
    calls = 0

    def __init__(self, config):
        self.config = config

    async def complete(self, messages, *, validator=None):
        type(self).calls += 1
        user = messages[-1]["content"]
        match = re.search(r"专门评估 `([^`]+)`", user)
        if match:
            payload = {match.group(1): [8.0]}
        else:
            payload = {dimension: [8.0] for dimension in DIMENSIONS}
        raw = f"{BEGIN_MARKER}\n{json.dumps(payload)}\n{END_MARKER}"
        if validator is not None:
            validator(raw)
        return JudgeResponse(
            raw_response=raw,
            attempt_count=1,
            latency_seconds=0.01,
            prompt_tokens=100,
            completion_tokens=10,
        )


def _dataset(tmp_path: Path):
    reports = tmp_path / "reports"
    report = reports / "Report1"
    report.mkdir(parents=True)
    document = Document()
    document.add_paragraph("A small report used by the CPU-only pipeline test.")
    document.save(report / "Doc_001.docx")

    rubrics = tmp_path / "rubrics"
    rubrics.mkdir()
    rubric = {
        "id": 1,
        "prompt": "Evaluate this report.",
        "dimension_weight": {dimension: 1.0 for dimension in DIMENSIONS},
        "criterions": {
            dimension: [
                {
                    "criterion": f"{dimension} criterion",
                    "explanation": "fixture",
                    "weight": 1.0,
                }
            ]
            for dimension in DIMENSIONS
        },
    }
    (rubrics / "criterion1.json").write_text(
        json.dumps(rubric, ensure_ascii=False), encoding="utf-8"
    )

    scales = tmp_path / "prompt_scales.json"
    scales.write_text(
        json.dumps(
            {
                "dim_scoring_scales": {
                    "comprehensiveness": "coverage scale",
                    "insight": "insight scale",
                    "instruction_following": "instruction scale",
                },
                "readability_rubrics": {"readability": "readability scale"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return reports, rubrics, scales


def _config(**overrides):
    values = dict(
        name="mock",
        model="mock-model",
        base_url="http://127.0.0.1:9999/v1",
        api_key_env="UNUSED_IN_FAKE",
        temperature=0.0,
        max_tokens=32,
        timeout_seconds=1.0,
        max_retries=1,
        concurrency=2,
        tokenizer=None,
        max_context_tokens=None,
        extra_body={},
    )
    values.update(overrides)
    return JudgeConfig(**values)


def test_mock_osodt_pipeline_writes_reproducible_artifacts_and_resumes(tmp_path, monkeypatch):
    reports, rubrics, scales = _dataset(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(pipeline, "OpenAICompatibleJudge", FakeJudge)
    FakeJudge.calls = 0

    pipeline.run_pipeline(
        method="osodt",
        config=_config(),
        report_root=reports,
        rubric_dir=rubrics,
        prompt_scales=scales,
        output_dir=output,
    )

    assert FakeJudge.calls == 8
    prediction = pd.read_csv(output / "predictions" / "osodt.tsv", sep="\t")
    assert prediction.shape[0] == 1
    assert all(prediction[f"{dimension}_score"].iloc[0] == 8.0 for dimension in DIMENSIONS)
    assert (output / "criterion_scores" / "original.jsonl").exists()
    assert (output / "criterion_scores" / "reverse.jsonl").exists()
    assert (output / "criterion_scores" / "symmetric.jsonl").exists()

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["inputs"]["document_count"] == 1
    assert manifest["inputs"]["request_count"] == 8
    assert len(manifest["inputs"]["reports_sha256"]) == 64
    assert len(manifest["inputs"]["request_prompts_sha256"]) == 64

    pipeline.run_pipeline(
        method="osodt",
        config=_config(),
        report_root=reports,
        rubric_dir=rubrics,
        prompt_scales=scales,
        output_dir=output,
    )
    assert FakeJudge.calls == 8, "completed compatible requests should be resumed without re-inference"


def test_preload_reuses_only_verified_responses_and_rejects_tampering(tmp_path, monkeypatch):
    reports, rubrics, scales = _dataset(tmp_path)
    monkeypatch.setattr(pipeline, "OpenAICompatibleJudge", FakeJudge)
    FakeJudge.calls = 0
    kwargs = dict(method="osodt", config=_config(), report_root=reports,
                  rubric_dir=rubrics, prompt_scales=scales)
    pipeline.run_pipeline(**kwargs, output_dir=tmp_path / "source")
    assert FakeJudge.calls == 8

    records = [json.loads(line) for line in (tmp_path / "source/responses.jsonl").read_text().splitlines()]
    seed = tmp_path / "seed.jsonl"
    seed.write_text(json.dumps(records[0]) + "\n")
    pipeline.run_pipeline(**kwargs, output_dir=tmp_path / "reused", preload_responses_from=seed)
    assert FakeJudge.calls == 15  # One of eight real requests came from the verified seed.
    assert json.loads((tmp_path / "reused/manifest.json").read_text())["preloaded_responses"]["sha256"]
    pipeline.run_pipeline(**kwargs, output_dir=tmp_path / "reused", preload_responses_from=seed)
    assert FakeJudge.calls == 15

    records[0]["scores"][records[0]["dimension"]][0] = 0.125
    bad_seed = tmp_path / "bad-seed.jsonl"
    bad_seed.write_text(json.dumps(records[0]) + "\n")
    with pytest.raises(ValueError, match="preloaded scores differ from raw response"):
        pipeline.run_pipeline(**kwargs, output_dir=tmp_path / "bad", preload_responses_from=bad_seed)
    assert FakeJudge.calls == 15


def test_pipeline_rejects_input_drift_in_existing_output_directory(tmp_path, monkeypatch):
    reports, rubrics, scales = _dataset(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(pipeline, "OpenAICompatibleJudge", FakeJudge)

    pipeline.run_pipeline(
        method="odt",
        config=_config(),
        report_root=reports,
        rubric_dir=rubrics,
        prompt_scales=scales,
        output_dir=output,
    )

    rubric_path = rubrics / "criterion1.json"
    payload = json.loads(rubric_path.read_text(encoding="utf-8"))
    payload["criterions"]["insight"][0]["explanation"] = "changed fixture"
    rubric_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest differs"):
        pipeline.run_pipeline(
            method="odt",
            config=_config(),
            report_root=reports,
            rubric_dir=rubrics,
            prompt_scales=scales,
            output_dir=output,
        )


def test_pipeline_records_context_preflight_diagnostics(tmp_path, monkeypatch):
    reports, rubrics, scales = _dataset(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(pipeline, "OpenAICompatibleJudge", FakeJudge)

    def fake_preflight(requests, *, tokenizer_name, context_length, max_tokens):
        assert tokenizer_name == "fixture-tokenizer"
        assert context_length == 4096
        return [
            {
                "request_id": request.request_id,
                "prompt_tokens": 100,
                "reserved_output_tokens": max_tokens,
                "total_budget_tokens": 100 + max_tokens,
                "context_length": context_length,
                "within_context": True,
            }
            for request in requests
        ]

    monkeypatch.setattr(pipeline, "validate_request_contexts", fake_preflight)
    pipeline.run_pipeline(
        method="joint",
        config=_config(tokenizer="fixture-tokenizer", max_context_tokens=4096),
        report_root=reports,
        rubric_dir=rubrics,
        prompt_scales=scales,
        output_dir=output,
    )

    diagnostics = json.loads(
        (output / "diagnostics" / "context_preflight.json").read_text(encoding="utf-8")
    )
    assert diagnostics["tokenizer"] == "fixture-tokenizer"
    assert diagnostics["requests"][0]["within_context"] is True
