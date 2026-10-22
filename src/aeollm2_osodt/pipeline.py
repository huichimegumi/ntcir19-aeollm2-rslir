"""Unified execution pipeline for Joint, ODT, and OSODT."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import (
    atomic_write_json,
    guard_manifest,
    load_jsonl_by_request_id,
    sha256_collection,
    sha256_file,
    sha256_messages,
    sha256_request_prompts,
    upsert_jsonl_record,
)
from .client import OpenAICompatibleJudge
from .config import JudgeConfig
from .constants import DIMENSIONS, READER_REVISION, SCHEMA_VERSION
from .dataset import discover_documents
from .methods.odt import CriterionScore, DimensionScore, map_prompt_scores_to_canonical
from .methods.osodt import (
    SymmetricCriterionScore,
    SymmetricDimensionScore,
    average_order_records,
)
from .parsing import parse_score_object
from .preflight import validate_request_contexts
from .prompting import build_joint_messages, build_odt_messages, load_prompt_scales
from .rubric import Rubric, load_rubric


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    method: str
    question_id: int
    answer_id: str
    dimension: str | None
    criterion_order: str | None
    messages: tuple[dict[str, str], ...]
    expected_counts: dict[str, int]
    prompt_sha256: str


def _load_rubrics(rubric_dir: Path, question_ids: set[int]) -> dict[int, Rubric]:
    return {question_id: load_rubric(rubric_dir, question_id) for question_id in sorted(question_ids)}


def _rubric_files(rubric_dir: Path, question_ids: set[int]) -> list[Path]:
    paths: list[Path] = []
    for question_id in sorted(question_ids):
        candidates = (
            rubric_dir / f"criterion{question_id}.json",
            rubric_dir / f"criteria{question_id}.json",
        )
        match = next((candidate for candidate in candidates if candidate.is_file()), None)
        if match is None:
            raise FileNotFoundError(f"rubric not found for q{question_id}: {candidates}")
        paths.append(match)
    return paths


def _request(
    *,
    request_id: str,
    method: str,
    question_id: int,
    answer_id: str,
    dimension: str | None,
    criterion_order: str | None,
    messages: tuple[dict[str, str], ...],
    expected_counts: dict[str, int],
) -> RequestSpec:
    return RequestSpec(
        request_id=request_id,
        method=method,
        question_id=question_id,
        answer_id=answer_id,
        dimension=dimension,
        criterion_order=criterion_order,
        messages=messages,
        expected_counts=expected_counts,
        prompt_sha256=sha256_messages(messages),
    )


def build_requests(
    method: str,
    documents,
    rubrics: dict[int, Rubric],
    scales: dict[str, str],
    *,
    odt_order: str = "original",
    prompt_protocol: str = "paper-v1",
) -> list[RequestSpec]:
    if method not in {"joint", "odt", "osodt"}:
        raise ValueError("method must be joint, odt, or osodt")
    if odt_order not in {"original", "reverse"}:
        raise ValueError("odt_order must be original or reverse")

    requests: list[RequestSpec] = []
    for document in documents:
        rubric = rubrics[document.question_id]
        if method == "joint":
            messages = build_joint_messages(document.text, rubric, scales, prompt_protocol=prompt_protocol)
            requests.append(
                _request(
                    request_id=f"joint:q{document.question_id}:{document.answer_id}",
                    method="joint",
                    question_id=document.question_id,
                    answer_id=document.answer_id,
                    dimension=None,
                    criterion_order=None,
                    messages=messages,
                    expected_counts={d: len(rubric.criteria[d]) for d in DIMENSIONS},
                )
            )
            continue

        orders = (odt_order,) if method == "odt" else ("original", "reverse")
        for order in orders:
            for dimension in DIMENSIONS:
                messages = build_odt_messages(
                    document.text,
                    rubric,
                    scales,
                    dimension,
                    criterion_order=order,
                    prompt_protocol=prompt_protocol,
                )
                requests.append(
                    _request(
                        request_id=(
                            f"odt:{order}:q{document.question_id}:"
                            f"{document.answer_id}:{dimension}"
                        ),
                        method="odt",
                        question_id=document.question_id,
                        answer_id=document.answer_id,
                        dimension=dimension,
                        criterion_order=order,
                        messages=messages,
                        expected_counts={dimension: len(rubric.criteria[dimension])},
                    )
                )

    ids = [request.request_id for request in requests]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate request IDs")
    return requests


def _resume_compatible(record: dict[str, Any], request: RequestSpec, config: JudgeConfig) -> bool:
    return (
        record.get("parse_valid") is True
        and record.get("model") == config.model
        and record.get("prompt_revision") == config.prompt_protocol
        and record.get("prompt_sha256") == request.prompt_sha256
        and record.get("method") == request.method
        and record.get("criterion_order") == request.criterion_order
        and int(record.get("question_id", -1)) == request.question_id
        and str(record.get("answer_id", "")) == request.answer_id
        and record.get("dimension") == request.dimension
    )


async def _execute_one(
    request: RequestSpec,
    judge: OpenAICompatibleJudge,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        try:
            validator = lambda raw: parse_score_object(
                raw, request.expected_counts, allow_comments=judge.config.prompt_protocol == "e9-schema-v1"
            )
            response = await judge.complete(request.messages, validator=validator)
            scores = validator(response.raw_response)
            return {
                "schema_version": SCHEMA_VERSION,
                "request_id": request.request_id,
                "method": request.method,
                "criterion_order": request.criterion_order,
                "question_id": request.question_id,
                "answer_id": request.answer_id,
                "dimension": request.dimension,
                "model": judge.config.model,
                "prompt_revision": judge.config.prompt_protocol,
                "prompt_sha256": request.prompt_sha256,
                "parse_valid": True,
                "attempt_count": response.attempt_count,
                "raw_response": response.raw_response,
                "scores": scores,
                "usage": {
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                },
                "latency_seconds": response.latency_seconds,
            }
        except Exception as exc:
            return {
                "schema_version": SCHEMA_VERSION,
                "request_id": request.request_id,
                "method": request.method,
                "criterion_order": request.criterion_order,
                "question_id": request.question_id,
                "answer_id": request.answer_id,
                "dimension": request.dimension,
                "model": judge.config.model,
                "prompt_revision": judge.config.prompt_protocol,
                "prompt_sha256": request.prompt_sha256,
                "parse_valid": False,
                "attempt_count": getattr(exc, "attempt_count", None),
                "raw_response": getattr(exc, "raw_response", None),
                "error": str(exc),
            }


async def execute_requests(
    requests: list[RequestSpec],
    config: JudgeConfig,
    responses_path: Path,
) -> dict[str, dict[str, Any]]:
    existing = load_jsonl_by_request_id(responses_path)
    request_ids = {request.request_id for request in requests}
    unexpected = sorted(set(existing) - request_ids)
    if unexpected:
        raise ValueError(
            "responses.jsonl contains request IDs not present in this run: "
            + ", ".join(unexpected[:10])
        )

    pending: list[RequestSpec] = []
    for request in requests:
        record = existing.get(request.request_id)
        if record is None:
            pending.append(request)
            continue
        if record.get("parse_valid") and not _resume_compatible(record, request, config):
            raise ValueError(
                f"completed response is incompatible with the current request: {request.request_id}"
            )
        if _resume_compatible(record, request, config):
            try:
                parsed = parse_score_object(record["raw_response"], request.expected_counts,
                                            allow_comments=config.prompt_protocol == "e9-schema-v1")
            except (KeyError, ValueError) as exc:
                raise ValueError(f"completed response has invalid raw response: {request.request_id}") from exc
            if parsed != record.get("scores"):
                raise ValueError(f"stored scores differ from raw response: {request.request_id}")
        if not _resume_compatible(record, request, config):
            pending.append(request)

    if not pending:
        return existing

    judge = OpenAICompatibleJudge(config)
    semaphore = asyncio.Semaphore(config.concurrency)
    tasks = [asyncio.create_task(_execute_one(request, judge, semaphore)) for request in pending]
    for task in asyncio.as_completed(tasks):
        record = await task
        upsert_jsonl_record(responses_path, record)
    return load_jsonl_by_request_id(responses_path)


def _dimension_record(
    response: dict[str, Any], rubric: Rubric, dimension: str, order: str
) -> DimensionScore:
    if not response.get("parse_valid"):
        raise ValueError(f"invalid response: {response.get('request_id')}")
    return map_prompt_scores_to_canonical(
        rubric,
        dimension,
        response["scores"][dimension],
        criterion_order=order,
    )


def _criterion_payload(item: CriterionScore) -> dict[str, Any]:
    return {
        "criterionId": item.criterion_id,
        "originalIndex": item.original_index,
        "promptPosition": item.prompt_position,
        "criterion": item.criterion,
        "weight": item.weight,
        "score": item.score,
    }


def _dimension_payload(score: DimensionScore) -> dict[str, Any]:
    return {
        "criterionOrder": score.criterion_order,
        "criteria": [_criterion_payload(item) for item in score.criteria],
        "dim_score": score.dim_score,
    }


def _symmetric_criterion_payload(item: SymmetricCriterionScore) -> dict[str, Any]:
    return {
        "criterionId": item.criterion_id,
        "originalIndex": item.original_index,
        "criterion": item.criterion,
        "weight": item.weight,
        "originalScore": item.original_score,
        "reverseScore": item.reverse_score,
        "score": item.score,
    }


def _symmetric_dimension_payload(score: SymmetricDimensionScore) -> dict[str, Any]:
    return {
        "criterionOrder": "symmetric",
        "criteria": [_symmetric_criterion_payload(item) for item in score.criteria],
        "original_dim_score": score.original_dim_score,
        "reverse_dim_score": score.reverse_dim_score,
        "dim_score": score.dim_score,
    }


def _write_prediction_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["questionId", "answerId"])
    columns = ["taskId", "questionId", "answerId", *[f"{d}_score" for d in DIMENSIONS]]
    frame[columns].to_csv(path, sep="\t", index=False)


def _write_criterion_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda r: (r["questionId"], r["answerId"])):
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _judge_model(responses: dict[str, dict[str, Any]]) -> str | None:
    models = {record.get("model") for record in responses.values() if record.get("model")}
    if len(models) > 1:
        raise ValueError(f"multiple judge models found in one run: {sorted(models)}")
    return next(iter(models), None)


def compile_artifacts(
    method: str,
    documents,
    rubrics: dict[int, Rubric],
    responses: dict[str, dict[str, Any]],
    output_dir: Path,
    *,
    odt_order: str = "original",
) -> None:
    if any(not record.get("parse_valid") for record in responses.values()):
        failed = [key for key, value in responses.items() if not value.get("parse_valid")]
        raise ValueError(f"cannot compile artifacts while requests are invalid: {failed[:10]}")
    judge_model = _judge_model(responses)

    if method == "joint":
        rows: list[dict[str, Any]] = []
        details: list[dict[str, Any]] = []
        for document in documents:
            rubric = rubrics[document.question_id]
            response = responses[f"joint:q{document.question_id}:{document.answer_id}"]
            dimensions: dict[str, Any] = {}
            row: dict[str, Any] = {
                "taskId": 0,
                "questionId": document.question_id,
                "answerId": document.answer_id,
            }
            for dimension in DIMENSIONS:
                score = _dimension_record(response, rubric, dimension, "original")
                dimensions[dimension] = _dimension_payload(score)
                row[f"{dimension}_score"] = score.dim_score
            details.append(
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "method": "joint",
                    "questionId": document.question_id,
                    "answerId": document.answer_id,
                    "judgeModel": judge_model,
                    "dimensions": dimensions,
                }
            )
            rows.append(row)
        _write_prediction_tsv(output_dir / "predictions" / "joint.tsv", rows)
        _write_criterion_jsonl(output_dir / "criterion_scores" / "joint.jsonl", details)
        return

    order_records: dict[str, dict[tuple[int, str, str], DimensionScore]] = {}
    orders = (odt_order,) if method == "odt" else ("original", "reverse")
    for order in orders:
        collected: dict[tuple[int, str, str], DimensionScore] = {}
        for document in documents:
            rubric = rubrics[document.question_id]
            for dimension in DIMENSIONS:
                request_id = f"odt:{order}:q{document.question_id}:{document.answer_id}:{dimension}"
                collected[(document.question_id, document.answer_id, dimension)] = _dimension_record(
                    responses[request_id], rubric, dimension, order
                )
        order_records[order] = collected

        rows = []
        details = []
        for document in documents:
            dimensions = {
                dimension: _dimension_payload(
                    collected[(document.question_id, document.answer_id, dimension)]
                )
                for dimension in DIMENSIONS
            }
            rows.append(
                {
                    "taskId": 0,
                    "questionId": document.question_id,
                    "answerId": document.answer_id,
                    **{
                        f"{dimension}_score": collected[
                            (document.question_id, document.answer_id, dimension)
                        ].dim_score
                        for dimension in DIMENSIONS
                    },
                }
            )
            details.append(
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "criterionOrder": order,
                    "questionId": document.question_id,
                    "answerId": document.answer_id,
                    "judgeModel": judge_model,
                    "dimensions": dimensions,
                }
            )
        _write_prediction_tsv(output_dir / "predictions" / f"{order}.tsv", rows)
        _write_criterion_jsonl(output_dir / "criterion_scores" / f"{order}.jsonl", details)

    if method != "osodt":
        return

    rows = []
    details = []
    criterion_deltas: list[float] = []
    dimension_deltas: list[float] = []
    for document in documents:
        symmetric_dimensions: dict[str, SymmetricDimensionScore] = {}
        for dimension in DIMENSIONS:
            key = (document.question_id, document.answer_id, dimension)
            symmetric = average_order_records(
                order_records["original"][key], order_records["reverse"][key]
            )
            symmetric_dimensions[dimension] = symmetric
            dimension_deltas.append(abs(symmetric.original_dim_score - symmetric.reverse_dim_score))
            criterion_deltas.extend(
                abs(item.original_score - item.reverse_score) for item in symmetric.criteria
            )
        rows.append(
            {
                "taskId": 0,
                "questionId": document.question_id,
                "answerId": document.answer_id,
                **{f"{d}_score": symmetric_dimensions[d].dim_score for d in DIMENSIONS},
            }
        )
        details.append(
            {
                "schemaVersion": SCHEMA_VERSION,
                "criterionOrder": "symmetric",
                "questionId": document.question_id,
                "answerId": document.answer_id,
                "judgeModel": judge_model,
                "dimensions": {
                    dimension: _symmetric_dimension_payload(symmetric_dimensions[dimension])
                    for dimension in DIMENSIONS
                },
            }
        )
    _write_prediction_tsv(output_dir / "predictions" / "osodt.tsv", rows)
    _write_criterion_jsonl(output_dir / "criterion_scores" / "symmetric.jsonl", details)
    atomic_write_json(
        output_dir / "diagnostics" / "order_effect.json",
        {
            "documents": len(documents),
            "criterionComparisons": len(criterion_deltas),
            "dimensionComparisons": len(dimension_deltas),
            "meanAbsoluteCriterionOrderDifference": round(
                sum(criterion_deltas) / len(criterion_deltas), 6
            ),
            "meanAbsoluteDimensionOrderDifference": round(
                sum(dimension_deltas) / len(dimension_deltas), 6
            ),
        },
    )


def run_pipeline(
    *,
    method: str,
    config: JudgeConfig,
    report_root: str | Path,
    rubric_dir: str | Path,
    prompt_scales: str | Path,
    output_dir: str | Path,
    odt_order: str = "original",
    preload_responses_from: str | Path | None = None,
) -> None:
    output_dir = Path(output_dir)
    report_root = Path(report_root)
    rubric_dir = Path(rubric_dir)
    prompt_scales = Path(prompt_scales)

    documents = discover_documents(report_root)
    question_ids = {document.question_id for document in documents}
    rubrics = _load_rubrics(rubric_dir, question_ids)
    scales = load_prompt_scales(prompt_scales)
    requests = build_requests(method, documents, rubrics, scales, odt_order=odt_order,
                              prompt_protocol=config.prompt_protocol)

    document_paths = [document.path for document in documents]
    rubric_paths = _rubric_files(rubric_dir, question_ids)
    prompt_collection_sha256 = sha256_request_prompts(
        (request.request_id, request.prompt_sha256) for request in requests
    )
    context_preflight = {
        "enabled": config.tokenizer is not None,
        "tokenizer": config.tokenizer,
        "max_context_tokens": config.max_context_tokens,
        "max_tokens": config.max_tokens,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "odt_order": odt_order if method == "odt" else None,
        "prompt_revision": config.prompt_protocol,
        "scoring_revision": "historical-criterion-round4-v1",
        "reader_revision": READER_REVISION,
        "model_config": {
            "name": config.name,
            "model": config.model,
            "base_url": config.base_url,
            "api_key_env": config.api_key_env,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "timeout_seconds": config.timeout_seconds,
            "max_retries": config.max_retries,
            "concurrency": config.concurrency,
            "tokenizer": config.tokenizer,
            "tokenizer_revision": config.tokenizer_revision,
            "max_context_tokens": config.max_context_tokens,
            "extra_body": config.extra_body,
        },
        "inputs": {
            "document_count": len(documents),
            "question_count": len(question_ids),
            "request_count": len(requests),
            "reports_sha256": sha256_collection(report_root, document_paths),
            "rubrics_sha256": sha256_collection(rubric_dir, rubric_paths),
            "prompt_scales_sha256": sha256_file(prompt_scales),
            "request_prompts_sha256": prompt_collection_sha256,
        },
        "context_preflight": context_preflight,
    }
    if preload_responses_from is not None:
        seed_path = Path(preload_responses_from)
        manifest["preloaded_responses"] = {"sha256": sha256_file(seed_path)}
    guard_manifest(output_dir / "manifest.json", manifest)

    if config.tokenizer is not None:
        diagnostics = validate_request_contexts(
            requests,
            tokenizer_name=config.tokenizer,
            context_length=int(config.max_context_tokens),
            max_tokens=config.max_tokens,
            **({"tokenizer_revision": config.tokenizer_revision} if config.tokenizer_revision else {}),
            **({"chat_template_kwargs": config.extra_body["chat_template_kwargs"]}
               if "chat_template_kwargs" in config.extra_body else {}),
        )
        atomic_write_json(
            output_dir / "diagnostics" / "context_preflight.json",
            {
                "tokenizer": config.tokenizer,
                "context_length": config.max_context_tokens,
                "max_tokens": config.max_tokens,
                "requests": diagnostics,
            },
        )

    if preload_responses_from is not None:
        seed_records = load_jsonl_by_request_id(preload_responses_from)
        expected = {request.request_id: request for request in requests}
        existing = load_jsonl_by_request_id(output_dir / "responses.jsonl")
        missing: list[dict[str, Any]] = []
        for request_id, record in seed_records.items():
            if request_id not in expected:
                raise ValueError(f"preloaded response does not belong to this run: {request_id}")
            request = expected[request_id]
            if not _resume_compatible(record, request, config):
                raise ValueError(f"preloaded response is incompatible: {request_id}")
            parsed = parse_score_object(
                record["raw_response"], request.expected_counts,
                allow_comments=config.prompt_protocol == "e9-schema-v1",
            )
            if parsed != record.get("scores"):
                raise ValueError(f"preloaded scores differ from raw response: {request_id}")
            if request_id in existing and existing[request_id] != record:
                raise ValueError(f"preloaded response conflicts with existing result: {request_id}")
            if request_id not in existing:
                missing.append(record)
        for record in missing:
            upsert_jsonl_record(output_dir / "responses.jsonl", record)
    responses = asyncio.run(execute_requests(requests, config, output_dir / "responses.jsonl"))
    compile_artifacts(method, documents, rubrics, responses, output_dir, odt_order=odt_order)
