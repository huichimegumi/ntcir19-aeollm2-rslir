"""Artifact writing, hashing, and resume integrity helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    text = json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(path, text)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_collection(root: str | Path, paths: Iterable[str | Path]) -> str:
    """Hash relative file names and bytes so the fingerprint is machine independent."""
    root = Path(root).resolve()
    normalized = sorted((Path(path).resolve() for path in paths), key=lambda p: p.as_posix())
    if not normalized:
        raise ValueError("cannot fingerprint an empty file collection")

    digest = hashlib.sha256()
    for path in normalized:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"fingerprinted file is outside root {root}: {path}") from exc
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def sha256_messages(messages: Sequence[Mapping[str, str]]) -> str:
    payload = json.dumps(
        [dict(message) for message in messages],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_request_prompts(items: Iterable[tuple[str, str]]) -> str:
    """Hash request IDs and their per-request prompt hashes in deterministic order."""
    digest = hashlib.sha256()
    for request_id, prompt_hash in sorted(items):
        digest.update(request_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(prompt_hash.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def guard_manifest(path: str | Path, expected: Mapping[str, Any]) -> None:
    """Create a run manifest once and reject resume under different inputs/settings."""
    path = Path(path)
    normalized = jsonable(expected)
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != normalized:
            raise ValueError(
                f"run manifest differs from the requested run: {path}; use a new output directory"
            )
        return

    if path.parent.exists():
        unexpected = [item for item in path.parent.iterdir() if item.name != path.name]
        if unexpected:
            preview = ", ".join(item.name for item in sorted(unexpected)[:10])
            raise ValueError(
                f"output directory is non-empty but has no manifest: {path.parent}; "
                f"unexpected entries: {preview}"
            )
    atomic_write_json(path, normalized)


def load_jsonl_by_request_id(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            request_id = str(record.get("request_id", ""))
            if not request_id:
                raise ValueError(f"missing request_id at {path}:{line_no}")
            if request_id in records:
                raise ValueError(f"duplicate request_id {request_id!r} at {path}:{line_no}")
            records[request_id] = record
    return records


def upsert_jsonl_record(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    records = load_jsonl_by_request_id(path)
    request_id = str(record.get("request_id", ""))
    if not request_id:
        raise ValueError("record must contain request_id")
    records[request_id] = jsonable(record)
    text = "".join(
        json.dumps(records[key], ensure_ascii=False, sort_keys=True) + "\n"
        for key in sorted(records)
    )
    _atomic_write_text(path, text)
