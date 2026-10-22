import json
from pathlib import Path

import pytest

from aeollm2_osodt.artifacts import guard_manifest, load_jsonl_by_request_id, sha256_collection


def test_guard_manifest_rejects_configuration_drift(tmp_path: Path):
    path = tmp_path / "run" / "manifest.json"
    guard_manifest(path, {"model": "a", "temperature": 0.0})
    guard_manifest(path, {"model": "a", "temperature": 0.0})

    with pytest.raises(ValueError, match="manifest differs"):
        guard_manifest(path, {"model": "b", "temperature": 0.0})


def test_guard_manifest_rejects_unmanaged_nonempty_directory(tmp_path: Path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "responses.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty"):
        guard_manifest(output / "manifest.json", {"model": "a"})


def test_jsonl_loader_rejects_duplicate_request_ids(tmp_path: Path):
    path = tmp_path / "responses.jsonl"
    rows = [
        {"request_id": "same", "parse_valid": True},
        {"request_id": "same", "parse_valid": False},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate request_id"):
        load_jsonl_by_request_id(path)


def test_collection_hash_depends_on_relative_path_and_content(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    first = root / "a.txt"
    second = root / "b.txt"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")

    before = sha256_collection(root, [first, second])
    second.write_text("changed", encoding="utf-8")
    after = sha256_collection(root, [first, second])

    assert before != after
