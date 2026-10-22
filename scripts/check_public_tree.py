#!/usr/bin/env python3
"""Reject local-only files in the Git index or an unpacked source distribution."""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import re
import subprocess

LOCAL_ROOTS = frozenset({'.local', 'data', 'outputs', 'logs', 'models', 'checkpoints', 'dist'})
MODEL_SUFFIXES = frozenset({'.safetensors', '.gguf', '.ckpt', '.pt', '.pth'})
HOME_PATH = re.compile(r'/(?:home|Users)/[\w.-]+/|[A-Za-z]:[\\/]Users[\\/][\w.-]+[\\/]')


def check_file(path: str, payload: bytes, *, symlink: bool = False) -> list[str]:
    """Check a relative path and its staged/released contents without modifying it."""
    name = PurePosixPath(path)
    issues = []
    if name.parts[0] in LOCAL_ROOTS or path == 'AGENTS.md':
        issues.append('local-only workspace file')
    if name.parts[:2] in {('results', 'validation'), ('configs', 'local')}:
        issues.append('local experiment results or configuration')
    if (name.name == '.env' or name.name.startswith('.env.')) and name.name not in {'.env.example', '.env.sample'}:
        issues.append('local environment file')
    if name.parts[0] == 'configs' and name.name.endswith(('.local.yaml', '.local.yml')):
        issues.append('personal service configuration')
    if name.suffix.lower() == '.docx' or name.suffix.lower() in MODEL_SUFFIXES:
        issues.append('task report or model weights')
    if symlink:
        issues.append('symlink: publish explicit portable files')
    if HOME_PATH.search(payload.decode('utf-8', errors='replace')):
        issues.append('personal home-directory path in contents')
    return issues


def indexed_files():
    raw = subprocess.check_output(['git', 'ls-files', '--stage', '-z'])
    for entry in raw.split(b'\0'):
        if not entry:
            continue
        metadata, raw_path = entry.split(b'\t', 1)
        mode, oid, stage = metadata.decode().split()
        if stage != '0':
            raise ValueError('resolve merge conflicts before checking the public tree')
        payload = subprocess.check_output(['git', 'cat-file', 'blob', oid])
        yield raw_path.decode(), payload, mode == '120000'


def released_files(root: Path):
    for path in sorted(root.rglob('*')):
        if '.git' in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            yield path.relative_to(root).as_posix(), b'', True
        elif path.is_file():
            yield path.relative_to(root).as_posix(), path.read_bytes(), False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, help='Inspect an unpacked source tree instead of the Git index')
    args = parser.parse_args()
    count, failures = 0, []
    files = released_files(args.root.resolve()) if args.root else indexed_files()
    for path, payload, symlink in files:
        count += 1
        failures.extend(f'{path}: {reason}' for reason in check_file(path, payload, symlink=symlink))
    if failures:
        print('\n'.join(failures))
        return 1
    print(f'Public tree check passed ({count} files).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
