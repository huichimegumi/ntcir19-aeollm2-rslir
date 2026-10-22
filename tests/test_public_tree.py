"""Keep publication checks sensitive to index contents and narrow local paths."""
import importlib.util
from pathlib import Path
import subprocess
import sys

SPEC = importlib.util.spec_from_file_location('public_tree', Path(__file__).parents[1] / 'scripts/check_public_tree.py')
PUBLIC_TREE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLIC_TREE)


def test_local_paths_and_symlinks_are_rejected():
    for path in ['.local/notes.md', 'data/report.docx', 'outputs/responses.jsonl',
                 'results/validation/metrics.csv', 'configs/local/server.yaml',
                 'configs/server.local.yaml', '.env.production', 'models/weights.safetensors']:
        assert PUBLIC_TREE.check_file(path, b'')
    assert PUBLIC_TREE.check_file('configs/server.yaml', b'', symlink=True)


def test_paper_artifacts_and_portable_examples_are_allowed():
    for path in ['results/paper/table3_dimension_isolation.csv', 'prompts/prompt_scales.json',
                 'configs/qwen3-8b-e9.yaml', '.env.example', 'tests/fixtures/layout_tie.json']:
        assert PUBLIC_TREE.check_file(path, b'portable example') == []


def test_git_index_is_checked_even_when_working_copy_is_clean(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    file = tmp_path / 'README.md'
    personal = '/'.join(['', 'home', 'researcher', 'private', 'model'])
    file.write_text(personal)
    subprocess.run(['git', 'add', 'README.md'], cwd=tmp_path, check=True)
    file.write_text('portable content')
    script = Path(PUBLIC_TREE.__file__)
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1
    assert 'personal home-directory' in result.stdout
    subprocess.run(['git', 'add', 'README.md'], cwd=tmp_path, check=True)
    assert subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True).returncode == 0
