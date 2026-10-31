"""Portable command line entry for the final SRAD configuration."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import yaml

from . import evaluator as ev, prompting


def configure(config_path: Path, report_root: Path, rubric_dir: Path, output_dir: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model, generation, method = config["model"], config["generation"], config["srad"]
    alpha, beta = float(method["alpha"]), float(method["beta"])
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (alpha, beta)):
        raise ValueError("SRAD alpha and beta must be finite values in [0, 1]")
    repeats, seeds = method["section_repeats"], method["global_order_seeds"]
    if type(repeats) is not int or repeats < 1:
        raise ValueError("section_repeats must be a positive integer")
    if not isinstance(seeds, list) or not seeds or any(type(x) is not int for x in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("global_order_seeds must be distinct integers")
    if generation["temperature"] != 0 or generation["thinking"] != "off":
        raise ValueError("The released SRAD protocol requires temperature=0 and thinking=off")
    for key in ("section_max_tokens", "global_max_tokens"):
        if type(generation[key]) is not int or generation[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(generation["api_retry_attempts"]) is not int or generation["api_retry_attempts"] < 0:
        raise ValueError("api_retry_attempts must be a nonnegative integer")
    for name, path in (("report root", report_root), ("rubric directory", rubric_dir)):
        if not path.is_dir():
            raise ValueError(f"Missing {name}: {path}")

    ev.BLIND_DOCS_DIR, ev.RUBRICS_DIR = report_root.resolve(), rubric_dir.resolve()
    ev.OUTPUTS_DIR = output_dir.resolve()
    ev.DEBUG_RAW_DIR = ev.OUTPUTS_DIR / "debug_raw"
    ev.READABILITY_PARAGRAPH_ALPHA, ev.READABILITY_GLOBAL_ALPHA = alpha, round(1 - alpha, 10)
    ev.READABILITY_LAST_WEIGHT = beta
    ev.READABILITY_PARAGRAPH_MAX_TOKENS = generation["section_max_tokens"]
    ev.READABILITY_GLOBAL_MAX_TOKENS = generation["global_max_tokens"]
    scales = json.loads(Path(prompting.__file__).with_name("prompt_scales.json").read_text(encoding="utf-8"))
    prompting.READABILITY_RUBRICS = scales["readability_rubrics"]
    prompting.DIM_SCORING_SCALES = {"readability": "\n\n".join(prompting.READABILITY_RUBRICS.values())}

    tasks = ev._discover_reports_from_train()
    if not tasks:
        raise ValueError("No ReportN directories found")
    inputs = {}
    for task in tasks:
        ev.load_rubric(task)
        files = ev.get_task_files(task)
        if not files:
            raise ValueError(f"Report{task} contains no Doc_###.docx files")
        for f in files:
            p = f["path"]
            inputs[f"reports/{p.relative_to(ev.BLIND_DOCS_DIR)}"] = ev._file_sha256(p)
        for p in rubric_dir.glob(f"criteri*{task}.json"):
            inputs[f"rubrics/{p.name}"] = ev._file_sha256(p)
    for p in sorted(Path(__file__).parent.iterdir()):
        if p.suffix in {".py", ".json"}:
            inputs[f"implementation/{p.name}"] = ev._file_sha256(p)
    payload = {"config": config, "inputs": inputs}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    manifest = output_dir / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Inputs, configuration or code changed; choose a new --output-dir")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Nonempty output directory has no SRAD manifest; choose a new --output-dir")
    ev.RUN_FINGERPRINT = fingerprint
    return {"config": config, "tasks": tasks, "manifest": {**payload, "fingerprint": fingerprint}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final SRAD readability method")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--rubric-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepared = configure(args.config, args.report_root, args.rubric_dir, args.output_dir)
    config = prepared["config"]
    key = os.environ.get(config["model"]["api_key_env"])
    if not key:
        parser.error(f"Set the {config['model']['api_key_env']} environment variable")
    ev._write_json_atomic(args.output_dir / "manifest.json", prepared["manifest"])
    ev.run_sequential_global_evaluation(
        prepared["tasks"], {"api_key": key, "api_base": config["model"]["base_url"]},
        model=config["model"]["id"], workers=1, doc_filter=None, thinking_mode="off",
        run_id=None, repeat_count=config["srad"]["section_repeats"],
        global_order_seeds=config["srad"]["global_order_seeds"], sample_size=0,
        sample_seed=42, api_retry_attempts=config["generation"]["api_retry_attempts"],
    )


if __name__ == "__main__":
    main()
