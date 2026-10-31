from __future__ import annotations

from pathlib import Path

import pandas as pd

from aeollm2_osodt.config import load_judge_config


ROOT = Path(__file__).resolve().parents[1]


def test_deepseek_paper_config_matches_archived_request_semantics() -> None:
    config = load_judge_config(ROOT / "configs/deepseek-v4-flash-paper.yaml")

    assert config.name == "deepseek-v4-flash-paper"
    assert config.model == "deepseek-v4-flash"
    assert config.base_url == "https://api.deepseek.com"
    assert config.api_key_env == "DEEPSEEK_API_KEY"
    assert config.temperature == 0.0
    assert config.max_tokens == 2048
    assert config.concurrency == 4
    assert config.tokenizer is None
    assert config.max_context_tokens is None
    assert config.extra_body == {"thinking": {"type": "disabled"}}


def test_formal_table10_delta_and_manifest_boundary() -> None:
    metrics_path = ROOT / "results/paper/formal/official_metrics.csv"
    metrics = pd.read_csv(metrics_path).set_index("method")

    baseline = metrics.loc["DeepSeek-v3.2"]
    rslir = metrics.loc["RSLIR (SRAD+OSODT)"]
    reported_delta = metrics.loc["Delta (RSLIR - DeepSeek-v3.2)"]

    recomputed_delta = rslir - baseline
    pd.testing.assert_series_equal(
        recomputed_delta.round(4),
        reported_delta.round(4),
        check_names=False,
    )

    assert rslir["weighted_spearman"] == 0.7071
    assert rslir["weighted_kendall"] == 0.5592
    assert rslir["weighted_accuracy"] == 0.7796
