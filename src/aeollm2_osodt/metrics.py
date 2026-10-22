"""AEOLLM-2 overview §3.2 metrics, with explicit historical diagnostics.

All official metrics are question-macro averages. Undefined correlations propagate
as NaN rather than silently changing the evaluated question population.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .constants import DIMENSIONS

KEY_COLUMNS = ["questionId", "answerId"]
SCORE_COLUMNS = {dimension: f"{dimension}_score" for dimension in DIMENSIONS}
SCOPES = ("weighted_overall", *DIMENSIONS)


def _validated_weight_vector(weights: Mapping[str, float]) -> np.ndarray:
    missing = [dimension for dimension in DIMENSIONS if dimension not in weights]
    if missing:
        raise ValueError(f"dimension weights missing keys: {missing}")
    vector = np.asarray([float(weights[dimension]) for dimension in DIMENSIONS], dtype=float)
    if not np.isfinite(vector).all() or (vector < 0).any():
        raise ValueError("dimension weights must be finite and non-negative")
    if vector.sum() <= 0:
        raise ValueError("dimension weights must have positive total weight")
    return vector / vector.sum()


def weighted_total(values: np.ndarray, weights: Mapping[str, float]) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(DIMENSIONS):
        raise ValueError(f"values must have shape (n, {len(DIMENSIONS)})")
    if not np.isfinite(matrix).all():
        raise ValueError("scores must be finite")
    # BLAS uses different reduction kernels for C/F strided matrices. A last-bit
    # difference can turn an exact tie into a preference; freeze the row layout.
    return np.ascontiguousarray(matrix) @ _validated_weight_vector(weights)


def pairwise_accuracy(gold: np.ndarray, pred: np.ndarray) -> tuple[float, int, int]:
    gold = np.asarray(gold, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if gold.ndim != 1 or pred.ndim != 1 or gold.shape != pred.shape:
        raise ValueError("gold and pred must be one-dimensional arrays of the same shape")
    if not np.isfinite(gold).all() or not np.isfinite(pred).all():
        raise ValueError("pairwise accuracy requires finite scores")
    correct = total = 0
    for i, j in itertools.combinations(range(len(gold)), 2):
        total += 1
        correct += int(np.sign(gold[i] - gold[j]) == np.sign(pred[i] - pred[j]))
    return (float(correct / total) if total else np.nan, correct, total)


def _normalize_score_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if not frame.columns.is_unique:
        raise ValueError("duplicate column names")
    if any(d in frame and SCORE_COLUMNS[d] in frame for d in DIMENSIONS):
        raise ValueError("ambiguous dimension and _score columns")
    return frame.rename(columns={v: k for k, v in SCORE_COLUMNS.items() if v in frame.columns})


def _validate_frame(name: str, frame: pd.DataFrame) -> None:
    missing = sorted({*KEY_COLUMNS, *DIMENSIONS} - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{name} is empty")
    qids = pd.to_numeric(frame["questionId"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(qids).all() or (qids <= 0).any() or (qids != np.floor(qids)).any():
        raise ValueError(f"{name} questionId must be positive integers")
    frame["questionId"] = qids.astype(int)
    if frame["answerId"].isna().any() or not frame["answerId"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise ValueError(f"{name} answerId must be non-empty strings")
    if "taskId" in frame and not (pd.to_numeric(frame["taskId"], errors="raise") == 0).all():
        raise ValueError(f"{name} requires taskId=0 for Deep Research Evaluation")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{name} contains duplicate document keys")
    if frame[list(DIMENSIONS)].apply(
        lambda column: column.map(lambda x: isinstance(x, (bool, np.bool_)))
    ).any().any():
        raise ValueError(f"{name} contains boolean scores")
    matrix = frame[list(DIMENSIONS)].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite scores")
    if ((matrix < 0) | (matrix > 10)).any():
        raise ValueError(f"{name} contains scores outside [0, 10]")


def _score_metrics(gold: np.ndarray, pred: np.ndarray) -> dict:
    if len(gold) < 2:
        status = "undefined_fewer_than_two_documents"
    elif np.ptp(gold) == 0 or np.ptp(pred) == 0:
        status = "undefined_constant_scores"
    else:
        status = "defined"
    accuracy, correct, pairs = pairwise_accuracy(gold, pred)
    return {
        "spearman": float(spearmanr(gold, pred).statistic) if status == "defined" else np.nan,
        "kendall": float(kendalltau(gold, pred, variant="b").statistic) if status == "defined" else np.nan,
        "accuracy": accuracy,
        "correlation_status": status,
        "pair_correct": correct,
        "pair_total": pairs,
    }


def evaluate_predictions(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    dimension_weights: Mapping[int, Mapping[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a wide summary and per-question table (overall names stay stable).

    Dimension fields use e.g. ``spearman_insight``. ``accuracy`` is the official
    question-macro value; ``accuracy_pair_micro_legacy`` preserves the historical
    weighting. Undefined metrics remain NaN, with counts and a warning.
    """
    pred = _normalize_score_columns(predictions.copy())
    truth = _normalize_score_columns(labels.copy())
    _validate_frame("predictions", pred)
    _validate_frame("labels", truth)
    pred_keys = set(map(tuple, pred[KEY_COLUMNS].itertuples(index=False, name=None)))
    truth_keys = set(map(tuple, truth[KEY_COLUMNS].itertuples(index=False, name=None)))
    if pred_keys != truth_keys:
        raise ValueError("prediction and label document sets differ")
    merged = pred.merge(truth, on=KEY_COLUMNS, suffixes=("_pred", "_gt"), validate="one_to_one")
    details = []
    for question_id, sub in merged.groupby("questionId", sort=True):
        qid = int(question_id)
        if qid not in dimension_weights:
            raise ValueError(f"missing dimension weights for question {qid}")
        pred_matrix = sub[[f"{d}_pred" for d in DIMENSIONS]].to_numpy(dtype=float)
        gold_matrix = sub[[f"{d}_gt" for d in DIMENSIONS]].to_numpy(dtype=float)
        values = [(weighted_total(gold_matrix, dimension_weights[qid]),
                   weighted_total(pred_matrix, dimension_weights[qid]))]
        values.extend((gold_matrix[:, i], pred_matrix[:, i]) for i in range(len(DIMENSIONS)))
        row = {"reportId": qid, "n_documents": len(sub)}
        for scope, (gold, predicted) in zip(SCOPES, values):
            suffix = "" if scope == "weighted_overall" else f"_{scope}"
            row.update({f"{k}{suffix}": v for k, v in _score_metrics(gold, predicted).items()})
        details.append(row)
    detail_frame = pd.DataFrame(details)
    summary = {
        "metric_protocol": "aeollm2-overview-v1-question-macro",
        "undefined_policy": "propagate_nan",
        "n_questions": len(detail_frame),
        "n_documents": len(merged),
    }
    undefined = False
    for scope in SCOPES:
        suffix = "" if scope == "weighted_overall" else f"_{scope}"
        for metric in ("spearman", "kendall", "accuracy"):
            column = f"{metric}{suffix}"
            summary[column] = float(detail_frame[column].mean(skipna=False))
            summary[f"{metric}_defined_questions{suffix}"] = int(detail_frame[column].notna().sum())
            if metric in {"spearman", "kendall"}:
                summary[f"{metric}_defined_question_mean_legacy{suffix}"] = float(
                    detail_frame[column].mean(skipna=True)
                )
            undefined |= detail_frame[column].isna().any()
        correct = int(detail_frame[f"pair_correct{suffix}"].sum())
        pairs = int(detail_frame[f"pair_total{suffix}"].sum())
        summary[f"pair_correct{suffix}"] = correct
        summary[f"pair_total{suffix}"] = pairs
        summary[f"accuracy_pair_micro_legacy{suffix}"] = correct / pairs if pairs else np.nan
    if undefined:
        warnings.warn("undefined per-question metrics; macro averages propagate NaN; "
                      "see correlation_status and *_defined_questions", RuntimeWarning, stacklevel=2)
    return pd.DataFrame([summary]), detail_frame


def metrics_by_scope(summary: pd.DataFrame) -> pd.DataFrame:
    """Present all fifteen official metrics as five readable rows."""
    source = summary.iloc[0]
    rows = []
    for scope in SCOPES:
        suffix = "" if scope == "weighted_overall" else f"_{scope}"
        rows.append({"scope": scope, **{
            name: source[f"{name}{suffix}"] for name in (
                "spearman", "kendall", "accuracy", "accuracy_pair_micro_legacy",
                "spearman_defined_questions", "kendall_defined_questions", "accuracy_defined_questions",
                "spearman_defined_question_mean_legacy", "kendall_defined_question_mean_legacy",
                "pair_correct", "pair_total",
            )
        }})
    return pd.DataFrame(rows)
