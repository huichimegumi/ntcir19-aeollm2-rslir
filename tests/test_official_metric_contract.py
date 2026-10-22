"""Hand-derived contracts independent of the implementation's aggregation."""
import math

import numpy as np
import pandas as pd
import pytest

from aeollm2_osodt.constants import DIMENSIONS
from aeollm2_osodt.metrics import evaluate_predictions, metrics_by_scope
from aeollm2_osodt.submission import read_predictions, write_submission


def frames(groups):
    pred, gold = [], []
    for qid, (g, p) in enumerate(groups, 1):
        for i, (gv, pv) in enumerate(zip(g, p)):
            base = {"questionId": qid, "answerId": f"Doc_{qid}_{i}"}
            pred.append({**base, **dict.fromkeys(DIMENSIONS, float(pv))})
            gold.append({**base, **dict.fromkeys(DIMENSIONS, float(gv))})
    weights = {q: dict.fromkeys(DIMENSIONS, 1) for q in range(1, len(groups) + 1)}
    return pd.DataFrame(pred), pd.DataFrame(gold), weights


def test_question_macro_is_distinct_from_legacy_pair_micro():
    summary, details = evaluate_predictions(*frames([([1, 2, 3], [1, 2, 3]), ([1, 2, 3, 4], [4, 3, 2, 1])]))
    assert summary.loc[0, "accuracy"] == 0.5
    assert summary.loc[0, "accuracy_pair_micro_legacy"] == 1 / 3
    assert summary.loc[0, "pair_total"] == 9
    assert summary.loc[0, "spearman"] == pytest.approx(0)
    scoped = metrics_by_scope(summary)
    assert len(scoped) == 5
    assert scoped.accuracy.tolist() == [0.5] * 5
    assert details.accuracy.tolist() == [1, 0]


@pytest.mark.parametrize('gold,pred,sp,tau,acc', [
    ([1, 2, 3], [1, 2, 3], 1, 1, 1),
    ([1, 2, 3], [3, 2, 1], -1, -1, 0),
    ([1, 2, 2], [1, 2, 3], math.sqrt(3) / 2, math.sqrt(2 / 3), 2 / 3),
    ([1, 2, 3], [1, 2, 2], math.sqrt(3) / 2, math.sqrt(2 / 3), 2 / 3),
    ([1, 2, 2], [1, 3, 3], 1, 1, 1),
    ([1, 2], [2, 1], -1, -1, 0),
])
def test_rank_and_strict_tie_contract(gold, pred, sp, tau, acc):
    summary, _ = evaluate_predictions(*frames([(gold, pred)]))
    row = summary.iloc[0]
    assert row.spearman == pytest.approx(sp)
    assert row.kendall == pytest.approx(tau)
    assert row.accuracy == pytest.approx(acc)


@pytest.mark.parametrize('g,p,accuracy', [([2, 2, 2], [3, 3, 3], 1), ([2], [3], np.nan), ([1, 2], [3, 3], 0)])
def test_undefined_metrics_are_explicit(g, p, accuracy):
    with pytest.warns(RuntimeWarning, match='propagate NaN'):
        summary, details = evaluate_predictions(*frames([(g, p)]))
    assert np.isnan(summary.loc[0, 'spearman'])
    assert summary.loc[0, 'spearman_defined_questions'] == 0
    assert details.loc[0, 'correlation_status'].startswith('undefined')
    if np.isnan(accuracy):
        assert np.isnan(summary.loc[0, 'accuracy'])
    else:
        assert summary.loc[0, 'accuracy'] == accuracy


def test_undefined_question_is_never_silently_excluded():
    with pytest.warns(RuntimeWarning):
        summary, _ = evaluate_predictions(*frames([([1, 2, 3], [1, 2, 3]), ([2, 2, 2], [3, 3, 3])]))
    assert np.isnan(summary.loc[0, 'spearman'])
    assert summary.loc[0, 'spearman_defined_questions'] == 1
    assert summary.loc[0, 'n_questions'] == 2


def test_question_specific_weights_apply_to_both_prediction_and_gold():
    pred, gold, weights = frames([([1, 2, 3], [3, 2, 1]), ([1, 2, 3], [3, 2, 1])])
    for qid in [1, 2]:
        mask = pred.questionId == qid
        pred.loc[mask, 'insight'] = [1, 2, 3]
        gold.loc[mask, 'insight'] = [3, 2, 1] if qid == 1 else [1, 2, 3]
    weights[1] = dict(zip(DIMENSIONS, [1, 0, 0, 0]))
    weights[2] = dict(zip(DIMENSIONS, [0, 7, 0, 0]))
    summary, detail = evaluate_predictions(pred, gold, weights)
    assert detail.accuracy.tolist() == [0, 1]
    assert summary.loc[0, 'accuracy'] == 0.5


@pytest.mark.parametrize('change,pattern', [
    ('duplicate', 'duplicate'), ('missing', 'document sets'), ('nan', 'non-finite'),
    ('inf', 'non-finite'), ('negative', 'outside'), ('large', 'outside'),
    ('bad_column', 'missing columns'), ('null_key', 'answerId'), ('fractional_qid', 'integers'),
    ('ambiguous', 'ambiguous'), ('empty', 'empty'), ('boolean', 'boolean'),
])
def test_invalid_population_and_values_fail(change, pattern):
    pred, gold, weights = frames([([1, 2, 3], [1, 2, 3])])
    if change == 'duplicate': pred = pd.concat([pred, pred.iloc[[0]]])
    if change == 'missing': pred = pred.iloc[:-1]
    if change in {'nan', 'inf', 'negative', 'large'}:
        pred.loc[0, 'insight'] = {'nan': np.nan, 'inf': np.inf, 'negative': -1, 'large': 11}[change]
    if change == 'bad_column': pred = pred.drop(columns=['insight'])
    if change == 'null_key': pred.loc[0, 'answerId'] = None
    if change == 'fractional_qid': pred['questionId'] = 1.5
    if change == 'ambiguous': pred['insight_score'] = pred['insight']
    if change == 'empty': pred = pred.iloc[:0]
    if change == 'boolean': pred['insight'] = True
    with pytest.raises(ValueError, match=pattern):
        evaluate_predictions(pred, gold, weights)


def test_submission_headerless_and_internal_tsv_roundtrip(tmp_path):
    pred, _, _ = frames([([1, 2, 3], [1, 2, 3])])
    path = tmp_path / 'RSLIR_OSODT.txt'
    write_submission(pred, path)
    assert path.read_text().splitlines()[0].split() == ['0', '1', 'Doc_1_0', '1.0', '1.0', '1.0', '1.0']
    read = read_predictions(path)
    internal = tmp_path / 'internal.tsv'
    read.to_csv(internal, sep='\t', index=False)
    pd.testing.assert_frame_equal(read, read_predictions(internal))
    assert read.shape == (3, 7)


@pytest.mark.parametrize('line', ['0 1 Doc_1 8 8 8', '0 1 Doc_1 8 8 8 8 EXTRA', '0 1 Doc_1 8 8 8 nan'])
def test_submission_wrong_width_or_nonfinite_rejected(tmp_path, line):
    path = tmp_path / 'bad.txt'
    path.write_text(line + '\n')
    with pytest.raises((ValueError, pd.errors.ParserError)):
        read_predictions(path)


def test_weighted_overall_is_invariant_to_matrix_memory_layout():
    import json
    from pathlib import Path
    from aeollm2_osodt.metrics import weighted_total
    fixture=json.loads((Path(__file__).parent/'fixtures/layout_tie.json').read_text())
    a=np.array(fixture['matrix'],order='C')
    b=np.array(fixture['matrix'],order='F')
    left=weighted_total(a,fixture['weights'])
    right=weighted_total(b,fixture['weights'])
    assert np.array_equal(left,right)
    # This pair is mathematically tied; the old F-strided BLAS path separated it.
    assert left[13] == left[15]


def test_prediction_file_roundtrip_preserves_exact_binary_float(tmp_path):
    pred,_,_=frames([([1,2,3],[1,2,3])])
    pred.loc[0,'insight']=6.7212499999999995
    path=tmp_path/'roundtrip.txt'
    write_submission(pred,path)
    assert read_predictions(path).loc[0,'insight_score']==pred.loc[0,'insight']
