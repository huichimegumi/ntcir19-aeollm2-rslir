"""Ensure the combined system preserves identities and the three OSODT scores."""
import pandas as pd
import pytest
from aeollm2_osodt.submission import COLUMNS, read_predictions, write_submission
from aeollm2_srad.submission import merge_predictions


@pytest.fixture
def inputs(tmp_path):
    a, b = tmp_path/'osodt.tsv', tmp_path/'srad.tsv'
    pd.DataFrame([[0,1,'Doc_001',7.123456789,6,5,1], [0,1,'Doc_002',4,3,2,9]], columns=COLUMNS).to_csv(a, sep='\t', index=False)
    pd.DataFrame([[0,1,'Doc_002','','','',6.5], [0,1,'Doc_001','','','',8.2]], columns=COLUMNS).to_csv(b, sep='\t', index=False)
    return a,b


def test_merge_matches_ids_and_preserves_content(inputs,tmp_path):
    a,b=inputs
    actual=merge_predictions(a,b)
    original=read_predictions(a)
    pd.testing.assert_frame_equal(actual[COLUMNS[:-1]],original[COLUMNS[:-1]])
    assert actual['readability_score'].tolist()==[8.2,6.5]
    out=tmp_path/'submission.tsv'
    write_submission(actual,out)
    pd.testing.assert_frame_equal(read_predictions(out),actual)
    assert not out.read_text().startswith('taskId')


@pytest.mark.parametrize('error',['duplicate','missing','extra','nan','infinity','range'])
def test_rejects_invalid_srad(inputs,error):
    a,b=inputs
    frame=pd.read_csv(b,sep='\t',keep_default_na=False)
    if error=='duplicate':frame=pd.concat([frame,frame.iloc[:1]])
    elif error=='missing':frame=frame.iloc[:1]
    elif error=='extra':frame.loc[0,'answerId']='Doc_999'
    else:frame.loc[0,'readability_score']={'nan':float('nan'),'infinity':float('inf'),'range':11}[error]
    frame.to_csv(b,sep='\t',index=False)
    with pytest.raises(ValueError):merge_predictions(a,b)
