import asyncio
from dataclasses import replace
import json
import re

import pandas as pd
import pytest
from docx import Document

import aeollm2_osodt.pipeline as pipeline
from aeollm2_osodt.client import JudgeRequestError, JudgeResponse
from aeollm2_osodt.config import JudgeConfig
from aeollm2_osodt.constants import DIMENSIONS


@pytest.mark.parametrize('protocol', ['paper-v1', 'e9-schema-v1'])
def test_multidocument_ordered_scores_concurrency_partial_failure_resume_and_drift(tmp_path, monkeypatch, protocol):
    root, rubric_dir = tmp_path/'reports', tmp_path/'rubrics'
    rubric_dir.mkdir()
    counts = dict(zip(DIMENSIONS, [2, 3, 5, 1]))
    for qid in [1, 2]:
        folder = root/f'Report{qid}'; folder.mkdir(parents=True)
        for number in [1, 2]:
            doc = Document(); doc.add_paragraph(f'FIXTURE Q={qid} DOC={number}')
            doc.save(folder/f'Doc_{qid}{number}.docx')
        rubric = {'id':qid,'prompt':'question', 'dimension_weight':dict.fromkeys(DIMENSIONS,1),
                  'criterions':{d:[{'criterion':f'C{i}', 'explanation':'same text allowed',
                                    'weight':0 if i==1 and count>1 else i}
                                   for i in range(1,count+1)] for d,count in counts.items()}}
        (rubric_dir/f'criterion{qid}.json').write_text(json.dumps(rubric))
    class Judge:
        calls=[]; active=0; peak=0; failed=False; fail=True
        def __init__(self, config): self.config=config
        async def complete(self, messages, *, validator):
            user=messages[-1]['content']
            q,n=map(int,re.search(r'FIXTURE Q=(\d) DOC=(\d)',user).groups())
            d=re.search(r'专门评估 `([^`]+)`',user).group(1)
            section=user.split('【本维度具体准则】')[1].split('【内部评审步骤】')[0]
            indices=list(map(int,re.findall(r'准则：C(\d+)',section)))
            reverse=indices[0]!=1
            key=(q,n,d,reverse)
            type(self).calls.append(key); type(self).active+=1
            type(self).peak=max(self.peak,self.active)
            try:
                await asyncio.sleep(0.001 if n==1 else 0)
                if self.fail and not self.failed and q==2 and n==1 and d=='insight':
                    type(self).failed=True
                    raise JudgeRequestError('fixture failure',attempt_count=1,raw_response='bad')
                values=[float((q+n+DIMENSIONS.index(d)+i)%7+1)+(0.375 if reverse else 0) for i in indices]
                raw=json.dumps({d:values}); validator(raw)
                return JudgeResponse(raw,1,0,10,4)
            finally: type(self).active-=1
    monkeypatch.setattr(pipeline,'OpenAICompatibleJudge',Judge)
    config=JudgeConfig(name='fixture',model='fixture',base_url='https://fixture',api_key_env='UNUSED',
                       concurrency=3,prompt_protocol=protocol)
    output=tmp_path/'out'
    kwargs=dict(method='osodt',config=config,report_root=root,rubric_dir=rubric_dir,
                prompt_scales='prompts/prompt_scales.json',output_dir=output)
    with pytest.raises(ValueError,match='requests are invalid'):
        pipeline.run_pipeline(**kwargs)
    assert len(Judge.calls)==32
    assert 1<Judge.peak<=3
    Judge.fail=False
    pipeline.run_pipeline(**kwargs)
    assert len(Judge.calls)==33
    records=[json.loads(line) for line in (output/'responses.jsonl').read_text().splitlines()]
    assert len(records)==32 and all(r['parse_valid'] for r in records)
    assert all(r['prompt_revision']==protocol for r in records)
    symmetric=[json.loads(l) for l in (output/'criterion_scores/symmetric.jsonl').read_text().splitlines()]
    for record in symmetric:
        q=record['questionId'];n=int(record['answerId'][-1])
        for di,d in enumerate(DIMENSIONS):
            scores=[];weights=[]
            for i,item in enumerate(record['dimensions'][d]['criteria'],1):
                expected=float((q+n+di+i)%7+1)+(0.1875 if counts[d]>1 else 0)
                assert item['criterionId']==f'{d}_{i}'
                assert item['score']==expected
                scores.append(expected);weights.append(0 if i==1 and counts[d]>1 else i)
            assert record['dimensions'][d]['dim_score']==round(sum(s*w for s,w in zip(scores,weights))/sum(weights),4)
    pipeline.run_pipeline(**kwargs)
    assert len(Judge.calls)==33
    with pytest.raises(ValueError,match='manifest differs'):
        pipeline.run_pipeline(**{**kwargs,'config':replace(config,temperature=0.2)})
    with pytest.raises(ValueError,match='manifest differs'):
        pipeline.run_pipeline(**{**kwargs,'config':replace(config,prompt_protocol='e9-schema-v1' if protocol=='paper-v1' else 'paper-v1')})
    records[0]['scores'][records[0]['dimension']][0]=0.1234
    (output/'responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises(ValueError,match='stored scores.*raw response'):
        pipeline.run_pipeline(**kwargs)
