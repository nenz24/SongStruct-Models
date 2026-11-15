"""Locate the first changed stage and same-input/different-output calls in two diagnostic reports."""
import argparse
from pathlib import Path
from finetuning.common import read, write


def compare(left,right):
    a,b=left['stage_diagnostics'],right['stage_diagnostics']
    if a['identity']!=b['identity']:raise ValueError('Diagnostic manifest/protocol identities differ.')
    old=a['stages'];new=b['stages']
    if [r['key'] for r in old]!=[r['key'] for r in new]:raise ValueError('Diagnostic stage inventories/order differ.')
    rows=[];inputs={}
    for x,y in zip(old,new):
        for field in ('shape','count','sample_indices','role','direction'):
            if x[field]!=y[field]:raise ValueError('Stage layout differs: '+y['key'])
        same=x['sha256_float32']==y['sha256_float32']
        row={'key':y['key'],'role':y['role'],'direction':y['direction'],'exact_float32_match':same,
            'sample_max_abs_error':max((abs(i-j) for i,j in zip(x['samples'],y['samples'])),default=0)}
        group='/'.join(y['key'].split('/')[:2])
        if y['direction']=='input':inputs.setdefault(group,[]).append(same)
        else:row['all_recorded_inputs_identical']=bool(inputs.get(group)) and all(inputs[group])
        rows.append(row)
    changed=[r for r in rows if not r['exact_float32_match']]
    outputs=[r for r in changed if r['direction']=='output']
    return {'stages_compared':len(rows),'changed_stages':len(changed),
        'first_changed_stage':next(iter(changed),None),'first_changed_output':next(iter(outputs),None),
        'same_input_changed_outputs':[r for r in outputs if r['all_recorded_inputs_identical']],
        'comparisons':rows,'note':'Hash is whole-array Float32 equality; sampled magnitude is not a full-array error bound. Same-input changes locate calls for investigation, not a proven runtime cause.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('left',type=Path);p.add_argument('right',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=compare(read(a.left),read(a.right));write(a.output,r)
    print('First changed output:',r['first_changed_output'])
    print('Same-input changed outputs:',len(r['same_input_changed_outputs']))
    print('Report:',a.output)


if __name__=='__main__':main()
