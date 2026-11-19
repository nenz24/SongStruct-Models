#!/usr/bin/env python3
"""Create Python-reference cases for the real Swift boundary runtime, including the uploaded app report."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from review_boundary_shifts import propose, validate, PARAMETERS


def expected_result(report):
    proposal = propose(report); sections, _, _, _ = validate(report)
    names = dict(boundary_index='boundaryIndex',left_label='leftLabel',right_label='rightLabel',
        original_seconds='originalSeconds',candidate_seconds='candidateSeconds',candidate_checks='candidateChecks',
        left_anchor='leftAnchor',right_anchor='rightAnchor',log_gain_seconds='logGainSeconds',mean_log_gain='meanLogGain',
        shift_seconds='shiftSeconds')
    def convert(value):
        if isinstance(value,list):return [convert(v) for v in value]
        if isinstance(value,dict):return {names.get(k,k):convert(v) for k,v in value.items()}
        return value
    decisions=convert(proposal['boundaries'])
    for row in decisions:
        row['applied']=row['status']=='candidate_requires_review'
        if row['applied']:row['status']='applied_debug'
    return dict(protocolVersion='context-boundary-shift-review-v1',
        parameters=dict(anchorProbability=PARAMETERS['anchor_probability'],anchorMargin=PARAMETERS['anchor_margin'],
            coverage=PARAMETERS['coverage'],meanLogGain=PARAMETERS['mean_log_gain'],
            minimumShiftBars=PARAMETERS['minimum_shift_bars'],minimumSectionBars=PARAMETERS['minimum_section_bars']),
        barSeconds=proposal['bar_seconds'],originalSections=sections,refinedSections=proposal['simulated_sections'],
        decisions=decisions,appliedCount=proposal['candidate_count'])


def synthetic(old=10, actual=12):
    sections=[dict(start=0.,end=float(old),label='bridge'),dict(start=float(old),end=24.,label='chorus')]
    frames=[]
    for i in range(96):
        p=[.01]*8;p[3 if i*.25<actual else 2]=.93
        frames.append(dict(time=i*.25,duration=.25,probabilities=p))
    return dict(title='synthetic',state='completed',contextEvidence=dict(selectedEpoch=5,precision='fp32',
        labelOrder=['intro','verse','chorus','bridge','inst','outro','silence','pre-chorus'],
        checkpointSHA256='synthetic',sections=copy.deepcopy(sections)),structureEvidence=dict(
            labelOrder=['intro','verse','chorus','bridge','inst','outro','silence','pre-chorus'],frames=frames,
            bpm=120.,beatsPerBar=4,downbeatOffset=0.,assembledSections=sections,
            appleBoundaries=[dict(start=s['start'],end=s['end']) for s in sections]))


def cases():
    reports=[('move_later',synthetic()),('move_earlier',synthetic(12,10)),
        ('identity',synthetic(10,10)),('large_move_abstains',synthetic(10,16))]
    r=synthetic();r['structureEvidence']['frames']=[f for f in r['structureEvidence']['frames'] if not 10<=f['time']<12]
    reports.append(('missing_coverage',r))
    for mode in ('same_label','silence','gap','short_anchor','weak_anchor'):
        r=synthetic()
        if mode in ('same_label','silence'):
            label='bridge' if mode=='same_label' else 'silence'
            r['contextEvidence']['sections'][1]['label']=label;r['structureEvidence']['assembledSections'][1]['label']=label
        elif mode=='gap':
            for field in ('assembledSections','appleBoundaries'):r['structureEvidence'][field][1]['start']=10.5
            r['contextEvidence']['sections'][1]['start']=10.5
        elif mode=='short_anchor':
            for field in ('assembledSections','appleBoundaries'):r['structureEvidence'][field][0]['start']=8.
            r['contextEvidence']['sections'][0]['start']=8.
        else:
            for f in r['structureEvidence']['frames']:
                if 6<=f['time']<8:f['probabilities']=[.125]*8
        reports.append((mode,r))
    root=Path(__file__).resolve().parents[1]
    path=root/'experiments/structure/review_inputs/analysis-66801F14-8157-4205-B07E-B18344B96E2B.json'
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='b133f9d71d30de4bca15586da49a6dffa3638e64da94517d16e4c9361991b23e':
        raise ValueError('The uploaded reference report changed.')
    reports.append(('uploaded_who_you_say_i_am',json.loads(raw)))
    output=[]
    for name,report in reports:
        sections,frames,_,_=validate(report);e=report['structureEvidence']
        output.append(dict(name=name,sections=sections,frames=frames,
            grid=dict(bpm=e['bpm'],beatsPerBar=e['beatsPerBar'],downbeatOffset=e['downbeatOffset']),expected=expected_result(report)))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fixture=cases();a.output.write_text(json.dumps(fixture,allow_nan=False)+'\n')
    print(f'Python reference prepared: {len(fixture)} cases, including the uploaded app report.')


if __name__=='__main__':main()
