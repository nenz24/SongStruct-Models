#!/usr/bin/env python3
"""Propose boundary splits from an existing report; never edit app maps or manual labels."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_section_range import audit


def strong_runs(evidence,confidence,margin):
    labels=evidence['labelOrder'];runs=[];previous_end=None
    if len(labels)<2 or len(set(labels))!=len(labels):raise ValueError('Invalid label order.')
    for f in evidence['frames']:
        time,duration,probs=f['time'],f['duration'],f['probabilities']
        if (not math.isfinite(time) or not math.isfinite(duration) or duration<=0 or len(probs)!=len(labels)
                or any(not math.isfinite(p) or not 0<=p<=1 for p in probs) or abs(sum(probs)-1)>1e-4):
            raise ValueError('Invalid frame evidence.')
        if previous_end is not None and time<previous_end-1e-5:raise ValueError('Overlapping or unsorted frames.')
        previous_end=time+duration
        ranked=sorted(range(len(labels)),key=lambda i:probs[i],reverse=True)
        best=ranked[0]
        if probs[best]<confidence or probs[best]-probs[ranked[1]]<margin:continue
        if runs and runs[-1]['label']==labels[best] and abs(runs[-1]['end']-time)<1e-5:
            runs[-1]['end']=time+duration;runs[-1]['probability_seconds']+=probs[best]*duration
        else:runs.append({'start':time,'end':time+duration,'label':labels[best],'probability_seconds':probs[best]*duration})
    return runs


def propose(report,confidence=.8,margin=.35,sustain=3.,minimum_piece=4.,maximum_transition_gap=12.):
    values=[confidence,margin,sustain,minimum_piece,maximum_transition_gap]
    if any(not math.isfinite(v) for v in values) or not 0<=confidence<=1 or not 0<=margin<=1 or min(values[2:])<=0:
        raise ValueError('Invalid proposal parameters.')
    evidence=report.get('structureEvidence')
    if not evidence or not evidence.get('frames'):raise ValueError('Saved eight-class frame evidence required.')
    runs=[r for r in strong_runs(evidence,confidence,margin) if r['end']-r['start']>=sustain-1e-5]
    candidates=[]
    # A proposal requires strong evidence for both the old and new label, plus enough time on each side.
    for before,after in zip(runs,runs[1:]):
        if before['label']==after['label'] or after['start']-before['end']>maximum_transition_gap:continue
        point=after['start']
        section=next((s for s in evidence['assembledSections'] if s['start']<point<s['end']),None)
        if not section or section['label']!=before['label']:continue
        if point-section['start']<minimum_piece or section['end']-point<minimum_piece:continue
        if min(after['end'],section['end'])-point<sustain-1e-5:continue
        left=audit(report,section['start'],point);right=audit(report,point,section['end'])
        if left['pooled_prediction']!=section['label'] or right['pooled_prediction']!=after['label']:continue
        if min(left['coverage_fraction'],right['coverage_fraction'])<.98:continue
        if right['probabilities'][after['label']]<confidence:continue
        candidates.append({'parent_section':section,'suggested_split_seconds':point,
            'transition_review_interval':[max(section['start'],before['end']),point],
            'new_label':after['label'],'sustained_evidence_seconds':after['end']-after['start'],
            'new_label_mean_probability_in_strong_run':after['probability_seconds']/(after['end']-after['start']),
            'left_prediction':left['pooled_prediction'],'right_prediction':right['pooled_prediction'],
            'left_probabilities':left['probabilities'],'right_probabilities':right['probabilities'],
            'status':'candidate_requires_review','applied':False})
    return {'title':report['title'],'parameters':{'confidence':confidence,'margin':margin,'sustain_seconds':sustain,
        'minimum_piece_seconds':minimum_piece,'maximum_transition_gap_seconds':maximum_transition_gap},
        'candidates':candidates,'maps_changed':False,'manual_labels_changed':False,
        'note':'Uncalibrated diagnostic defaults, not validated production thresholds. The split is the first confirmed strong frame; review interval includes preceding uncertainty. No bar snapping, forced outro or forced bridge.'}


def bridge_audit(payload):
    if payload.get('split')=='test':raise ValueError('Use validation for diagnosis; do not retune against test errors.')
    if not payload.get('full_validation') and payload.get('split')!='val':raise ValueError('Expected validation section predictions.')
    rows=payload.get('sections',[])
    if not rows or any('label_manual' not in r for r in rows):raise ValueError('Manual-reference section predictions required.')
    names=('intro','verse','chorus','bridge','inst','outro','silence','pre-chorus')
    cases=[];matrix={label:0 for label in names};support=0
    for r in rows:
        if r['label_manual'] not in names or r['prediction'] not in names:raise ValueError('Unknown section label.')
        if r['label_manual']=='bridge':
            support+=1;matrix[r['prediction']]+=1
        if (r['label_manual'],r['prediction']) in [('bridge','chorus'),('chorus','bridge')]:
            cases.append({k:r.get(k) for k in ('track_id','start','end','label_manual','prediction','confidence','margin','probabilities')})
    return {'reference_muq_scheme':payload.get('muq_scheme'),'reference_split':payload.get('split','validation' if payload.get('full_validation') else 'unknown'),'manual_bridge_support':support,'manual_bridge_predicted_as':matrix,'bridge_chorus_cases':cases,
        'manual_labels_changed':False,'training_started':False,
        'note':'Existing validation errors only. These cases are not new training targets or automatic relabels; select future model changes using the frozen validation protocol.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manual-sections-report',type=Path)
    a=p.parse_args()
    if a.output.exists():p.error('Choose a new output path.')
    source=a.report.expanduser().resolve();output=a.output.expanduser().resolve()
    if source==output:p.error('Output must differ from input.')
    result=propose(json.loads(source.read_text()))
    result['source_report_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    manual=a.manual_sections_report
    if manual is None:
        sibling=Path(__file__).resolve().parents[2]/'Section-Annotations/outputs.local'
        found=list(sibling.glob('muq-compression-*/palette6-all-validation.json'))
        if found:manual=max(found,key=lambda p:p.stat().st_mtime)
    if manual:
        result['validation_bridge_audit']=bridge_audit(json.loads(manual.read_text()))
        result['validation_report_sha256']=hashlib.sha256(manual.read_bytes()).hexdigest()
    else:result['validation_bridge_audit_status']='No saved validation section report found; no model was run.'
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    for c in result['candidates']:
        print(f"REVIEW {c['suggested_split_seconds']:.2f}s: {c['left_prediction']} -> {c['right_prediction']}; interval={c['transition_review_interval']}")
    print('Saved:',output)
