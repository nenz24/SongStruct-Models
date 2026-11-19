#!/usr/bin/env python3
"""Assess diagnostic boundary proposals on frozen validation features; no encoder execution or training."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import platform
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_section_range import audit
from review_structure_quality import propose


def refinement(report,proposals):
    """One conservative split per original Apple section; preserve all other ranges."""
    result=[]
    for section in report['structureEvidence']['assembledSections']:
        inside=sorted(p['suggested_split_seconds'] for p in proposals
            if p['parent_section']['start']==section['start'] and p['parent_section']['end']==section['end'])
        if not inside:
            result.append(dict(section));continue
        cuts=[section['start']]+inside[:1]+[section['end']]
        for start,end in zip(cuts,cuts[1:]):
            prediction=audit(report,start,end)['pooled_prediction']
            if prediction is None:raise ValueError('No feature coverage in proposed section.')
            result.append({'start':start,'end':end,'label':prediction})
    return result


def score(reference,sections,labels):
    matrix=[[0]*len(labels) for _ in labels];correct_duration=duration_total=0.;coverages=[]
    for truth in reference:
        start,end=truth['start'],truth['end'];duration=end-start
        if not duration>0:raise ValueError('Invalid reference section.')
        votes=Counter()
        for section in sections:
            if section['label'] not in labels:continue
            overlap=max(0,min(end,section['end'])-max(start,section['start']))
            if overlap:votes[section['label']]+=overlap
        covered=sum(votes.values());coverages.append(covered/duration)
        if covered<=0:raise ValueError('Reference section has no predicted range coverage.')
        prediction=max(labels,key=lambda label:votes[label])
        matrix[labels.index(truth['label'])][labels.index(prediction)]+=1
        correct_duration+=votes[truth['label']];duration_total+=duration
    present_f1=[]
    for i in range(len(labels)):
        support=sum(matrix[i]);predicted=sum(row[i] for row in matrix)
        if support:present_f1.append(2*matrix[i][i]/(support+predicted))
    return {'eligible_reference_sections':len(reference),
        'manual_section_duration_vote_accuracy':sum(matrix[i][i] for i in range(len(labels)))/len(reference),
        'manual_section_duration_vote_macro_f1':sum(present_f1)/len(present_f1),
        'eligible_duration_label_agreement':correct_duration/duration_total,
        'reference_range_low_coverage_sections':sum(c<.98 for c in coverages),'confusion_matrix':matrix}


def overall(results,labels):
    # Aggregate weighted quantities from per-track sufficient statistics.
    matrix=[[sum(r['confusion_matrix'][i][j] for r in results) for j in range(len(labels))] for i in range(len(labels))]
    count=sum(sum(row) for row in matrix)
    f1=[]
    for i in range(len(labels)):
        support=sum(matrix[i]);predicted=sum(row[i] for row in matrix)
        if support:f1.append(2*matrix[i][i]/(support+predicted))
    duration=sum(r['eligible_duration_seconds'] for r in results)
    return {'eligible_reference_sections':count,'manual_section_duration_vote_accuracy':sum(matrix[i][i] for i in range(len(labels)))/count,
        'manual_section_duration_vote_macro_f1':sum(f1)/len(f1),
        'eligible_duration_label_agreement':sum(r['eligible_duration_label_agreement']*r['eligible_duration_seconds'] for r in results)/duration,
        'reference_range_low_coverage_sections':sum(r['reference_range_low_coverage_sections'] for r in results),
        'confusion_matrix':matrix,'label_order':labels}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--songstruct-root',type=Path,default=Path(__file__).resolve().parents[2]/'Section-Annotations')
    p.add_argument('--validation-report',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--limit',type=int,default=1,help='0=all validation')
    a=p.parse_args()
    if platform.system()!='Darwin':raise RuntimeError('Run cached-feature Core ML head prediction on macOS.')
    if a.limit<0:raise ValueError('limit must be >=0')
    root=a.songstruct_root.expanduser().resolve()
    if not (root/'finetuning/config.json').is_file():raise ValueError('SongStruct root missing; supply --songstruct-root.')
    sys.path.insert(0,str(root))
    import numpy as np
    import torch
    import coremltools as ct
    from finetuning.common import config,read,write,fingerprint,sha_file,LABELS,CLASS_IDS
    from finetuning.engine import manifest
    from deployment.evaluate_pipeline import cached_features
    from deployment.coreml_features import native_input
    from deployment.compress_muq import package_digest
    torch.set_num_threads(2)
    cfg=config(root/'finetuning/config.json');data,provenance=manifest(cfg)
    report_path=a.validation_report
    if report_path is None:
        found=list((root/'outputs.local').glob('muq-compression-*/palette6-all-validation.json'))
        if not found:raise ValueError('Palette6 full-validation report not found.')
        report_path=max(found,key=lambda p:p.stat().st_mtime)
    proof=read(report_path)
    if not proof.get('full_validation') or proof.get('muq_scheme')!='palette6' or proof.get('head_scheme')!='palette6':
        raise ValueError('Expected full validation of palette6 MuQ, MusicFM and head.')
    rows=[r for r in data['records'] if r['approved'] and r['split']=='val']
    if (proof['tracks']!=len(rows) or proof['plan_id']!=data['plan_id'] or proof['provenance_id']!=provenance
            or proof['annotations_id']!=fingerprint([(r['track_id'],r['sections']) for r in rows])):
        raise ValueError('Frozen validation inventory/labels changed.')
    if a.limit:rows=rows[:a.limit]
    if not rows:raise ValueError('No validation tracks.')
    baseline_path=Path(proof['baseline_combination'])
    if sha_file(baseline_path)!=proof['baseline_combination_sha256']:raise ValueError('Baseline report changed.')
    baseline=read(baseline_path)
    package=Path(baseline['head_package'])
    cache=Path(proof['diagnostic_cache']);identity=proof['candidate_encoder_id']
    meta=read(cache/'encoder_provenance.json')
    if meta['id']!=identity or fingerprint({k:v for k,v in meta.items() if k!='id'})!=identity:raise ValueError('Feature cache identity invalid.')
    model=ct.models.MLModel(str(package),compute_units=ct.ComputeUnit.CPU_ONLY)
    if model.user_defined_metadata.get('checkpoint_sha256')!=proof['checkpoint_sha256'] or model.user_defined_metadata.get('compression_scheme')!='palette6':
        raise ValueError('Head package differs from validation.')
    head_digest=package_digest(package)
    out=a.output_dir.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    evidence_dir=out/'frame-evidence';evidence_dir.mkdir(exist_ok=True)
    previous={(s['track_id'],s['start'],s['end']):s['prediction'] for s in proof['sections']}
    track_results=[];baseline_metrics=[];candidate_metrics=[];baseline_mismatches=[]
    for index,row in enumerate(rows,1):
        print(f'Cached head [{index}/{len(rows)}]: {row["track_id"]}',flush=True)
        original=read(row['source_json'])
        if original['track_id']!=row['track_id']:raise ValueError('Original Apple source track mismatch.')
        ranges=[]
        for section in original['sections']:
            start,end=float(section['start']),float(section['end'])
            if (not math.isfinite(start) or not math.isfinite(end) or not 0<=start<end
                    or end>row['duration']+.02):raise ValueError('Original section range invalid.')
            ranges.append({'start':start,'end':min(end,row['duration'])})
        if any(b['start']<a['end']-1e-5 for a,b in zip(ranges,ranges[1:])):raise ValueError('Overlapping original Apple sections.')
        cache_key=fingerprint({'protocol':'boundary-frame-evidence-v1','head':head_digest,'encoder':identity,
            'audio':row['audio_sha256'],'source':sha_file(row['source_json']),'targets':row['annotation_sha256']})
        frame_path=evidence_dir/(row['id']+'.json')
        if frame_path.exists():
            stored=read(frame_path)
            if stored['cache_key']!=cache_key:raise ValueError('Evidence cache stale; choose another output directory.')
            report=stored['report']
        else:
            frames=[]
            for values,offset,_ in cached_features(cache,row,identity):
                logits=np.asarray(model.predict({'input_embeddings':native_input(values[None])})['function_logits'])
                if logits.shape!=(1,len(values)//3,128) or not np.isfinite(logits).all():raise ValueError('Invalid cached-head output.')
                probabilities=torch.from_numpy(logits[0][:,list(CLASS_IDS)].copy()).softmax(-1).tolist()
                frames.extend({'time':offset+i*.12,'duration':.12,'probabilities':p} for i,p in enumerate(probabilities))
            report={'title':row['track_id'],'structureEvidence':{'labelOrder':list(LABELS),'frames':frames,
                'appleBoundaries':ranges,'assembledSections':[]}}
            report['structureEvidence']['assembledSections']=[{**r,'label':audit(report,r['start'],r['end'])['pooled_prediction']} for r in ranges]
            write(frame_path,{'cache_key':cache_key,'report':report})
        for truth in row['sections']:
            actual=audit(report,truth['start'],truth['end'])['pooled_prediction']
            if actual!=previous[(row['track_id'],truth['start'],truth['end'])]:
                baseline_mismatches.append({'track_id':row['track_id'],'start':truth['start'],'end':truth['end'],'prediction':actual})
        proposals=propose(report)
        old_sections=report['structureEvidence']['assembledSections']
        new_sections=refinement(report,proposals['candidates'])
        before=score(row['sections'],old_sections,list(LABELS));after=score(row['sections'],new_sections,list(LABELS))
        duration=sum(s['end']-s['start'] for s in row['sections'])
        for metrics in (before,after):metrics['eligible_duration_seconds']=duration
        baseline_metrics.append(before);candidate_metrics.append(after)
        track_results.append({'track_id':row['track_id'],'source_json_sha256':sha_file(row['source_json']),
            'proposals':proposals,'simulated_new_splits':len(new_sections)-len(old_sections),'baseline':before,'candidate':after,'proposed_sections':new_sections})
    result={'tracks':len(rows),'full_validation':not a.limit or len(rows)==proof['tracks'],
        'plan_id':data['plan_id'],'provenance_id':provenance,'validation_report_sha256':sha_file(report_path),
        'head_package_digest':head_digest,'baseline_prediction_mismatches':baseline_mismatches,
        'status':'baseline_mismatch_requires_review' if baseline_mismatches else 'diagnostic_comparison_completed',
        'tracks_with_proposals':sum(bool(r['proposals']['candidates']) for r in track_results),
        'total_proposals':sum(len(r['proposals']['candidates']) for r in track_results),
        'simulated_total_new_splits':sum(r['simulated_new_splits'] for r in track_results),
        'baseline':overall(baseline_metrics,list(LABELS)),'candidate':overall(candidate_metrics,list(LABELS)),
        'track_results':track_results,'encoder_executed':False,'training_started':False,'maps_changed':False,
        'boundary_accuracy_validated':False,'ready_for_app_integration':False,
        'limitations':['Manual labels are eligible section targets; their times may share Apple boundaries. This is not independent boundary ground truth.',
            'Duration-vote section metrics differ from fixed-boundary probability-pooling metrics; compare baseline vs candidate only within this report.',
            'One split per original section; default thresholds are uncalibrated. Pending/held sections are not scored.',
            'This is Mac cached-feature head evaluation, not the complete iPhone frontend pipeline.']}
    destination=out/('validation-all.json' if not a.limit else f'validation-first-{len(rows)}.json')
    write(destination,result)
    print('Proposals:',result['total_proposals'],'| tracks:',result['tracks_with_proposals'])
    print('Baseline mismatches:',len(baseline_mismatches))
    print('Report:',destination)
    if baseline_mismatches:raise RuntimeError('Baseline did not reproduce previous predictions; inspect report before interpreting proposals.')


if __name__=='__main__':main()
