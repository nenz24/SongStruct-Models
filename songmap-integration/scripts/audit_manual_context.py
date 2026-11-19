#!/usr/bin/env python3
"""Audit frozen context errors against existing eligible manual train/val targets; no fitting or encoders."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

LABELS=('intro','verse','chorus','bridge','inst','outro','silence','pre-chorus')
VOCAL={'verse','chorus','bridge','pre-chorus'}


def validate_records(records):
    if not records or any(not r['approved'] or r['split'] not in ('train','val') for r in records):
        raise ValueError('Only approved train/val records are allowed.')
    if len({r['track_id'] for r in records})!=len(records):raise ValueError('Duplicate track.')
    for key in ('track_id','group_id','audio_sha256'):
        train={r[key] for r in records if r['split']=='train'}
        val={r[key] for r in records if r['split']=='val'}
        if train & val:raise ValueError('Train/validation leakage: '+key)
    for r in records:
        if not math.isfinite(r['duration']) or r['duration']<=0:raise ValueError('Invalid song duration.')
        if not r['sections']:raise ValueError('No eligible manual targets in an audit record.')
        previous=0.
        for s in r['sections']:
            if (not math.isfinite(s['start']+s['end']) or not previous-1e-6<=s['start']<s['end']<=r['duration']+1e-6
                    or s['label'] not in LABELS):raise ValueError('Invalid eligible manual section.')
            previous=s['end']


def validate_details(records,details):
    validate_records(records)
    targets={(r['track_id'],s['start'],s['end']):s['label'] for r in records for s in r['sections']}
    found={}
    for item in details:
        key=item['track_id'],item['start'],item['end'];p=item['probabilities']
        if key in found or targets.get(key)!=item['label_manual']:raise ValueError('Manual target/prediction inventory changed.')
        if (set(p)!=set(LABELS) or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in p.values())
                or abs(sum(p.values())-1)>1e-4 or item['prediction']!=max(LABELS,key=lambda label:p[label])):
            raise ValueError('Invalid predicted probabilities/argmax.')
        found[key]=item
    if set(found)!=set(targets):raise ValueError('Incomplete eligible prediction inventory.')
    return found


def score(details):
    matrix=[[0]*8 for _ in LABELS];duration=correct_duration=nll=0.
    for item in details:
        i=LABELS.index(item['label_manual']);j=LABELS.index(item['prediction']);matrix[i][j]+=1
        length=item['end']-item['start'];duration+=length
        if i==j:correct_duration+=length
        nll-=math.log(max(item['probabilities'][item['label_manual']],1e-8))
    classes={};f1=[]
    for i,label in enumerate(LABELS):
        support=sum(matrix[i]);predicted=sum(row[i] for row in matrix);tp=matrix[i][i]
        value=2*tp/(support+predicted) if support+predicted else 0.
        classes[label]=dict(support=support,precision=tp/predicted if predicted else 0.,recall=tp/support if support else 0.,f1=value)
        if support:f1.append(value)
    return dict(sections=len(details),section_accuracy=sum(matrix[i][i] for i in range(8))/len(details),
        macro_f1_present=sum(f1)/len(f1),duration_weighted_accuracy=correct_duration/duration,
        section_nll=nll/len(details),per_class=classes,confusion_matrix=matrix,label_order=list(LABELS))


def pattern(truth,prediction):
    if (truth,prediction)==('bridge','chorus'):return 'bridge_to_chorus'
    if (truth,prediction)==('chorus','bridge'):return 'chorus_to_bridge'
    if truth in VOCAL and prediction=='inst':return 'manual_vocal_function_to_inst'
    if truth=='inst' and prediction in VOCAL:return 'manual_inst_to_vocal_function'
    return 'other_label_error'


def summarize(records,baseline,context):
    before=validate_details(records,baseline);after=validate_details(records,context)
    results={};cases=[]
    for split in ('train','val'):
        selected=[r for r in records if r['split']==split]
        if not selected:continue
        old=[];new=[];split_cases=[];fixed=regressed=0
        for r in selected:
            for i,s in enumerate(r['sections']):
                key=r['track_id'],s['start'],s['end'];a=before[key];b=after[key];old.append(a);new.append(b)
                ac=a['prediction']==s['label'];bc=b['prediction']==s['label']
                fixed+=not ac and bc;regressed+=ac and not bc
                if bc:continue
                p=b['probabilities'];prior=r['sections'][i-1] if i else None
                following=r['sections'][i+1] if i+1<len(r['sections']) else None
                case=dict(split=split,track_id=r['track_id'],group_id=r['group_id'],start=s['start'],end=s['end'],label=s['label'],
                    baseline_prediction=a['prediction'],context_prediction=b['prediction'],
                    original_section_index=b.get('original_section_index'),
                    duration=s['end']-s['start'],relative_position=s['start']/r['duration'],
                    pattern=pattern(s['label'],b['prediction']),model_probability=p[b['prediction']],
                    manual_class_probability=p[s['label']],wrong_minus_manual_probability=p[b['prediction']]-p[s['label']],
                    baseline_correct=ac,context_correct=bc,context_probabilities=p,
                    previous_eligible_label=prior['label'] if prior else None,
                    gap_from_previous_eligible_seconds=s['start']-prior['end'] if prior else None,
                    next_eligible_label=following['label'] if following else None,
                    gap_to_next_eligible_seconds=following['start']-s['end'] if following else None)
                split_cases.append(case)
        cases.extend(split_cases)
        position={name:Counter() for name in ('early','middle','late')}
        for b in new:
            if b['label_manual']!='bridge':continue
            duration=next(r['duration'] for r in selected if r['track_id']==b['track_id'])
            t=b['start']/duration;position['early' if t<1/3 else 'middle' if t<2/3 else 'late'][b['prediction']]+=1
        results[split]=dict(tracks=len(selected),manual_label_counts=dict(Counter(s['label'] for r in selected for s in r['sections'])),
            excluded_sections_not_targets=sum(r.get('excluded_sections',0) for r in selected),
            baseline=score(old),context=score(new),wrong_to_correct=fixed,correct_to_wrong=regressed,
            context_error_patterns=dict(Counter(c['pattern'] for c in split_cases)),
            context_confusion_pairs=[dict(manual=a,predicted=b,count=n) for (a,b),n in Counter((c['label'],c['context_prediction']) for c in split_cases).most_common()],
            affected_tracks=len({c['track_id'] for c in split_cases}),
            high_probability_errors_ge_0_8=sum(c['model_probability']>=.8 for c in split_cases),
            manual_bridge_predictions_by_position={k:dict(v) for k,v in position.items()})
    cases.sort(key=lambda c:(c['split'],-c['wrong_minus_manual_probability'],c['track_id'],c['start']))
    return dict(splits=results,error_cases=cases,manual_labels_changed=False,boundaries_changed=False,
        training_started=False,encoder_executed=False,test_predictions_used=False,
        limitations=['Train errors describe fit and do not measure generalization.',
            'Validation has already guided earlier experiments; this is not a new independent final accuracy estimate.',
            'Manual targets define class correctness here. High model probability does not prove manual annotation error.',
            'Held/excluded annotations are not targets. Neighbor labels are eligible targets; gaps are explicit.',
            'Original section times may come from Apple. Label confusion alone cannot establish bad boundaries or mixed musical content.',
            'Original-feature context predictions are separate from compressed Core ML/iPhone audio inference.'])


def require_cached_sections(rows,cache):
    missing=[r['track_id'] for r in rows if not all((cache/(r['id']+suffix)).is_file() for suffix in ('.json','.npz'))]
    if missing:raise ValueError('Existing pooled section cache required; no encoders/repooling started. Missing: '+', '.join(missing[:5]))


def predict_sample(model,sample,target):
    if sample['split'] not in ('train','val'):raise ValueError('This audit does not infer test records.')
    import torch
    from train_section_context import forward_targets
    if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('A frozen eval-mode model is required.')
    with torch.inference_mode():
        scores,_,labels=forward_targets(model,sample,target)
        loss=torch.nn.functional.cross_entropy(scores,labels,reduction='sum').item()
        probabilities=scores.softmax(-1).cpu().tolist()
    if not math.isfinite(loss) or loss<0 or len(probabilities)!=len(sample['details']):raise ValueError('Invalid frozen context output/loss.')
    produced=[]
    for base,p,index in zip(sample['details'],probabilities,sample['indices'].tolist()):
        order=sorted(range(8),key=lambda j:p[j],reverse=True)
        produced.append({**base,'prediction':LABELS[order[0]],'probabilities':dict(zip(LABELS,p)),
            'confidence':p[order[0]],'margin':p[order[0]]-p[order[1]],'original_section_index':int(index)+1})
    return produced,loss


def read_native_validation(path,rows,source,read):
    proof=read(path)
    if (proof.get('protocol')!='section-context-coreml-fp32-v1' or proof.get('split')!='val'
            or not proof.get('full_validation') or not proof.get('passed') or proof.get('selected_epoch')!=5
            or proof.get('status')!='numerically_consistent' or proof.get('precision')!='fp32'
            or proof['tracks']!=len(rows) or any(proof.get(k)!=v for k,v in source.items())):
        raise ValueError('Saved native context proof differs from the frozen validation profile.')
    checks=proof['checks'];names={r['track_id'] for r in rows}
    if len(checks)!=len(rows) or {c['track_id'] for c in checks}!=names or any(not c.get('passed') for c in checks):
        raise ValueError('Incomplete/failed native context proof.')
    produced=[];identities=set()
    for row in rows:
        cached=read(path.parent/'per-track'/(row['id']+'.json'))
        if not cached['check'].get('passed'):raise ValueError('Cached native parity failed.')
        identities.add(cached['identity']);produced.extend(cached['sections'])
    if len(identities)!=1 or not next(iter(identities)):raise ValueError('Mixed native prediction cache identities.')
    validate_details(rows,produced);actual=score(produced);expected=proof['coreml']
    if any(actual[k]!=expected[k] for k in ('sections','confusion_matrix','label_order')) or any(
            abs(actual[k]-expected[k])>1e-10 for k in ('section_accuracy','macro_f1_present')):
        raise ValueError('Saved native metrics do not reproduce from predictions.')
    return produced


def load_frozen_inputs(root,frozen,cache,limit=0,songmap=None):
    """Read-only original train/val cache and frozen teacher, with the audit source gates."""
    if limit<0:raise ValueError("limit must be >=0")
    root=Path(root).expanduser().resolve();frozen=Path(frozen).expanduser().resolve()
    cache=Path(cache).expanduser().resolve();songmap=Path(songmap or Path(__file__).resolve().parents[1]).resolve()
    sys.path.insert(0,str(root))
    import torch
    from finetuning.common import config,read,write,fingerprint,sha_file,iter_features,device
    from finetuning.engine import manifest
    from evaluate_frozen_context import split_records,verify_scores
    from train_section_context import SectionContextHead,prepare_samples,PROTOCOL,COVERAGE_POLICY
    cfg=config(root/'finetuning/config.json');data,provenance=manifest(cfg);groups=split_records(data)
    all_rows=groups['train']+groups['val'];validate_records(all_rows)
    if any(r['annotation_sha256']!=fingerprint(r['sections']) for r in all_rows):raise ValueError('Manual annotation digest changed.')
    lock=read(frozen/'selection.json');snapshot=frozen/'frozen-context.pt';checkpoint_sha=sha_file(snapshot)
    if (lock.get('protocol')!='frozen-section-context-test-v1' or lock['identity']!=fingerprint({k:v for k,v in lock.items() if k!='identity'})
            or lock['context_epoch']!=5 or lock['context_checkpoint_sha256']!=checkpoint_sha
            or lock['plan_id']!=data['plan_id'] or lock['provenance_id']!=provenance
            or lock['feature_settings']!=cfg['features'] or lock['torch']!=str(torch.__version__) or lock['coverage_policy']!=COVERAGE_POLICY):
        raise ValueError('Frozen context/model/data/runtime identity differs.')
    run_path=next((q for q in (songmap/'outputs.local').glob('section-context-*/run.json') if sha_file(q)==lock['training_run_sha256']),None)
    if run_path is None:raise ValueError('Exact original context run.json is missing.')
    run=read(run_path);selected_path=run_path.parent/'comparison.json';selected=read(selected_path)
    if (run['identity']!=fingerprint({k:v for k,v in run.items() if k!='identity'})
            or sha_file(selected_path)!=lock['selected_validation_report_sha256'] or selected['best_epoch']!=5
            or run['protocol']!=PROTOCOL or selected['experiment_identity']!=run['identity']
            or selected.get('split')!='val' or not selected.get('full_validation') or selected['tracks']!=len(groups['val'])):
        raise ValueError('Original selected run changed.')
    expected_samples=dict(run['sample_identities'])
    if set(expected_samples)!={r['track_id'] for r in all_rows}:raise ValueError('Original train/val sample inventory changed.')
    calibration_path=next((q for q in (songmap/'outputs.local').glob('bridge-calibration-*/comparison.json') if sha_file(q)==lock['calibration_report_sha256']),None)
    if calibration_path is None:raise ValueError('Exact original calibration report is missing.')
    calibration=read(calibration_path);train_audit_path=Path(calibration['train_audit']).expanduser().resolve();train_audit=read(train_audit_path)
    if (calibration['checkpoint_sha256']!=lock['checkpoint_sha256'] or calibration['plan_id']!=data['plan_id']
            or calibration['provenance_id']!=provenance or run['calibration_report_sha256']!=sha_file(calibration_path)
            or not calibration.get('full_validation') or calibration.get('split')!='val'
            or train_audit.get('split')!='train' or not train_audit.get('full_train')
            or train_audit['checkpoint_sha256']!=lock['checkpoint_sha256']
            or train_audit['plan_id']!=data['plan_id'] or train_audit['provenance_id']!=provenance):
        raise ValueError('Original head probability source changed.')
    baseline_all=[]
    for split in ('train','val'):
        for row in groups[split]:
            path=(train_audit_path.parent/'per-track' if split=='train' else calibration_path.parent/'val-predictions')/(row['id']+'.json')
            item=read(path);expected=train_audit['audit_identity'] if split=='train' else calibration['experiment_identity']
            if item['identity']!=expected or (split=='train' and item['audio_sha256']!=row['audio_sha256']):
                raise ValueError('Original probability cache identity changed.')
            baseline_all.extend(item['sections'])
    validate_details(all_rows,baseline_all)
    # No baseline inference is rerun. Existing run/cache identities below bind these exact probabilities.
    rows=[r for split in ('train','val') for r in (groups[split][:limit] if limit else groups[split])]
    require_cached_sections(rows,cache)
    known={r['track_id'] for r in rows};baseline=[s for s in baseline_all if s['track_id'] in known]
    helpers=read,write,fingerprint,sha_file,iter_features
    samples=prepare_samples(rows,baseline,cfg,provenance,cache,helpers)
    if any(s['identity']!=expected_samples[s['track_id']] for s in samples):raise ValueError('Pooled cache/sample identity differs from selected training run.')
    state=torch.load(snapshot,map_location='cpu',weights_only=True)
    if (state['identity']!=run['identity'] or state['architecture']!=PROTOCOL or state['settings']!=run['settings']
            or state['best_epoch']!=5 or state['plan_id']!=data['plan_id'] or state['provenance_id']!=provenance):
        raise ValueError('Frozen context weights differ from run.')
    target=device(run['device']);torch.set_num_threads(2)
    model=SectionContextHead(run['settings']['hidden'],run['settings']['dropout']).to(target)
    model.load_state_dict(state['model'],strict=True);model.eval();model.requires_grad_(False);del state
    return {key:value for key,value in locals().items() if key in (
        "root","frozen","cache","songmap","cfg","data","provenance","groups","all_rows","rows",
        "baseline","samples","model","target","lock","run","selected","run_path",
        "calibration_path","snapshot","checkpoint_sha")}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--songstruct-root',type=Path,default=Path(__file__).resolve().parents[2]/'Section-Annotations')
    p.add_argument('--frozen-evaluation',type=Path,required=True);p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if a.limit<0:raise ValueError('limit must be >=0')
    inputs=load_frozen_inputs(a.songstruct_root,a.frozen_evaluation,a.cache_dir,a.limit)
    from finetuning.common import read,write,fingerprint,sha_file
    from evaluate_frozen_context import verify_scores
    import torch
    root,frozen,cache,songmap,cfg,data,provenance,groups,all_rows,rows,baseline,samples,model,target,lock,run,selected,run_path,calibration_path,snapshot,checkpoint_sha=(
        inputs[k] for k in ("root","frozen","cache","songmap","cfg","data","provenance","groups","all_rows","rows",
        "baseline","samples","model","target","lock","run","selected","run_path","calibration_path","snapshot","checkpoint_sha"))
    out=a.output_dir.expanduser().resolve()
    identity=fingerprint(dict(protocol='manual-context-audit-v1',context_checkpoint_sha256=checkpoint_sha,
        plan_id=data['plan_id'],provenance_id=provenance,run_identity=run['identity'],device=str(target),torch=str(torch.__version__)))
    out.mkdir(parents=True,exist_ok=True);prediction_cache=out/'per-track';prediction_cache.mkdir(exist_ok=True)
    selection=dict(identity=identity,context_checkpoint_sha256=checkpoint_sha,all_samples=run['sample_identities'])
    if (out/'selection.json').exists() and read(out/'selection.json')!=selection:raise ValueError('Audit inputs changed; choose another output directory.')
    write(out/'selection.json',selection);details=[];losses={'train':0.,'val':0.}
    print(f'Frozen context epoch 5; {len(rows)} train/val tracks; {target}. No encoders, old head or optimizer.',flush=True)
    for i,(row,sample) in enumerate(zip(rows,samples),1):
        path=prediction_cache/(row['id']+'.json')
        print(f'Context audit [{i}/{len(rows)}] {row["split"]}: {row["track_id"]}',flush=True)
        if path.exists():
            item=read(path)
            if (item['identity']!=identity or item['sample_identity']!=sample['identity']
                    or item.get('payload_sha256')!=fingerprint({k:v for k,v in item.items() if k!='payload_sha256'})):
                raise ValueError('Stale or modified per-track audit cache.')
        else:
            produced,loss=predict_sample(model,sample,target)
            validate_details([row],produced)
            item=dict(identity=identity,sample_identity=sample['identity'],sections=produced,nll_sum=loss)
            item['payload_sha256']=fingerprint(item);write(path,item)
        validate_details([row],item['sections'])
        if not math.isfinite(item['nll_sum']) or item['nll_sum']<0:raise ValueError('Invalid cached loss.')
        details.extend(item['sections']);losses[row['split']]+=item['nll_sum']
    result=summarize(rows,baseline,details)
    for split,entry in result['splits'].items():entry['context']['section_nll']=losses[split]/entry['context']['sections']
    full=len(rows)==len(all_rows);reproduced=False;failure=None
    if full:
        try:
            verify_scores(result['splits']['val']['baseline'],selected['baseline'])
            verify_scores(result['splits']['val']['context'],selected['candidate']);reproduced=True
        except ValueError as error:failure=str(error)
    result['saved_native_palette6_validation']={'status':'not_available_or_not_full_audit',
        'scope':'Optional saved Core ML context predictions on palette6 features; no native inference is executed.'}
    if full and reproduced:
        expected_source=dict(context_checkpoint_sha256=checkpoint_sha,plan_id=data['plan_id'],provenance_id=provenance)
        available=[]
        for path in (songmap/'outputs.local').glob('context-coreml-fp32-*/parity/validation-all.json'):
            proof=read(path)
            if proof.get('passed') and proof.get('full_validation') and all(proof.get(k)==v for k,v in expected_source.items()):available.append(path)
        if available:
            path=max(available,key=lambda p:p.stat().st_mtime);native=read_native_validation(path,groups['val'],expected_source,read)
            val_ids={r['track_id'] for r in groups['val']};original=[s for s in details if s['track_id'] in val_ids]
            comparison=summarize(groups['val'],original,native)
            result['saved_native_palette6_validation']=dict(status='saved_metrics_reproduced',source_report_sha256=sha_file(path),
                original_feature_context=comparison['splits']['val']['baseline'],
                saved_native_context=comparison['splits']['val']['context'],
                wrong_to_correct=comparison['splits']['val']['wrong_to_correct'],correct_to_wrong=comparison['splits']['val']['correct_to_wrong'],
                error_patterns=comparison['splits']['val']['context_error_patterns'],error_cases=comparison['error_cases'],
                scope='Saved CPU Core ML context with palette6 features; differs from original training encoders. Not an iPhone audio benchmark.')
    if sha_file(snapshot)!=checkpoint_sha:raise ValueError('Frozen model changed during audit.')
    result.update(protocol='manual-context-audit-v1',audit_identity=identity,full_train_val=full,
        original_context_validation_reproduced=reproduced,validation_reproduction_error=failure,
        context_checkpoint_sha256=checkpoint_sha,plan_id=data['plan_id'],provenance_id=provenance,
        sample_profile='original training encoders + frozen PyTorch context; no compressed/iPhone claim',
        source_run_sha256=sha_file(run_path),source_calibration_sha256=sha_file(calibration_path),
        feature_cache_read_only=True,weights_updated=False,ready_for_app_integration=False)
    destination=out/('audit-all.json' if full else 'audit-smoke.json');write(destination,result)
    fields=['split','track_id','group_id','start','end','label','baseline_prediction','context_prediction','pattern',
            'model_probability','manual_class_probability','wrong_minus_manual_probability','baseline_correct','original_section_index']
    with (out/('errors-all.csv' if full else 'errors-smoke.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:c[k] for k in fields} for c in result['error_cases'])
    for split,entry in result['splits'].items():
        print(f'{split}: {entry["tracks"]} tracks; context acc={entry["context"]["section_accuracy"]:.4f}, F1={entry["context"]["macro_f1_present"]:.4f}; errors={entry["context_error_patterns"]}',flush=True)
    print('Report:',destination,flush=True)
    if failure:raise RuntimeError('Selected validation did not reproduce: '+failure)


if __name__=='__main__':main()
