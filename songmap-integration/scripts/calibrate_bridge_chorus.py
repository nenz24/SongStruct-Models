#!/usr/bin/env python3
"""Train-only, group-CV selected bridge/chorus decision adapter; evaluate once on frozen validation."""
import argparse
import hashlib
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_train_bridge import LABELS, summarize


def features(item,duration):
    p=item['probabilities']
    if (not all(math.isfinite(v) for v in (duration,item['start'],item['end']))
            or not 0<=item['start']<item['end']<=duration):raise ValueError('Invalid section timing.')
    return [math.log(max(p['bridge'],1e-8)/max(p['chorus'],1e-8)),
            item['start']/duration,math.log(item['end']-item['start'])]


def fit(items,metadata,regularization):
    import numpy as np
    if not math.isfinite(regularization) or regularization<=0:raise ValueError('Regularization must be finite and positive.')
    if any(r['split']!='train' or not r['approved'] for r in metadata.values()):
        raise ValueError('Only approved train metadata can fit calibration.')
    eligible=[i for i in items if i['label_manual'] in ('bridge','chorus') and i['prediction'] in ('bridge','chorus')]
    if {i['label_manual'] for i in eligible}!={'bridge','chorus'}:raise ValueError('Training fold needs both manual pair classes.')
    raw=np.asarray([features(i,metadata[i['track_id']]['duration']) for i in eligible],dtype=np.float64)
    y=np.asarray([i['label_manual']=='bridge' for i in eligible],dtype=np.float64)
    mean=raw.mean(0);scale=raw.std(0);scale[scale<1e-8]=1
    x=np.column_stack([np.ones(len(raw)),(raw-mean)/scale]);w=np.zeros(4)
    def objective(weights):
        z=raw[:,0]+x@weights
        return float(np.mean(np.logaddexp(0,z)-y*z)+regularization*.5*(weights@weights))
    for _ in range(100):
        z=raw[:,0]+x@w;q=1/(1+np.exp(-np.clip(z,-50,50)))
        gradient=x.T@(q-y)/len(y)+regularization*w
        if np.max(np.abs(gradient))<1e-8:break
        hessian=(x.T*(q*(1-q)))@x/len(y)+regularization*np.eye(4)
        step=np.linalg.solve(hessian,gradient);old=objective(w);rate=1.
        while rate>1e-8:
            candidate=w-rate*step
            # Keep the adapter monotonic in the original bridge/chorus odds.
            candidate[1]=max(candidate[1],-scale[0])
            if objective(candidate)<=old+1e-12:break
            rate*=.5
        if rate<=1e-8:break
        if np.max(np.abs(candidate-w))<1e-8:w=candidate;break
        w=candidate
    return {'kind':'pair_gate_log_odds_correction','weights':w.tolist(),'mean':mean.tolist(),'scale':scale.tolist(),
        'regularization':regularization,'fit_sections':len(eligible),
        'features':['log_bridge_over_chorus','relative_start','log_duration_seconds']}


def predict(item,metadata,model):
    baseline=item['prediction']
    if model['kind']=='identity' or baseline not in ('bridge','chorus'):return baseline,None
    raw=features(item,metadata[item['track_id']]['duration'])
    x=[1.]+[(v-m)/s for v,m,s in zip(raw,model['mean'],model['scale'])]
    z=raw[0]+sum(w*v for w,v in zip(model['weights'],x))
    q=1/(1+math.exp(-max(-50,min(50,z))))
    return ('bridge' if q>.5 else 'chorus'),q


def metrics(items,metadata,model):
    matrix=[[0]*8 for _ in range(8)]
    for item in items:
        pred,_=predict(item,metadata,model)
        matrix[LABELS.index(item['label_manual'])][LABELS.index(pred)]+=1
    classes={};f1=[]
    for i,name in enumerate(LABELS):
        support=sum(matrix[i]);n_pred=sum(row[i] for row in matrix);tp=matrix[i][i]
        value=2*tp/(support+n_pred) if support+n_pred else 0
        classes[name]={'support':support,'precision':tp/n_pred if n_pred else 0,'recall':tp/support if support else 0,'f1':value}
        if support:f1.append(value)
    return {'sections':len(items),'section_accuracy':sum(matrix[i][i] for i in range(8))/len(items),
        'macro_f1_present':sum(f1)/len(f1),'per_class':classes,'confusion_matrix':matrix,'label_order':list(LABELS)}


def select_train_only(records,items):
    metadata={r['track_id']:r for r in records}
    if not records or not items:raise ValueError('No training targets.')
    if any(r['split']!='train' or not r['approved'] for r in records):raise ValueError('Only approved train records can fit/select calibration.')
    summarize(records,items)
    groups=sorted({r['group_id'] for r in records},key=lambda g:hashlib.sha256(('bridge-calibration-v1:'+g).encode()).hexdigest())
    if len(groups)<3:raise ValueError('At least three training groups required.')
    folds={g:i%3 for i,g in enumerate(groups)}
    choices=[('identity',None),('ridge_0.01',.01),('ridge_0.1',.1),('ridge_1',1.)]
    cv=[]
    for name,reg in choices:
        oof=[]
        for fold in range(3):
            held=[i for i in items if folds[metadata[i['track_id']]['group_id']]==fold]
            training=[i for i in items if folds[metadata[i['track_id']]['group_id']]!=fold]
            model={'kind':'identity'} if reg is None else fit(training,metadata,reg)
            for item in held:
                predicted,_=predict(item,metadata,model)
                oof.append({**item,'prediction':predicted})
        score=metrics(oof,metadata,{'kind':'identity'})
        cv.append({'candidate':name,'regularization':reg,'out_of_fold_metrics':score})
    # Identity wins ties. No validation values enter this function.
    winner=0
    for i in range(1,len(cv)):
        if cv[i]['out_of_fold_metrics']['macro_f1_present']>cv[winner]['out_of_fold_metrics']['macro_f1_present']+1e-12:winner=i
    chosen=cv[winner]
    model={'kind':'identity'} if chosen['regularization'] is None else fit(items,metadata,chosen['regularization'])
    return model,{'selection_split':'train_group_out_of_fold','folds':3,'group_folds':folds,'selected':chosen['candidate'],'candidates':cv}


def evaluate_validation(records,items,model):
    if not records or not items:raise ValueError('No validation targets.')
    if any(r['split']!='val' or not r['approved'] for r in records):raise ValueError('Expected validation records only.')
    # Reuse exact target validation without allowing a fitting path to see holdout records.
    summarize([{**r,'split':'train'} for r in records],items)
    metadata={r['track_id']:r for r in records}
    return metrics(items,metadata,model)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--songstruct-root',type=Path,default=Path(__file__).resolve().parents[2]/'Section-Annotations')
    p.add_argument('--train-audit',type=Path)
    p.add_argument('--device',choices=['mps','cpu','cuda'],default='mps')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();root=args.songstruct_root.expanduser().resolve()
    if not (root/'finetuning/config.json').is_file():raise ValueError('SongStruct root missing.')
    sys.path.insert(0,str(root))
    from finetuning.common import config,read,write,fingerprint,sha_file,device,load_head
    from finetuning.engine import manifest,evaluate
    import torch
    cfg=config(root/'finetuning/config.json');data,provenance=manifest(cfg)
    train=[r for r in data['records'] if r['approved'] and r['split']=='train']
    val=[r for r in data['records'] if r['approved'] and r['split']=='val']
    audit_path=args.train_audit
    if audit_path is None:
        candidates=list((Path(__file__).resolve().parents[1]/'outputs.local').glob('train-bridge-audit-*/train-all.json'))
        if not candidates:raise ValueError('Full train audit missing; supply --train-audit.')
        audit_path=max(candidates,key=lambda p:p.stat().st_mtime)
    audit_path=audit_path.expanduser().resolve();audit=read(audit_path)
    if not audit.get('full_train') or audit.get('split')!='train' or audit['tracks']!=len(train):raise ValueError('Expected complete train-only audit.')
    if audit['plan_id']!=data['plan_id'] or audit['provenance_id']!=provenance:raise ValueError('Frozen plan/provenance changed.')
    checkpoint=Path(audit['checkpoint']).resolve()
    if sha_file(checkpoint)!=audit['checkpoint_sha256']:raise ValueError('Checkpoint changed since audit.')
    target=device(args.device)
    expected=fingerprint({'protocol':'train-bridge-audit-v1','device':str(target),'torch':str(torch.__version__),
        'plan_id':data['plan_id'],'provenance_id':provenance,'checkpoint_sha256':sha_file(checkpoint),
        'train_annotations':[(r['id'],r['annotation_sha256']) for r in train]})
    if expected!=audit['audit_identity']:raise ValueError('Train audit runtime/device/annotations differ; use its environment.')
    details=[]
    for row in train:
        stored=read(audit_path.parent/'per-track'/(row['id']+'.json'))
        if stored['identity']!=expected or stored['audio_sha256']!=row['audio_sha256']:raise ValueError('Train prediction cache stale.')
        details.extend(stored['sections'])
    summarize(train,details)
    out=args.output_dir.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'comparison.json').exists():raise ValueError('Completed experiment exists; choose a new output directory.')
    print('Selecting adapter using group-held-out TRAIN predictions only...',flush=True)
    adapter,selection=select_train_only(train,details)
    identity=fingerprint({'protocol':'pair-calibration-v1','train_audit_sha256':sha_file(audit_path),
        'checkpoint_sha256':sha_file(checkpoint),'plan_id':data['plan_id'],'provenance_id':provenance,
        'val_annotations':[(r['id'],r['annotation_sha256']) for r in val],
        'selected_adapter':adapter,'device':str(target),'torch':str(torch.__version__)})
    write(out/'adapter.json',{'adapter':adapter,'selection':selection,'identity':identity,
        'checkpoint_sha256':sha_file(checkpoint),'plan_id':data['plan_id'],'scope':'Post-section binary decision adapter; not frame probability calibration or SongFormer fine-tuning.',
        'ready_for_app_integration':False})
    cache=out/'val-predictions';cache.mkdir(exist_ok=True)
    stored_val={};pending=[]
    for row in val:
        path=cache/(row['id']+'.json')
        if path.exists():
            stored=read(path)
            if stored['identity']!=identity:raise ValueError('Validation cache stale.')
            evaluate_validation([row],stored['sections'],{'kind':'identity'})
            stored_val[row['track_id']]=stored['sections']
        else:pending.append(row)
    if pending:
        torch.set_num_threads(2)
        model,_=load_head(cfg,target,checkpoint)
        for i,row in enumerate(pending,1):
            print(f'Validation head [{i}/{len(pending)}]: {row["track_id"]}',flush=True)
            section_details=[];evaluate(model,[row],cfg,target,provenance,details=section_details)
            evaluate_validation([row],section_details,{'kind':'identity'})
            write(cache/(row['id']+'.json'),{'identity':identity,'sections':section_details})
            stored_val[row['track_id']]=section_details
    val_details=[s for r in val for s in stored_val[r['track_id']]]
    baseline=evaluate_validation(val,val_details,{'kind':'identity'});candidate=evaluate_validation(val,val_details,adapter)
    val_meta={r['track_id']:r for r in val};changes=[]
    for item in val_details:
        pred,q=predict(item,val_meta,adapter)
        if pred!=item['prediction']:
            changes.append({k:item[k] for k in ('track_id','start','end','label_manual','prediction')})
            changes[-1].update(candidate_prediction=pred,binary_bridge_probability=q,
                baseline_correct=item['prediction']==item['label_manual'],candidate_correct=pred==item['label_manual'])
    report={'split':'val','tracks':len(val),'full_validation':True,
        'plan_id':data['plan_id'],'provenance_id':provenance,'checkpoint':str(checkpoint),
        'checkpoint_sha256':sha_file(checkpoint),'train_audit':str(audit_path),
        'adapter_selected_by_train_only':selection['selected'],
        'selection':selection,'baseline':baseline,'candidate':candidate,
        'macro_f1_delta':candidate['macro_f1_present']-baseline['macro_f1_present'],
        'bridge_f1_delta':candidate['per_class']['bridge']['f1']-baseline['per_class']['bridge']['f1'],
        'chorus_f1_delta':candidate['per_class']['chorus']['f1']-baseline['per_class']['chorus']['f1'],
        'changed_sections':changes,'experiment_identity':identity,'test_used':False,'encoder_executed':False,
        'songformer_weights_changed':False,'manual_labels_changed':False,'maps_changed':False,'ready_for_app_integration':False,
        'limitations':['Selection fits and scores folds inside train; validation only evaluates the selected adapter and identity.',
            'Frozen SongFormer was trained on all train groups, so internal folds only hold out adapter fitting, not backbone training.',
            'This is post-pooling pair decision calibration on original PyTorch feature caches, not the compressed app pipeline.',
            'Binary probability is conditional on bridge/chorus, not a normalized eight-class probability vector.',
            'Classes with baseline predictions outside the pair are unchanged. Other-class precision still needs review.']}
    write(out/'comparison.json',report)
    print('Selected:',selection['selected'],'| val macro F1 delta:',report['macro_f1_delta'])
    print('Report:',out/'comparison.json')


if __name__=='__main__':main()
