#!/usr/bin/env python3
"""Diagnose bridge/chorus errors on approved TRAIN targets with frozen encoder caches; no training."""
import argparse
from collections import Counter
import math
from pathlib import Path
import sys

LABELS=('intro','verse','chorus','bridge','inst','outro','silence','pre-chorus')


def summarize(records,details):
    by_id={r['track_id']:r for r in records}
    if len(by_id)!=len(records) or any(r['split']!='train' or not r['approved'] for r in records):
        raise ValueError('Only distinct approved train records may enter this audit.')
    targets={(r['track_id'],s['start'],s['end']):s['label'] for r in records for s in r['sections']}
    lookup={};matrix=[[0]*8 for _ in range(8)]
    for item in details:
        key=(item['track_id'],item['start'],item['end'])
        if key in lookup or key not in targets or item['label_manual']!=targets[key]:
            raise ValueError('Prediction is duplicate or differs from the manual train target.')
        if item['prediction'] not in LABELS:raise ValueError('Unknown predicted label.')
        probs=item['probabilities']
        if (set(probs)!=set(LABELS) or any(not math.isfinite(v) or not 0<=v<=1 for v in probs.values())
                or abs(sum(probs.values())-1)>1e-4):raise ValueError('Invalid probabilities.')
        lookup[key]=item
        matrix[LABELS.index(item['label_manual'])][LABELS.index(item['prediction'])]+=1
    if set(lookup)!=set(targets):raise ValueError('Incomplete train prediction inventory.')
    cases=[];true_bridges=[]
    for record in records:
        sections=record['sections']
        for i,s in enumerate(sections):
            item=lookup[(record['track_id'],s['start'],s['end'])]
            true,pred=item['label_manual'],item['prediction']
            if true=='bridge':true_bridges.append(item)
            if (true,pred) not in [('bridge','chorus'),('chorus','bridge')]:continue
            previous=sections[i-1] if i else None;following=sections[i+1] if i+1<len(sections) else None
            cases.append({'track_id':record['track_id'],'group_id':record['group_id'],
                'start':s['start'],'end':s['end'],'duration':s['end']-s['start'],
                'relative_track_position':s['start']/record['duration'],
                'label_manual':true,'prediction':pred,'probabilities':item['probabilities'],
                'confidence':item['probabilities'][pred],
                'wrong_minus_manual_probability':item['probabilities'][pred]-item['probabilities'][true],
                'previous_eligible_label':previous['label'] if previous else None,
                'gap_from_previous_eligible_seconds':s['start']-previous['end'] if previous else None,
                'next_eligible_label':following['label'] if following else None,
                'gap_to_next_eligible_seconds':following['start']-s['end'] if following else None,
                'coverage':item.get('coverage'),'status':'diagnostic_only_no_label_change'})
    cases.sort(key=lambda x:(-x['wrong_minus_manual_probability'],x['track_id'],x['start']))
    bins={'early_0_to_33pct':Counter(),'middle_33_to_67pct':Counter(),'late_67_to_100pct':Counter()}
    for item in true_bridges:
        position=item['start']/by_id[item['track_id']]['duration']
        bucket='early_0_to_33pct' if position<1/3 else 'middle_33_to_67pct' if position<2/3 else 'late_67_to_100pct'
        bins[bucket][item['prediction']]+=1
    return {'tracks':len(records),'eligible_sections':len(details),'manual_label_counts':dict(Counter(targets.values())),
        'manual_bridge_count':len(true_bridges),'manual_bridge_predicted_as':dict(Counter(i['prediction'] for i in true_bridges)),
        'bridge_predictions_by_relative_position':{k:dict(v) for k,v in bins.items()},
        'bridge_to_chorus_count':sum(c['label_manual']=='bridge' for c in cases),
        'chorus_to_bridge_count':sum(c['label_manual']=='chorus' for c in cases),
        'affected_tracks':len({c['track_id'] for c in cases}),
        'high_confidence_wrong_cases_ge_0_8':sum(c['confidence']>=.8 for c in cases),
        'confusion_matrix':matrix,'label_order':list(LABELS),'cases':cases,
        'manual_labels_changed':False,'training_started':False,'val_test_used':False,
        'limitations':['Training diagnostics do not measure generalization.',
            'Previous/next labels are eligible targets, not necessarily adjacent original sections; gaps are reported.',
            'High-confidence disagreement is not proof that the manual label is wrong.',
            'This uses the PyTorch checkpoint and original training feature cache, not compressed app encoders.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--songstruct-root',type=Path,default=Path(__file__).resolve().parents[2]/'Section-Annotations')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--device',choices=['auto','mps','cpu','cuda'],default='mps')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--limit',type=int,default=1,help='0=all train songs')
    args=p.parse_args()
    if args.limit<0:raise ValueError('limit must be >=0')
    root=args.songstruct_root.expanduser().resolve()
    if not (root/'finetuning/config.json').is_file():raise ValueError('SongStruct root missing.')
    sys.path.insert(0,str(root))
    from finetuning.common import config,device,load_head,read,write,sha_file,fingerprint
    from finetuning.engine import manifest,evaluate
    import torch
    cfg=config(root/'finetuning/config.json');data,provenance=manifest(cfg)
    all_rows=[r for r in data['records'] if r['approved'] and r['split']=='train']
    rows=all_rows[:args.limit] if args.limit else all_rows
    if not rows:raise ValueError('No approved train songs.')
    checkpoint=(args.checkpoint or Path(cfg['work_dir'])/'runs/worship-v3/best.pt').expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.suffix!='.pt':raise ValueError('Expected trusted local best.pt; no fallback to pretrained.')
    meta=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if (meta.get('plan_id')!=data['plan_id'] or meta.get('provenance_id')!=provenance
            or meta.get('config',{}).get('features')!=cfg['features']):
        raise ValueError('Checkpoint differs from frozen split/encoder profile.')
    del meta
    out=args.output_dir.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    cache=out/'per-track';cache.mkdir(exist_ok=True)
    target=device(args.device)
    identity=fingerprint({'protocol':'train-bridge-audit-v1','device':str(target),'torch':str(torch.__version__),'plan_id':data['plan_id'],'provenance_id':provenance,
        'checkpoint_sha256':sha_file(checkpoint),'train_annotations':[(r['id'],r['annotation_sha256']) for r in all_rows]})
    cached={};pending=[]
    for row in rows:
        path=cache/(row['id']+'.json')
        if path.exists():
            result=read(path)
            if result['identity']!=identity or result['audio_sha256']!=row['audio_sha256']:
                raise ValueError('Audit cache changed; choose another output directory.')
            summarize([row],result['sections'])
            cached[row['track_id']]=result['sections']
        else:pending.append(row)
    print('Checkpoint:',checkpoint,flush=True)
    print(f'Train only: {len(rows)} songs; {len(pending)} pending; encoders are not run.',flush=True)
    if pending:
        torch.set_num_threads(2)
        model,_=load_head(cfg,target,checkpoint)
        for index,row in enumerate(pending,1):
            print(f'Head [{index}/{len(pending)}]: {row["track_id"]}',flush=True)
            details=[];metrics=evaluate(model,[row],cfg,target,provenance,details=details)
            summarize([row],details)
            write(cache/(row['id']+'.json'),{'identity':identity,'audio_sha256':row['audio_sha256'],'sections':details,'metrics':metrics})
            cached[row['track_id']]=details
    details=[s for row in rows for s in cached[row['track_id']]]
    result=summarize(rows,details)
    result.update(full_train=len(rows)==len(all_rows),split='train',plan_id=data['plan_id'],provenance_id=provenance,
        checkpoint=str(checkpoint),checkpoint_sha256=sha_file(checkpoint),audit_identity=identity,
        excluded_sections_not_targets=sum(r['excluded_sections'] for r in rows),encoder_executed=False)
    destination=out/('train-all.json' if not args.limit else f'train-first-{len(rows)}.json')
    write(destination,result)
    print('Bridge -> chorus:',result['bridge_to_chorus_count'],'| Chorus -> bridge:',result['chorus_to_bridge_count'])
    print('Report:',destination)


if __name__=='__main__':main()
