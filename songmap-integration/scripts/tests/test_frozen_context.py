import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

spec = importlib.util.spec_from_file_location('frozen_context', Path(__file__).resolve().parents[1] / 'evaluate_frozen_context.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def record(split='test'):
    return dict(track_id='song', group_id='group', audio_sha256='audio', split=split, approved=True,
                sections=[dict(start=0., end=.08, label='bridge')])


def prediction():
    p=dict.fromkeys(module.LABELS,.01); p['bridge']=.93
    return dict(track_id='song',start=0.,end=.08,label_manual='bridge',prediction='bridge',probabilities=p)


class FrozenContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_only_complete_original_holdout_predictions_are_valid(self):
        row=record(); item=prediction(); module.validate_inventory([row],[item],'test')
        with self.assertRaises(ValueError):module.validate_inventory([row],[],'test')
        with self.assertRaises(ValueError):module.validate_inventory([row],[item,item],'test')
        for change in ({'label_manual':'chorus'},{'prediction':'chorus'},
                       {'probabilities':{**item['probabilities'],'bridge':float('nan')}}):
            with self.assertRaises(ValueError):module.validate_inventory([row],[{**item,**change}],'test')
        row['split']='train'
        with self.assertRaises(ValueError):module.validate_inventory([row],[item],'test')

    def test_group_audio_and_track_leakage_are_rejected(self):
        rows=[]
        for split in ('train','val','test'):
            rows.append({**record(split),'track_id':split,'group_id':split,'audio_sha256':split})
        module.split_records({'records':rows})
        for key in ('track_id','group_id','audio_sha256'):
            changed=copy.deepcopy(rows);changed[2][key]=changed[0][key]
            with self.assertRaises(ValueError):module.split_records({'records':changed})

    def test_validation_reproduction_checks_confusion_and_nll(self):
        score=module.metrics([prediction()],{}, {'kind':'identity'});score['section_nll']=.2
        module.verify_scores(score,copy.deepcopy(score))
        for key,value in (('section_nll',.21),('macro_f1_present',.9),('confusion_matrix',[[0]*8]*8)):
            with self.assertRaises(ValueError):module.verify_scores({**score,key:value},score)

    def test_context_test_evaluation_is_explicit_and_never_a_training_split(self):
        model=module.SectionContextHead();row=record()
        sample=dict(track_id='song',split='test',audio=torch.ones(1,4096),timing=torch.zeros(1,6),
            indices=torch.tensor([0]),probabilities=torch.tensor([[prediction()['probabilities'][k] for k in module.LABELS]]),
            labels=torch.tensor([3]),details=[prediction()])
        with self.assertRaises(ValueError):module.evaluate_context(model,[sample],[row],torch.device('cpu'))
        score,_=module.evaluate_context(model,[sample],[row],torch.device('cpu'),split='test')
        self.assertEqual(score['section_accuracy'],1.)
        with self.assertRaises(ValueError):module.evaluate_context(model,[sample],[row],torch.device('cpu'),split='train')

    def test_cli_freezes_before_test_handles_chdir_and_resumes_without_fitting(self):
        fingerprint=lambda obj:hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        sha_file=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        read=lambda p:json.loads(Path(p).read_text())
        def write(p,value):
            p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,allow_nan=False))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); rows=[]; details=[]; cfg={'work_dir':str(root),'features':{'profile':'frozen'}}
            for i,split in enumerate(('train','val','test','test')):
                sections=[dict(start=0.,end=.08,label='chorus'),dict(start=.16,end=.24,label='bridge')]
                row=dict(id=f'id-{i}',track_id=f'song-{i}',group_id=f'group-{i}',audio_sha256=f'audio-{i}',
                    annotation_sha256=fingerprint(sections),source_json=str(root/f'source-{i}.json'),
                    duration=.24,approved=True,split=split,sections=sections)
                rows.append(row)
                write(row['source_json'],dict(track_id=row['track_id'],duration=.24,
                    sections=[dict(start=0.,end=.08),dict(start=.08,end=.16),dict(start=.16,end=.24)]))
                write(root/'features'/row['id']/'metadata.json',dict(audio_sha256=row['audio_sha256'],
                    features=cfg['features'],provenance_id='provenance',chunks=[dict(offset=0.,frames=6)]))
                for s in sections:
                    p=dict.fromkeys(module.LABELS,.01);p[s['label']]=.93
                    details.append(dict(track_id=row['track_id'],start=s['start'],end=s['end'],
                        label_manual=s['label'],prediction=s['label'],probabilities=p))
            train,val=rows[:1],rows[1:2]
            checkpoint=root/'original.pt';checkpoint.write_bytes(b'original frozen checkpoint')
            audit_path=root/'audit/train-all.json'
            audit_identity=fingerprint(dict(protocol='train-bridge-audit-v1',device='cpu',torch=str(torch.__version__),
                plan_id='plan',provenance_id='provenance',checkpoint_sha256=sha_file(checkpoint),
                train_annotations=[(r['id'],r['annotation_sha256']) for r in train]))
            write(audit_path,dict(audit_identity=audit_identity))
            for row in train:
                write(audit_path.parent/'per-track'/(row['id']+'.json'),
                    dict(identity=audit_identity,audio_sha256=row['audio_sha256']))
            calibration_path=root/'calibration/comparison.json';adapter={'kind':'identity'}
            write(calibration_path.parent/'adapter.json',{'adapter':adapter})
            val_identity=fingerprint(dict(protocol='pair-calibration-v1',device='cpu',torch=str(torch.__version__),
                train_audit_sha256=sha_file(audit_path),checkpoint_sha256=sha_file(checkpoint),
                plan_id='plan',provenance_id='provenance',
                val_annotations=[(r['id'],r['annotation_sha256']) for r in val],selected_adapter=adapter))
            val_details=[d for d in details if d['track_id']==val[0]['track_id']]
            baseline=module.metrics(val_details,{},adapter)
            write(calibration_path,dict(checkpoint=str(checkpoint),checkpoint_sha256=sha_file(checkpoint),
                plan_id='plan',provenance_id='provenance',train_audit=str(audit_path),experiment_identity=val_identity))
            write(calibration_path.parent/'val-predictions/id-1.json',dict(identity=val_identity,sections=val_details))
            pooled=root/'pooled';pooled.mkdir();out=root/'evaluation';context=root/'context-run';context.mkdir()
            def iter_features(cfg,row):
                yield np.arange(6*4096,dtype=np.float32).reshape(6,4096)/4096,0.,'provenance'
            helpers=read,write,fingerprint,sha_file,iter_features
            tv_details=[d for d in details if d['track_id'] in {r['track_id'] for r in train+val}]
            samples=module.prepare_samples(train+val,tv_details,cfg,'provenance',pooled,helpers)
            import train_section_context as training
            model=module.SectionContextHead();model.eval()
            selected_score,_=module.evaluate_context(model,samples[1:],val,torch.device('cpu'))
            metadata=dict(protocol=module.PROTOCOL,settings=dict(training.SETTINGS),device='cpu',torch=str(torch.__version__),
                coverage_policy=module.COVERAGE_POLICY,plan_id='plan',provenance_id='provenance',
                checkpoint_sha256=sha_file(checkpoint),calibration_report_sha256=sha_file(calibration_path),
                sample_identities=[(s['track_id'],s['identity']) for s in samples])
            identity=fingerprint(metadata)
            write(context/'run.json',{'identity':identity,**metadata})
            write(context/'comparison.json',dict(protocol=module.PROTOCOL,split='val',tracks=1,full_validation=True,
                test_used=False,plan_id='plan',provenance_id='provenance',experiment_identity=identity,
                best_epoch=5,baseline=baseline,previous_calibration=baseline,candidate=selected_score))
            torch.save(dict(identity=identity,architecture=module.PROTOCOL,settings=dict(training.SETTINGS),
                model=model.state_dict(),best_epoch=5,plan_id='plan',provenance_id='provenance'),context/'best.pt')
            original_hashes=sha_file(checkpoint),sha_file(context/'best.pt')
            forwarded=[]
            def evaluate(original,records,cfg,target,provenance,details):
                self.assertTrue((out/'selection.json').is_file()) # Selection is durable BEFORE test scoring.
                self.assertEqual(len(read(out/'selection.json')['test_inventory']),2)
                self.assertTrue(all(r['split']=='test' for r in records))
                self.assertFalse(any(p.requires_grad for p in original.parameters()))
                for row in records:
                    forwarded.append(row['track_id'])
                    details.extend(d for d in all_details if d['track_id']==row['track_id'])
            all_details=details
            def load_head(cfg,target,checkpoint):
                (root/'vendor').mkdir(exist_ok=True);os.chdir(root/'vendor')
                return torch.nn.Linear(1,1),None
            common=types.ModuleType('finetuning.common');engine=types.ModuleType('finetuning.engine')
            for key,value in dict(config=lambda p:cfg,read=read,write=write,fingerprint=fingerprint,sha_file=sha_file,
                iter_features=iter_features,device=torch.device,load_head=load_head).items():setattr(common,key,value)
            engine.manifest=lambda cfg:({'plan_id':'plan','records':rows},'provenance');engine.evaluate=evaluate
            package=types.ModuleType('finetuning');package.__path__=[]
            args=['evaluate_frozen_context.py','--songstruct-root',str(root),'--context-run',str(context),
                '--calibration',str(calibration_path),'--cache-dir',str(pooled),'--output-dir',str(out),
                '--baseline-device','cpu','--device','cpu','--expected-epoch','5']
            old_path=list(sys.path);cwd=Path.cwd()
            try:
                with patch.dict(sys.modules,{'finetuning':package,'finetuning.common':common,'finetuning.engine':engine}), \
                        patch.object(torch.optim,'AdamW',side_effect=AssertionError('No optimizer allowed during frozen test')):
                    with patch.object(sys,'argv',args+['--limit','1']):module.main()
                    self.assertFalse(read(out/'test-first-1.json')['full_test'])
                    self.assertEqual(sha_file(out/'frozen-context.pt'),original_hashes[1])
                    self.assertEqual(forwarded,['song-2'])
                    with patch.object(sys,'argv',args):module.main()
                    report=read(out/'test-all.json')
                    self.assertTrue(report['full_test']);self.assertEqual(report['tracks'],2)
                    self.assertEqual(forwarded,['song-2','song-3'])
                    self.assertFalse(report['weights_updated']);self.assertFalse(report['parameters_selected_on_test'])
                    self.assertEqual(report['worship_v3']['confusion_matrix'],report['section_context']['confusion_matrix'])
                    with patch.object(sys,'argv',args):module.main()
                    self.assertEqual(forwarded,['song-2','song-3'])
                    self.assertEqual(original_hashes,(sha_file(checkpoint),sha_file(context/'best.pt')))
                    write(calibration_path.parent/'adapter.json',{'adapter':adapter,'extra':'changed artifact'})
                    with patch.object(sys,'argv',args), self.assertRaisesRegex(ValueError,'Frozen selection'):
                        module.main()
                    write(calibration_path.parent/'adapter.json',{'adapter':adapter})
                    changed=torch.load(context/'best.pt',weights_only=True)
                    changed['model']['output.bias'][2]+=.5
                    torch.save(changed,context/'best.pt')
                    with patch.object(sys,'argv',args), self.assertRaisesRegex(ValueError,'does not reproduce'):
                        module.main()
                    self.assertEqual(forwarded,['song-2','song-3'])
            finally:
                os.chdir(cwd);sys.path[:]=old_path


if __name__=='__main__':unittest.main()
