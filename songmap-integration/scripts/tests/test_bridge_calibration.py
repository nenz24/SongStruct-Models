import copy
import importlib.util
import math
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('calibration',Path(__file__).resolve().parents[1]/'calibrate_bridge_chorus.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture(biased=False):
    records=[];items=[]
    for index in range(7):
        track=f'song-{index}'
        sections=[{'start':0.,'end':10.,'label':'intro'},
            {'start':20.,'end':40.,'label':'chorus'},{'start':70.,'end':90.,'label':'bridge'}]
        records.append({'track_id':track,'group_id':f'group-{index//2}',
            'split':'train','approved':True,'duration':120.,'sections':sections})
        for section in sections:
            truth=section['label'];probs=dict.fromkeys(module.LABELS,0.)
            if truth=='intro':probs['intro']=1.
            else:
                probs['bridge']=.3 if biased else (.9 if truth=='bridge' else .1)
                probs['chorus']=1-probs['bridge']
            items.append({'track_id':track,'start':section['start'],'end':section['end'],
                'label_manual':truth,'prediction':max(probs,key=probs.get),'probabilities':probs})
    return records,items


class BridgeCalibrationTests(unittest.TestCase):
    def test_identity_and_pair_gate_preserve_other_predictions_and_inputs(self):
        records,items=fixture();metadata={r['track_id']:r for r in records}
        before=copy.deepcopy(items)
        adapter={'kind':'pair_gate_log_odds_correction','weights':[100.,0.,0.,0.],
            'mean':[0.,0.,0.],'scale':[1.,1.,1.]}
        for item in items:
            self.assertEqual(module.predict(item,metadata,{'kind':'identity'}),(item['prediction'],None))
        self.assertEqual(module.predict(items[0],metadata,adapter),('intro',None))
        self.assertEqual(module.predict(items[1],metadata,adapter)[0],'bridge')
        self.assertEqual(items,before)

    def test_fit_reduces_binary_nll_on_biased_train_and_stays_finite(self):
        records,items=fixture(biased=True);metadata={r['track_id']:r for r in records}
        adapter=module.fit(items,metadata,.01)
        selected=[i for i in items if i['label_manual']!='intro']
        baseline=candidate=0.
        for item in selected:
            q=module.predict(item,metadata,adapter)[1]
            self.assertTrue(math.isfinite(q));self.assertTrue(0<q<1)
            baseline-=math.log(item['probabilities'][item['label_manual']])
            candidate-=math.log(q if item['label_manual']=='bridge' else 1-q)
        self.assertLess(candidate,baseline*.5)
        self.assertGreaterEqual(1+adapter['weights'][1]/adapter['scale'][0],0)

    def test_group_cv_holds_out_aliases_and_identity_wins_ties(self):
        records,items=fixture();before=copy.deepcopy((records,items));calls=[]
        original=module.fit
        def fit_spy(training,metadata,regularization):
            calls.append({metadata[i['track_id']]['group_id'] for i in training})
            return original(training,metadata,regularization)
        with patch.object(module,'fit',side_effect=fit_spy):
            adapter,selection=module.select_train_only(records,items)
        self.assertEqual(adapter,{'kind':'identity'})
        self.assertEqual(selection['selected'],'identity')
        self.assertEqual(len(calls),9) # Three regularizers, three held-out folds; no identity refit.
        groups=set(selection['group_folds'])
        for index,seen in enumerate(calls):
            fold=index%3
            held={g for g,f in selection['group_folds'].items() if f==fold}
            self.assertEqual(seen,groups-held)
        self.assertEqual((records,items),before)

    def test_biased_train_selects_nonidentity_deterministically(self):
        records,items=fixture(biased=True)
        model,selection=module.select_train_only(records,items)
        repeated,repeated_selection=module.select_train_only(records,items)
        self.assertNotEqual(model['kind'],'identity')
        self.assertEqual(model,repeated);self.assertEqual(selection,repeated_selection)
        self.assertEqual(model['fit_sections'],14)

    def test_holdouts_cannot_fit_or_select(self):
        for split in ('val','test'):
            records,items=fixture();records[0]['split']=split
            with self.assertRaises(ValueError):module.select_train_only(records,items)
            with self.assertRaises(ValueError):module.fit(items,{r['track_id']:r for r in records},.1)

    def test_validation_is_read_only_and_requires_complete_val_targets(self):
        records,items=fixture();identity={'kind':'identity'}
        for split in ('train','test'):
            records[0]['split']=split
            with self.assertRaises(ValueError):module.evaluate_validation(records,items,identity)
        for row in records:row['split']='val'
        before=copy.deepcopy((records,items))
        result=module.evaluate_validation(records,items,identity)
        self.assertEqual(result['section_accuracy'],1.)
        self.assertEqual(result['macro_f1_present'],1.)
        self.assertEqual((records,items),before)
        with self.assertRaises(ValueError):module.evaluate_validation(records,items[:-1],identity)

    def test_invalid_inputs_are_rejected(self):
        records,items=fixture();metadata={r['track_id']:r for r in records}
        for reg in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError):module.fit(items,metadata,reg)
        for duration in (0.,float('inf'),float('nan')):
            with self.assertRaises(ValueError):module.features(items[1],duration)
        with self.assertRaises(ValueError):module.select_train_only([],[])
        with self.assertRaises(ValueError):module.evaluate_validation([],[],{'kind':'identity'})

    def test_cli_uses_only_val_forward_and_rejects_stale_audit(self):
        # Exercise file identities and path handling with an inference stub, not model accuracy.
        import torch
        records,items=fixture()
        val=copy.deepcopy(records[0]);val.update(track_id='validation-song',group_id='val-group',split='val')
        for index,row in enumerate(records+[val]):
            row.update(id=f'id-{index}',audio_sha256=f'audio-{index}',annotation_sha256=f'annotation-{index}')
        fingerprint=lambda obj:hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        sha_file=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        read=lambda p:json.loads(Path(p).read_text())
        def write(path,value):
            path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value,allow_nan=False))
        forwarded=[]
        def evaluate(model,rows,cfg,target,provenance,details):
            self.assertTrue(all(r['split']=='val' for r in rows))
            forwarded.extend(r['track_id'] for r in rows)
            details.extend({**i,'track_id':val['track_id']} for i in items[:3])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);songstruct=root/'Section-Annotations';out=root/'experiment'
            write(songstruct/'finetuning/config.json',{})
            checkpoint=root/'best.pt';checkpoint.write_bytes(b'checkpoint fixture')
            identity=fingerprint({'protocol':'train-bridge-audit-v1','device':'cpu','torch':str(torch.__version__),
                'plan_id':'plan','provenance_id':'provenance','checkpoint_sha256':sha_file(checkpoint),
                'train_annotations':[(r['id'],r['annotation_sha256']) for r in records]})
            audit_path=root/'audit/train-all.json'
            write(audit_path,{'full_train':True,'split':'train','tracks':len(records),'plan_id':'plan',
                'provenance_id':'provenance','checkpoint':str(checkpoint),
                'checkpoint_sha256':sha_file(checkpoint),'audit_identity':identity})
            for row in records:
                write(audit_path.parent/'per-track'/(row['id']+'.json'),{'identity':identity,
                    'audio_sha256':row['audio_sha256'],'sections':[i for i in items if i['track_id']==row['track_id']]})
            common=types.ModuleType('finetuning.common');engine=types.ModuleType('finetuning.engine')
            for key,value in dict(config=lambda p:{},read=read,write=write,fingerprint=fingerprint,
                    sha_file=sha_file,device=torch.device).items():setattr(common,key,value)
            def load_head(*args):
                os.chdir(songstruct) # Real load_head changes directory into the vendor repo.
                return object(),None
            common.load_head=load_head
            engine.manifest=lambda cfg:({'plan_id':'plan','records':records+[val]},'provenance')
            engine.evaluate=evaluate
            parent=types.ModuleType('finetuning');parent.__path__=[]
            arguments=['calibrate_bridge_chorus.py','--songstruct-root',str(songstruct),
                '--train-audit',str(audit_path),'--device','cpu','--output-dir',str(out)]
            cwd=Path.cwd();old_path=list(sys.path)
            try:
                with patch.dict(sys.modules,{'finetuning':parent,'finetuning.common':common,'finetuning.engine':engine}),patch.object(sys,'argv',arguments):
                    module.main()
                    report=read(out/'comparison.json')
                    self.assertEqual(forwarded,['validation-song'])
                    self.assertEqual(report['baseline'],report['candidate'])
                    self.assertFalse(report['test_used']);self.assertFalse(report['encoder_executed'])
                    # Resume reads complete per-song predictions without a second forward.
                    (out/'comparison.json').unlink();module.main()
                    self.assertEqual(forwarded,['validation-song'])
                    (out/'comparison.json').unlink();checkpoint.write_bytes(b'changed checkpoint')
                    with self.assertRaisesRegex(ValueError,'Checkpoint changed'):module.main()
            finally:
                os.chdir(cwd);sys.path[:]=old_path


if __name__=='__main__':unittest.main()
