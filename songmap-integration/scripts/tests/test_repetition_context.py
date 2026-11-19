import copy
import hashlib
import types
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import train_repetition_context as m
from train_section_context import SectionContextHead, forward_targets


def raw_sample(split='train', name='t'):
    torch.manual_seed(19)
    probabilities=torch.full((3,8), .05);probabilities[:,2]=.65
    return dict(split=split,track_id=name,identity=name,audio=torch.randn(5,4096),timing=torch.rand(5,6),
        indices=torch.tensor([0,2,4]),probabilities=probabilities,labels=torch.tensor([2,3,2]),
        details=[dict(track_id=name,start=float(i*10),end=float(i*10+10),label_manual=label)
                 for i,label in enumerate(('chorus','bridge','chorus'))])


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value))


class RepetitionContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        # Warm lazy optimizer imports before temporarily supplying finetuning.common.
        from torch import _dynamo

    def teacher(self):
        torch.manual_seed(29)
        model=SectionContextHead(hidden=8,dropout=.2).eval().requires_grad_(False)
        with torch.no_grad():model.output.weight.normal_(std=.02)
        return model

    def prepared(self):
        return m.prepare_teacher_samples(self.teacher(),[raw_sample(),raw_sample('val','v')],torch.device('cpu'))

    def test_graph_links_nonadjacent_audio_repetition_only(self):
        torch.manual_seed(9)
        base=torch.randn(4096);other=torch.randn(4096)
        audio=torch.stack([base,other,base,-base,-other])
        sim,w,allowed=m.recurrence_graph(audio)
        self.assertAlmostEqual(sim[0,2].item(),1.,places=5)
        self.assertGreater(w[0,2].item(),.99)
        self.assertTrue(torch.all(w[~allowed]==0))
        self.assertEqual(w[0,0].item(),0);self.assertEqual(w[0,1].item(),0)
        self.assertTrue(torch.all((w.sum(1)==0)|(torch.isclose(w.sum(1),torch.ones(5)))))
        self.assertTrue(torch.equal(audio,torch.stack([base,other,base,-base,-other])))

    def test_tiny_or_constant_songs_are_finite_without_spurious_neighbors(self):
        for n in (1,2,4):
            a=torch.ones(n,4096);timing=torch.zeros(n,6);h=torch.ones(n,8);r=torch.zeros(n,8)
            sim,w,_=m.recurrence_graph(a)
            self.assertEqual(w.count_nonzero().item(),0)
            f=m.adapter_features(a,timing,h,r,'repetition_adapter')
            self.assertTrue(torch.isfinite(f).all());self.assertEqual(f[:,-6:].count_nonzero().item(),0)
        for a in (torch.zeros(0,4096),torch.zeros(513,4096),torch.ones(2,1),torch.full((2,4096),float('nan'))):
            with self.assertRaises(ValueError):m.recurrence_graph(a)

    def test_ablation_keeps_own_frozen_representation_and_zeros_only_extra_features(self):
        s=raw_sample();h=torch.randn(5,8);r=torch.randn(5,8)
        f=m.adapter_features(s['audio'],s['timing'],h,r,'context_adapter')
        self.assertTrue(torch.equal(f[:,:8],h));self.assertTrue(torch.equal(f[:,16:24],r))
        self.assertEqual(f[:,8:16].count_nonzero().item(),0)
        self.assertEqual(f[:,24:].count_nonzero().item(),0)
        with self.assertRaises(ValueError):m.adapter_features(s['audio'],s['timing'],h,r,'bad')

    def test_teacher_features_reproduce_real_forward_and_ignore_manual_labels(self):
        teacher=self.teacher();sample=raw_sample()
        with torch.no_grad():expected,_,_=forward_targets(teacher,sample,torch.device('cpu'))
        a=m.prepare_teacher_samples(teacher,[sample],torch.device('cpu'))[0]
        changed=copy.deepcopy(sample);changed['labels'].fill_(7);changed['details'][0]['label_manual']='pre-chorus'
        b=m.prepare_teacher_samples(teacher,[changed],torch.device('cpu'))[0]
        self.assertTrue(torch.equal(expected,a['base_scores']))
        for mode in m.MODES:self.assertTrue(torch.equal(a['features'][mode],b['features'][mode]))
        self.assertFalse(a['base_scores'].requires_grad)
        for features in a['features'].values():self.assertFalse(features.requires_grad)
        self.assertEqual(a['indices'].tolist(),[0,2,4]) # Held section labels do not become targets.

    def test_zero_adapter_is_identity_and_backward_changes_only_adapter(self):
        teacher=self.teacher();saved=copy.deepcopy(teacher.state_dict())
        prepared=m.prepare_teacher_samples(teacher,[raw_sample(),raw_sample('val','v')],torch.device('cpu'))
        sample=prepared[0]
        for mode in m.MODES:
            model=m.RepetitionAdapter(sample['features'][mode].shape[1],hidden=8,dropout=.2).eval()
            baseline,details=m.predict(None,prepared,mode);actual,produced=m.predict(model,prepared,mode)
            self.assertEqual(actual,baseline);self.assertEqual(produced,details)
            before=copy.deepcopy(model.state_dict());optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
            random.seed(42);m.train_epoch(model,optimizer,[sample],mode,m.SETTINGS)
            self.assertTrue(any(not torch.equal(before[k],v) for k,v in model.state_dict().items()))
            self.assertTrue(all(torch.equal(saved[k],v) for k,v in teacher.state_dict().items()))
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            with self.assertRaises(ValueError):m.supervised_loss(model,prepared[1],mode,.02)
            with self.assertRaises(ValueError):m.train_epoch(model,optimizer,prepared,mode,m.SETTINGS)
            with self.assertRaises(ValueError):m.predict(model,prepared,mode) # Training-mode dropout rejected.

    def test_test_inference_and_unfrozen_teacher_rejected(self):
        teacher=self.teacher();s=raw_sample('test')
        with self.assertRaises(ValueError):m.prepare_teacher_samples(teacher,[s],torch.device('cpu'))
        s['split']='train';teacher.train()
        with self.assertRaises(ValueError):m.prepare_teacher_samples(teacher,[s],torch.device('cpu'))
        teacher.eval().requires_grad_(True)
        with self.assertRaises(ValueError):m.prepare_teacher_samples(teacher,[s],torch.device('cpu'))

    def test_selection_retains_identity_or_simpler_ablation_and_protects_chorus(self):
        baseline=dict(macro_f1_present=.7,per_class={k:dict(f1=.7) for k in m.LABELS})
        candidates={mode:dict(validation=copy.deepcopy(baseline)) for mode in m.MODES}
        self.assertEqual(m.choose_candidate(baseline,candidates),'identity')
        for value in candidates.values():value['validation']['macro_f1_present']=.72
        self.assertEqual(m.choose_candidate(baseline,candidates),'context_adapter')
        candidates['repetition_adapter']['validation']['macro_f1_present']=.73
        self.assertEqual(m.choose_candidate(baseline,candidates),'repetition_adapter')
        candidates['repetition_adapter']['validation']['per_class']['chorus']['f1']=.68
        self.assertEqual(m.choose_candidate(baseline,candidates),'context_adapter')
        candidates['context_adapter']['validation']['per_class']['bridge']['f1']=.69
        self.assertEqual(m.choose_candidate(baseline,candidates),'identity')

    def test_interrupted_cpu_training_resumes_identically_and_checks_identity(self):
        train,val=self.prepared();mode='repetition_adapter'
        baseline,_=m.predict(None,[val],mode)
        settings=dict(m.SETTINGS,epochs=3,patience=4,hidden=8,learning_rate=.01)
        with tempfile.TemporaryDirectory() as d:
            full=Path(d)/'full';resumed=Path(d)/'resumed'
            result_a,pred_a=m.fit_mode([train],[val],mode,baseline,settings,'id',full,read,write)
            m.fit_mode([train],[val],mode,baseline,settings,'id',resumed,read,write,epoch_budget=1)
            result_b,pred_b=m.fit_mode([train],[val],mode,baseline,settings,'id',resumed,read,write)
            a=torch.load(full/'latest.pt',weights_only=False);b=torch.load(resumed/'latest.pt',weights_only=False)
            for k in a['model']:self.assertTrue(torch.equal(a['model'][k],b['model'][k]))
            self.assertEqual(result_a,result_b);self.assertEqual(pred_a,pred_b)
            self.assertEqual(a['python_rng'],b['python_rng']);self.assertTrue(torch.equal(a['torch_rng'],b['torch_rng']))
            with self.assertRaises(ValueError):m.fit_mode([train],[val],mode,baseline,settings,'changed',resumed,read,write)
            changed=copy.deepcopy(baseline);changed['section_nll']+=.1
            with self.assertRaises(ValueError):m.fit_mode([train],[val],mode,changed,settings,'id',resumed,read,write)

    def test_cli_smoke_full_and_resume_preserve_source_files_and_reject_stale_metrics(self):
        teacher=self.teacher();samples=[raw_sample(),raw_sample('val','v')]
        prepared=m.prepare_teacher_samples(teacher,samples,torch.device('cpu'))
        def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
        common=types.ModuleType('finetuning.common')
        common.read,common.write,common.sha_file,common.fingerprint=read,write,sha,fingerprint
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);snapshot=root/'frozen.pt';snapshot.write_bytes(b'frozen teacher')
            run=root/'old-run.json';run.write_text('{}');calibration=root/'calibration.json';calibration.write_text('{}')
            rows=[]
            for sample in samples:
                rows.append(dict(track_id=sample['track_id'],group_id=sample['track_id'],audio_sha256=sample['track_id'],
                    approved=True,split=sample['split'],duration=30.,sections=[dict(start=x['start'],end=x['end'],label=x['label_manual'])
                                                                            for x in sample['details']]))
            audit=dict(protocol='manual-context-audit-v1',full_train_val=True,original_context_validation_reproduced=True,
                context_checkpoint_sha256=sha(snapshot),plan_id='plan',provenance_id='prov',source_run_sha256=sha(run),
                source_calibration_sha256=sha(calibration),splits={})
            for split in ('train','val'):
                metrics,_=m.predict(None,[s for s in prepared if s['split']==split],m.MODES[0])
                audit['splits'][split]=dict(context=metrics)
            audit_path=root/'audit.json';write(audit_path,audit);before=sha(snapshot)
            def load(*args):
                return dict(model=teacher,samples=samples,target=torch.device('cpu'),rows=rows,
                    groups={split:[r for r in rows if r['split']==split] for split in ('train','val')},
                    snapshot=snapshot,checkpoint_sha=sha(snapshot),data=dict(plan_id='plan'),provenance='prov',
                    run_path=run,calibration_path=calibration)
            argv=['experiment','--frozen-evaluation',str(root),'--cache-dir',str(root),
                  '--audit',str(audit_path),'--output-dir',str(root/'new')]
            with patch.dict(sys.modules,{'finetuning.common':common}),patch.object(m,'load_frozen_inputs',side_effect=load),patch.object(
                    m,'SETTINGS',dict(m.SETTINGS,epochs=2,hidden=8)):
                with patch.object(sys,'argv',argv+['--smoke-test']):m.main()
                self.assertFalse((root/'new').exists())
                with patch.object(sys,'argv',argv):m.main()
                first=read(root/'new/comparison.json')
                self.assertEqual(first['baseline']['val'],audit['splits']['val']['context'])
                self.assertFalse(first['test_used']);self.assertFalse(first['ready_for_app_integration'])
                with patch.object(sys,'argv',argv+['--resume']):m.main()
                self.assertEqual(first,read(root/'new/comparison.json'))
                with patch.object(sys,'argv',argv):
                    with self.assertRaises(ValueError):m.main()
                for split in ('train','val'):
                    changed=copy.deepcopy(audit);changed['splits'][split]['context']['section_nll']+=.1;write(audit_path,changed)
                    with patch.object(sys,'argv',argv+['--resume']):
                        with self.assertRaises(ValueError):m.main()
            self.assertEqual(before,sha(snapshot))

    def test_audit_requires_full_reproduced_matching_frozen_sources(self):
        inputs=dict(checkpoint_sha='w',data=dict(plan_id='p'),provenance='v',run_path='r',calibration_path='c',sha_file=lambda p:p)
        audit=dict(protocol='manual-context-audit-v1',full_train_val=True,original_context_validation_reproduced=True,
            context_checkpoint_sha256='w',plan_id='p',provenance_id='v',source_run_sha256='r',source_calibration_sha256='c')
        m.validate_audit(audit,inputs)
        for key in audit:
            wrong=dict(audit);wrong[key]=False
            with self.assertRaises(ValueError):m.validate_audit(wrong,inputs)


if __name__=='__main__':unittest.main()
