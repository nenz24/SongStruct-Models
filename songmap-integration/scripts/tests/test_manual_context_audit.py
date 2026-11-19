import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('manual_audit',SCRIPTS/'audit_manual_context.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture():
    names=[('t1','train','bridge'),('t2','train','inst'),('v1','val','chorus'),('v2','val','verse')]
    return [dict(id=name,track_id=name,group_id=name,audio_sha256=name,split=split,approved=True,duration=10.,
                 excluded_sections=2,sections=[dict(start=0.,end=10.,label=label)]) for name,split,label in names]


def details(records,predictions):
    result=[]
    for r,pred in zip(records,predictions):
        p={label:.2/7 for label in m.LABELS};p[pred]=.8
        result.append(dict(track_id=r['track_id'],start=0.,end=10.,label_manual=r['sections'][0]['label'],
                           prediction=pred,probabilities=p))
    return result


class ManualContextAuditTests(unittest.TestCase):
    def test_manual_confusion_fixes_regressions_and_splits_are_distinct(self):
        rows=fixture();base=details(rows,['chorus','inst','chorus','inst']);candidate=details(rows,['bridge','chorus','bridge','inst'])
        before=copy.deepcopy((rows,base,candidate));r=m.summarize(rows,base,candidate)
        self.assertEqual(r['splits']['train']['wrong_to_correct'],1)
        self.assertEqual(r['splits']['train']['correct_to_wrong'],1)
        self.assertEqual(r['splits']['val']['context_error_patterns'],{'chorus_to_bridge':1,'manual_vocal_function_to_inst':1})
        self.assertEqual(r['splits']['train']['context_error_patterns'],{'manual_inst_to_vocal_function':1})
        self.assertEqual(r['splits']['train']['excluded_sections_not_targets'],4)
        self.assertEqual(len(r['error_cases']),3)
        self.assertEqual((rows,base,candidate),before)
        self.assertFalse(r['manual_labels_changed']);self.assertFalse(r['test_predictions_used'])

    def test_test_pending_and_leaking_groups_or_audio_are_rejected(self):
        for kind in ('test','pending','group_id','audio_sha256'):
            rows=fixture()
            if kind=='test':rows[0]['split']='test'
            elif kind=='pending':rows[0]['approved']=False
            else:rows[2][kind]=rows[0][kind]
            with self.assertRaises(ValueError):m.validate_records(rows)

    def test_incomplete_duplicate_held_or_changed_targets_are_rejected(self):
        rows=fixture();items=details(rows,['bridge','inst','chorus','verse'])
        for mode in ('missing','duplicate','held','changed','nan','argmax'):
            actual=copy.deepcopy(items)
            if mode=='missing':actual.pop()
            elif mode=='duplicate':actual.append(actual[0])
            elif mode=='held':actual.append(dict(actual[0],start=1.,end=2.))
            elif mode=='changed':actual[0]['label_manual']='chorus'
            elif mode=='nan':actual[0]['probabilities']['bridge']=float('nan')
            else:actual[0]['prediction']='chorus'
            with self.assertRaises(ValueError):m.validate_details(rows,actual)

    def test_duration_accuracy_does_not_equal_section_count_accuracy(self):
        rows=fixture()[:2];rows[0]['duration']=rows[0]['sections'][0]['end']=2
        items=details(rows,['bridge','chorus']);items[0]['end']=2
        score=m.score(items)
        self.assertEqual(score['section_accuracy'],.5)
        self.assertAlmostEqual(score['duration_weighted_accuracy'],2/12)

    def test_eligible_neighbor_gaps_remain_visible(self):
        rows=fixture()[:1];rows[0]['sections'].append(dict(start=15.,end=20.,label='verse'));rows[0]['duration']=20
        base=details(rows,['bridge']);extra=dict(base[0],start=15.,end=20.,label_manual='verse')
        base.append(extra);r=m.summarize(rows,base,base)
        self.assertEqual(r['error_cases'][0]['gap_from_previous_eligible_seconds'],5)

    def test_missing_pooled_cache_fails_without_creating_it(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'missing'
            with self.assertRaises(ValueError):m.require_cached_sections(fixture(),path)
            self.assertFalse(path.exists())

    def test_real_frozen_context_forward_has_no_gradients_and_keeps_target_indices(self):
        import torch
        from train_section_context import SectionContextHead
        torch.set_num_threads(2)
        model=SectionContextHead(hidden=8,dropout=0).eval().requires_grad_(False)
        p=torch.tensor([[.01,.02,.8,.1,.02,.02,.02,.01],[.01,.02,.1,.8,.02,.02,.02,.01]])
        base=[dict(track_id='x',start=0.,end=1.,label_manual='chorus'),dict(track_id='x',start=2.,end=3.,label_manual='bridge')]
        sample=dict(split='train',audio=torch.zeros(3,4096),timing=torch.zeros(3,6),indices=torch.tensor([0,2]),
                    probabilities=p,labels=torch.tensor([2,3]),details=base)
        saved={k:v.clone() for k,v in model.state_dict().items()}
        produced,loss=m.predict_sample(model,sample,torch.device('cpu'))
        self.assertEqual([s['prediction'] for s in produced],['chorus','bridge'])
        self.assertEqual([s['original_section_index'] for s in produced],[1,3]);self.assertGreater(loss,0)
        self.assertAlmostEqual(produced[0]['confidence'],produced[0]['probabilities']['chorus'])
        self.assertAlmostEqual(produced[0]['margin'],.7,places=6)
        self.assertTrue(all(torch.equal(saved[k],v) for k,v in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        sample['split']='test'
        with self.assertRaises(ValueError):m.predict_sample(model,sample,torch.device('cpu'))
        sample['split']='train';model.train()
        with self.assertRaises(ValueError):m.predict_sample(model,sample,torch.device('cpu'))

    def native_fixture(self,root):
        rows=fixture()[2:];produced=details(rows,['chorus','inst'])
        source=dict(context_checkpoint_sha256='context',plan_id='plan',provenance_id='features')
        proof=dict(protocol='section-context-coreml-fp32-v1',split='val',full_validation=True,passed=True,selected_epoch=5,
            status='numerically_consistent',precision='fp32',tracks=len(rows),**source,
            checks=[dict(track_id=r['track_id'],passed=True) for r in rows],coreml=m.score(produced))
        path=root/'validation-all.json';path.write_text(json.dumps(proof));folder=root/'per-track';folder.mkdir()
        for r,item in zip(rows,produced):(folder/(r['id']+'.json')).write_text(json.dumps(dict(identity='native',check=dict(passed=True),sections=[item])))
        return path,rows,source

    def test_saved_native_predictions_are_scored_separately_and_checked(self):
        with tempfile.TemporaryDirectory() as d:
            path,rows,source=self.native_fixture(Path(d));read=lambda p:json.loads(p.read_text())
            actual=m.read_native_validation(path,rows,source,read)
            self.assertEqual(m.score(actual)['section_accuracy'],.5)
            p=path.parent/'per-track'/(rows[0]['id']+'.json');cache=read(p);cache['identity']='other';p.write_text(json.dumps(cache))
            with self.assertRaises(ValueError):m.read_native_validation(path,rows,source,read)

    def test_native_wrong_epoch_scores_or_source_fail(self):
        for mode in ('epoch','scores','source'):
            with tempfile.TemporaryDirectory() as d:
                path,rows,source=self.native_fixture(Path(d));read=lambda p:json.loads(p.read_text());proof=read(path)
                if mode=='epoch':proof['selected_epoch']=4
                elif mode=='scores':proof['coreml']['section_accuracy']=1.
                else:proof['plan_id']='changed'
                path.write_text(json.dumps(proof))
                with self.assertRaises(ValueError):m.read_native_validation(path,rows,source,read)


if __name__=='__main__':unittest.main()
