import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import prepare_encoder_pipeline_iphone as m


def quality():
    metric=dict(section_accuracy=.8,macro_f1_present=.75,confusion_matrix=[[1,0],[0,1]])
    return dict(protocol=m.QUALITY_PROTOCOL,split='val',full_validation=True,section_agreement=1.,changed_sections=[],
        comparison_to_current_pipeline=dict(no_observed_drop=True),weights_updated=False,test_used=False,
        manual_labels_changed=False,boundaries_changed=False,selection_identity='id',candidate=copy.deepcopy(metric),
        baseline_palette6_context=metric)


class PrepareEncoderPipelineTests(unittest.TestCase):
    def test_output_admission_is_restricted_to_isolated_frontend_probe(self):
        m.validate_preflight_policy('strict-feature',False)
        m.validate_preflight_policy('functional-benchmark',True)
        for policy,probe in [('functional-benchmark',False),('ignore-errors',True)]:
            with self.assertRaises(ValueError):m.validate_preflight_policy(policy,probe)

    def test_feature_mismatch_is_retained_and_endpoint_still_checked(self):
        import torch
        reference=np.zeros((1,4096),dtype=np.float32);audio=reference.copy();audio[0,3072]=0.00067228
        p=np.array([[.8,.2]],dtype=np.float32)
        modules={'deployment.verify_coreml':SimpleNamespace(CoreMLAdapter=lambda model:model),
                 'finetuning.engine':SimpleNamespace(logits=lambda *args:None,
                     pooled_sections=lambda *args:[(0,torch.tensor(p[0]),1.)])}
        class Backend:
            pair=None
            released=False
            def load(self,key):return key
            def release(self):self.released=True
        def pool(observations,*args):list(observations);return audio,np.ones(1)
        def run(policy):
            backend=Backend()
            try:
                return m.mac_preflight(backend,None,dict(track_id='song',sections=[],duration=1.),
                    [dict(start=0.,end=1.)],[0],reference,p,[],{},policy=policy)
            finally:self.assertTrue(backend.released)
        with patch.dict(sys.modules,modules),patch.object(m,'pcm_chunks',return_value=[(0.,np.zeros((3,4096)))]),\
                patch.object(m,'pool_audio',side_effect=pool),patch.object(m,'geometry',return_value=np.zeros((1,6))):
            with patch.object(m,'native_context_check',return_value=dict(passed=True,section_agreement=1.)) as context:
                with self.assertRaises(m.PreflightMismatch) as raised:run('strict-feature')
                self.assertTrue(raised.exception.check['functional_passed']);context.assert_called_once()
                result=run('functional-benchmark')
                self.assertTrue(result['passed']);self.assertFalse(result['numerical_passed'])
                self.assertEqual(result['audio_features']['failed_values'],1)
                self.assertFalse(result['stream_features']['muq_120']['passed'])
                self.assertTrue(result['stream_features']['musicfm_120']['passed'])
                self.assertEqual(result['worst_source_interval']['index'],0)
                modules['finetuning.engine'].pooled_sections=lambda *args:[(0,torch.tensor(p[0])+0.0002,1.)]
                with self.assertRaises(m.PreflightMismatch) as raised:run('functional-benchmark')
                self.assertFalse(raised.exception.check['base_probabilities']['passed'])
                modules['finetuning.engine'].pooled_sections=lambda *args:[(0,torch.tensor(p[0]),1.)]
            with patch.object(m,'native_context_check',return_value=dict(passed=False,section_agreement=0.)):
                with self.assertRaises(m.PreflightMismatch) as raised:run('functional-benchmark')
                self.assertFalse(raised.exception.check['functional_passed'])
            audio[0,3072]=float('nan')
            with patch.object(m,'native_context_check') as context:
                with self.assertRaises(m.PreflightMismatch):run('functional-benchmark')
                context.assert_not_called()

    def test_frontend_gpu_probe_requires_full_quality_of_exact_repaired_graph(self):
        report=quality();report.update(static_graph_policy='split-before-conv-v2',graph_repair_sha256='repair')
        selection=dict(identity='id',static_graph_policy='split-before-conv-v2',graph_repair_sha256='repair')
        m.validate_frontend_probe(report,selection,Path('repair.json'),lambda p:'repair')
        for key,value in [('full_validation',False),('section_agreement',.99),('static_graph_policy','v1'),('graph_repair_sha256','other')]:
            invalid=dict(report);invalid[key]=value
            with self.assertRaises(ValueError):m.validate_frontend_probe(invalid,selection,Path('repair.json'),lambda p:'repair')
        for path,digest in [(None,'repair'),(Path('wrong'),'other')]:
            with self.assertRaises(ValueError):m.validate_frontend_probe(report,selection,path,lambda p:digest)
        with self.assertRaises(ValueError):m.validate_frontend_probe(report,dict(identity='id'),Path('repair.json'),lambda p:'repair')

    def test_repaired_quality_cannot_be_prepared_with_old_or_different_graph(self):
        selection=dict(graph_repair_sha256='validated')
        m.validate_repair_selection(selection,Path('correct'),lambda p:'validated')
        m.validate_repair_selection({},None,lambda p:'unused') # Legacy control reports remain supported.
        for path in (None,Path('different')):
            with self.assertRaises(ValueError):m.validate_repair_selection(selection,path,lambda p:'wrong')

    def test_feature_gate_uses_export_relative_scale_but_rejects_near_zero_drift(self):
        reference=np.array([[2.,0.]],dtype=np.float32)
        actual=reference.copy();actual[0,0]+=0.000108242
        check=m.feature_parity(reference,actual)
        self.assertTrue(check['passed']);self.assertEqual(check['absolute_only_failed_values'],1)
        self.assertLess(check['max_tolerance_ratio'],1.)
        actual[0,1]=0.000108242
        check=m.feature_parity(reference,actual)
        self.assertFalse(check['passed']);self.assertEqual(check['failed_values'],1)
        self.assertEqual(check['worst_index'],[0,1])
        self.assertGreater(check['max_tolerance_ratio'],1.)
        for invalid in (np.array([[float('nan'),0.]]),np.array([[float('inf'),0.]]),np.zeros((2,2))):
            self.assertFalse(m.feature_parity(reference,invalid)['passed'])

    def test_feature_tolerance_does_not_relax_probability_gate(self):
        reference=np.array([[2.]],dtype=np.float32);actual=reference+0.000108242
        p=np.array([[.8,.2]],dtype=np.float32)
        same=m.pcm_parity_check(actual,p+0.000000626,reference,p)
        self.assertTrue(same['passed'])
        failed=m.pcm_parity_check(actual,p+0.000108242,reference,p)
        self.assertTrue(failed['audio_features']['passed']);self.assertFalse(failed['passed'])
        self.assertEqual(failed['base_probabilities']['rtol'],0.)

    def test_pcm_preflight_still_requires_native_context_and_releases_models_on_rejection(self):
        import torch
        reference=np.array([[2.]],dtype=np.float32);audio=reference+0.000108242
        p=np.array([[.8,.2]],dtype=np.float32)
        modules={'deployment.verify_coreml':SimpleNamespace(CoreMLAdapter=lambda model:model),
                 'finetuning.engine':SimpleNamespace(logits=lambda *args:None,
                     pooled_sections=lambda *args:[(0,torch.tensor(p[0]),1.)])}
        class Backend:
            released=False
            pair=None  # pcm_chunks is replaced by the controlled observation fixture below.
            def load(self,key):return key
            def release(self):self.released=True
        def pool(observations,*args):
            list(observations);return audio,np.ones(1)
        with patch.dict(sys.modules,modules),patch.object(m,'pcm_chunks',return_value=[(0.,np.zeros((3,4)))]),\
                patch.object(m,'pool_audio',side_effect=pool),patch.object(m,'geometry',return_value=np.zeros((1,6))):
            with patch.object(m,'native_context_check',return_value=dict(passed=True,section_agreement=1.)) as context:
                backend=Backend()
                check=m.mac_preflight(backend,None,dict(track_id='song',sections=[],duration=1.),[],[0],reference,p,[],{})
                self.assertTrue(check['passed']);self.assertTrue(backend.released);context.assert_called_once()
            with patch.object(m,'native_context_check',side_effect=ValueError('changed label')):
                backend=Backend()
                with self.assertRaises(m.PreflightMismatch) as raised:
                    m.mac_preflight(backend,None,dict(track_id='song',sections=[],duration=1.),[],[0],reference,p,[],{})
                self.assertTrue(backend.released)
                self.assertFalse(raised.exception.check['passed'])
                self.assertEqual(raised.exception.check['native_context']['error'],'changed label')

    def test_complete_unchanged_label_validation_required(self):
        source=quality();m.validate_quality(source,dict(identity='id'))
        for key,value in [('full_validation',False),('section_agreement',.999),('test_used',True),('weights_updated',True),
                          ('boundaries_changed',True),('manual_labels_changed',True),('selection_identity','other')]:
            wrong=copy.deepcopy(source);wrong[key]=value
            with self.assertRaises(ValueError):m.validate_quality(wrong,dict(identity='id'))
        wrong=copy.deepcopy(source);wrong['candidate']['macro_f1_present']-=.01
        with self.assertRaises(ValueError):m.validate_quality(wrong,dict(identity='id'))
        wrong=copy.deepcopy(source);wrong['comparison_to_current_pipeline']['no_observed_drop']=False
        with self.assertRaises(ValueError):m.validate_quality(wrong,dict(identity='id'))

    def test_track_selection_exercises_long_head_and_only_approved_validation(self):
        rows=[dict(track_id=name,approved=True,split='val',duration=duration) for name,duration in [('first',240),('second',460),('longest',660)]]
        self.assertEqual([r['track_id'] for r in m.choose_tracks(rows)],['first','longest'])
        self.assertEqual([r['track_id'] for r in m.choose_tracks(rows,['second','first'])],['second','first'])
        for requested in (['first'],['first','first'],['first','missing']):
            with self.assertRaises(ValueError):m.choose_tracks(rows,requested)
        with self.assertRaises(ValueError):m.choose_tracks([dict(r,duration=200) for r in rows])
        for key,value in [('split','test'),('approved',False)]:
            wrong=copy.deepcopy(rows);wrong[0][key]=value
            with self.assertRaises(ValueError):m.choose_tracks(wrong)

    def test_native_context_checks_selected_original_indices_and_final_probabilities(self):
        p=np.array([[.05,.05,.65,.05,.05,.05,.05,.05],[.05,.05,.05,.65,.05,.05,.05,.05]],dtype=np.float32)
        expected=p/p.sum(1,keepdims=True)
        details=[dict(prediction='chorus',probabilities=dict(zip(m.LABELS,expected[0].tolist()))),
                 dict(prediction='bridge',probabilities=dict(zip(m.LABELS,expected[1].tolist())))]
        class Native:
            def predict(self,inputs):
                self.inputs=inputs
                residual=np.zeros((3,8),dtype=np.float32);residual[1]=100 # Held section is not a target.
                return dict(context_residual=residual)
        native=Native();r=m.native_context_check(native,np.zeros((3,4096)),np.zeros((3,6)),p,[0,2],details)
        self.assertTrue(r['passed']);self.assertLess(r['max_probability_error'],1e-6)
        self.assertEqual(native.inputs['section_audio'].shape,(3,4096))
        for output in (np.zeros((2,8)),np.full((3,8),float('nan'))):
            with self.assertRaises(ValueError):m.native_context_check(SimpleNamespace(predict=lambda values:dict(context_residual=output)),np.zeros((3,4096)),np.zeros((3,6)),p,[0,2],details)
        wrong=copy.deepcopy(details);wrong[0]['prediction']='verse'
        with self.assertRaises(ValueError):m.native_context_check(native,np.zeros((3,4096)),np.zeros((3,6)),p,[0,2],wrong)
        diagnostic=m.native_context_check(native,np.zeros((3,4096)),np.zeros((3,6)),p,[0,2],wrong,strict=False)
        self.assertFalse(diagnostic['passed']);self.assertEqual(diagnostic['changed_eligible_indices'],[0])

    def test_long_stream_thirty_second_tail_is_not_reclassified_as_full_short_profile(self):
        calls=[]
        def encode(wave,seconds):
            full=len(wave)==seconds*1000
            calls.append((len(wave),seconds,full))
            frames=len(wave)//1000*25
            return [np.full((frames,2),float(full),dtype=np.float32),np.full((frames,2),seconds,dtype=np.float32)]
        result=list(m.pcm_chunks(np.zeros(150000),encode,dict(sample_rate=1000,head_window_seconds=420),width=2))
        self.assertEqual(len(result),1)
        self.assertEqual(result[0][1].shape,(3750,8))
        self.assertIn((30000,120,False),calls)
        self.assertEqual(sum(full for _,seconds,full in calls if seconds==30),5)
        np.testing.assert_array_equal(result[0][1][3000:,4:6],0.)
        np.testing.assert_array_equal(result[0][1][:,0:2],1.)

    def test_mac_backend_keeps_one_profile_per_encoder_role(self):
        calls=[]
        def model(path,compute_units):
            value=SimpleNamespace(path=path,units=compute_units);calls.append(value);return value
        ct=SimpleNamespace(models=SimpleNamespace(CompiledMLModel=model),ComputeUnit=SimpleNamespace(CPU_ONLY='cpuOnly'))
        with tempfile.TemporaryDirectory() as d:
            backend=m.MacPipeline(ct,Path(d),{}, {})
            first=backend.load('musicfm_750');self.assertIs(first,backend.load('musicfm_750'))
            backend.load('muq_750');backend.load('musicfm_3000');backend.load('musicfm')
            self.assertEqual(set(backend.models),{'musicfm','muq'})
            self.assertEqual(backend.models['musicfm'][0],'musicfm')
            self.assertTrue(all(c.units=='cpuOnly' for c in calls))
            self.assertEqual(len(calls),4)
            backend.release();self.assertEqual(backend.models,{})
            old=backend.load('muq',frames=50);fresh=backend.load('muq',frames=1550)
            self.assertIsNot(old,fresh);self.assertIs(fresh,backend.load('muq',frames=1550))

    def test_mac_frontend_model_is_reused_only_for_matching_input_shape(self):
        created=[]
        class Native:
            def __init__(self,path,compute_units):
                self.role=Path(path).stem;self.units=compute_units;created.append(self)
            def predict(self,inputs):
                values=next(iter(inputs.values()))
                if self.role=='log_mel':
                    return dict(mel=np.zeros((1,128,(values.shape[1]-2048)//240),dtype=np.float32))
                if self.role.endswith('_conv'):
                    return dict(conv=np.zeros((1,values.shape[-1]//4,1024),dtype=np.float32))
                return dict(embedding=np.zeros_like(inputs['conv_in']))
        ct=SimpleNamespace(models=SimpleNamespace(CompiledMLModel=Native),ComputeUnit=SimpleNamespace(CPU_ONLY='cpuOnly'))
        rotary=SimpleNamespace(rotary=lambda frames,frequencies:np.zeros((2,frames,1,1,64),dtype=np.float32))
        with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,{'deployment.verify_encoders':rotary}):
            backend=m.MacPipeline(ct,Path(d),{role:'input' for role in ('log_mel','musicfm_conv','muq_conv')},dict(musicfm=[],muq=[]))
            backend.pair(np.zeros(30*24000,dtype=np.float32),30)
            backend.pair(np.zeros(30*24000,dtype=np.float32),30)
            self.assertEqual(sum(v.role=='musicfm_conv' for v in created),1)
            backend.pair(np.zeros(120*24000,dtype=np.float32),120)
            self.assertEqual(sum(v.role=='musicfm_conv' for v in created),2)
            self.assertEqual(sum(v.role=='muq_conv' for v in created),2)
            self.assertTrue(all(v.units=='cpuOnly' for v in created))
            backend.release()


if __name__=='__main__':unittest.main()
