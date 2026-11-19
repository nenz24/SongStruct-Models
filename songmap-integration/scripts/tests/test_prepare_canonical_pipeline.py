import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import prepare_canonical_pipeline_iphone as m


def quality():
    selection=dict(protocol=m.QUALITY_PROTOCOL,reference_quality_sha256='source')
    selection['identity']=m.fingerprint(selection)
    metric=dict(sections=8,section_accuracy=1.,macro_f1_present=1.,duration_weighted_accuracy=1.,
        per_class={label:dict(support=1,f1=1.) for label in m.LABELS},label_order=m.LABELS,
        confusion_matrix=[[int(i==j) for j in range(8)] for i in range(8)])
    q=dict(protocol=m.QUALITY_PROTOCOL,split='val',full_validation=True,section_agreement=1.,changed_sections=[],coverage_diagnostics=[],
        weights_updated=True,weights_unified=True,weights_requantized=False,retrained=False,test_used=False,manual_labels_changed=False,boundaries_changed=False,
        all_windows_reencoded=True,partial_embeddings_reused=False,encoder_compute_units='CPU_ONLY',canonical_profile='long_120',selection_identity=selection['identity'],
        baseline_current_pipeline=copy.deepcopy(metric),candidate=copy.deepcopy(metric),artifact_sizes=dict(previous_compiled_encoder_bytes=1000,canonical_compiled_encoder_bytes=350))
    q['comparison_to_current_pipeline']=m.quality_comparison(q['candidate'],q['baseline_current_pipeline'])
    return q,selection


class CanonicalPhonePreparationTests(unittest.TestCase):
    def test_owned_diagnostic_is_bound_to_reviewed_cpu_control_and_exact_artifacts(self):
        path=Path(m.__file__).parent/'evidence/canonical_cpu_control_v1.json'
        self.assertEqual(m.sha_file(path),m.CPU_RECEIPT_SHA);receipt=m.read(path)
        selection=dict(reports=receipt['encoder_reports']);q=dict(artifact_sizes=dict(canonical_compiled_encoder_bytes=receipt['compiled_encoder_bytes']))
        m.validate_owned_diagnostic_control(receipt,receipt['quality_sha256'],selection,q)
        for key,value in [('mode','cpuAndGPU'),('completed_runs',3),('all_functional_checks_passed',False),('numerical_identity_claimed',True)]:
            bad=copy.deepcopy(receipt);bad[key]=value
            with self.assertRaises(ValueError):m.validate_owned_diagnostic_control(bad,receipt['quality_sha256'],selection,q)
        bad=copy.deepcopy(receipt);bad['runs'][2]['final_probability_max_abs_error']=.026
        with self.assertRaises(ValueError):m.validate_owned_diagnostic_control(bad,receipt['quality_sha256'],selection,q)
        with self.assertRaises(ValueError):m.validate_owned_diagnostic_control(receipt,'different-quality',selection,q)
        with self.assertRaises(ValueError):m.validate_owned_diagnostic_control(receipt,receipt['quality_sha256'],dict(reports={}),q)

    def test_diagnostic_admission_preserves_feature_failure_and_does_not_relax_final_gates(self):
        check=dict(preflight_policy='functional-benchmark',numerical_passed=False,functional_passed=True,
            audio_features=dict(passed=False,shape_and_finite_passed=True,failed_values=45),
            base_probabilities=dict(passed=True,max_abs_error=9.328126907348633e-6),
            native_context=dict(passed=True,section_agreement=1.,max_probability_error=1.0281801223754883e-5,changed_eligible_indices=[]))
        with self.assertRaises(m.PreflightMismatch):m.finalize_preflight_admission(copy.deepcopy(check),'strict-feature')
        admitted=m.finalize_preflight_admission(copy.deepcopy(check),'owned-buffer-diagnostic')
        self.assertTrue(admitted['passed']);self.assertFalse(admitted['numerical_passed']);self.assertFalse(admitted['audio_features']['passed'])
        self.assertEqual(admitted['audio_features']['failed_values'],45)
        for field,key,value in [('native_context','max_probability_error',.0756),('native_context','section_agreement',.95),
                                ('base_probabilities','max_abs_error',.055),('audio_features','shape_and_finite_passed',False)]:
            bad=copy.deepcopy(check);bad[field][key]=value
            with self.assertRaises(m.PreflightMismatch):m.finalize_preflight_admission(bad,'owned-buffer-diagnostic')

    def test_serial_role_schedule_preserves_all_windows_fusion_and_head_chunk_clocks(self):
        from prepare_encoder_pipeline_iphone import pcm_chunks
        settings=dict(sample_rate=1000,head_window_seconds=420);wave=np.arange(540000,dtype=np.float32)
        active=[];events=[]
        def encode(role,audio,seconds):
            if role not in active:active.append(role)
            self.assertEqual(active,[role]);events.append((role,len(audio),seconds))
            frames=len(audio)//1000*25
            return np.full((frames,4),float(audio[0])+seconds+(role=='muq'),dtype=np.float32)
        def release():events.append(('release',));active.clear()
        def pair(audio,seconds):
            return [np.full((len(audio)//1000*25,4),float(audio[0])+seconds+(role=='muq'),dtype=np.float32) for role in ('musicfm','muq')]
        actual=list(m.serial_chunks(wave,encode,release,settings,width=4));expected=list(pcm_chunks(wave,pair,settings,width=4))
        self.assertEqual([offset for offset,_ in actual],[0.,420.])
        for (a,av),(b,bv) in zip(actual,expected):self.assertEqual(a,b);np.testing.assert_array_equal(av,bv)
        self.assertEqual(sum(v==('release',) for v in events),4);self.assertFalse(active)
        # The 60-second remainder in the long stream retains the requested 120-second profile.
        self.assertIn(('musicfm',60000,120),events)

    def test_serial_schedule_releases_on_failure_and_rejects_window_clock_mismatch(self):
        settings=dict(sample_rate=1000,head_window_seconds=420);wave=np.zeros(150000,dtype=np.float32);releases=[]
        def bad(role,audio,seconds):
            if role=='musicfm':raise RuntimeError('native failure')
        with self.assertRaises(RuntimeError):list(m.serial_chunks(wave,bad,lambda:releases.append(1),settings,width=4))
        self.assertEqual(len(releases),1)
        def shifted(role,audio,seconds):
            frames=len(audio)//1000*25-(role=='muq')
            return np.zeros((frames,4),dtype=np.float32)
        with self.assertRaisesRegex(ValueError,'per-window frame counts'):list(m.serial_chunks(wave,shifted,lambda:None,settings,width=4))

    def test_only_full_unchanged_quality_with_reencoded_tails_and_actual_saving_is_admitted(self):
        q,s=quality();m.validate_canonical_quality(q,s)
        for key,value in [('full_validation',False),('section_agreement',.99),('changed_sections',[1]),('coverage_diagnostics',[1]),
                          ('test_used',True),('partial_embeddings_reused',True),('weights_requantized',True),('boundaries_changed',True)]:
            bad=copy.deepcopy(q);bad[key]=value
            with self.assertRaises(ValueError):m.validate_canonical_quality(bad,s)
        bad=copy.deepcopy(q);bad['artifact_sizes']['canonical_compiled_encoder_bytes']=1001
        with self.assertRaises(ValueError):m.validate_canonical_quality(bad,s)
        bad=copy.deepcopy(q);bad['candidate']['per_class']['bridge']['f1']=.5
        with self.assertRaises(ValueError):m.validate_canonical_quality(bad,s)
        bad=copy.deepcopy(s);bad['reference_quality_sha256']='changed'
        with self.assertRaises(ValueError):m.validate_canonical_quality(q,bad)

    def test_mac_dispatch_uses_one_package_per_role_and_releases_on_function_or_shape_change(self):
        calls=[]
        class Model:
            def __init__(self,path,compute_units,function_name=None):calls.append((path,compute_units,function_name))
        ct=SimpleNamespace(models=SimpleNamespace(CompiledMLModel=Model),ComputeUnit=SimpleNamespace(CPU_ONLY='cpuOnly'))
        backend=m.CanonicalMacPipeline(ct,Path('/tmp/Assets'),{},{});first=backend.load('musicfm_750',750)
        self.assertIs(first,backend.load('musicfm_750',750))
        self.assertIsNot(first,backend.load('musicfm',750)) # Same frames, different function (long-stream tail).
        backend.load('musicfm_750',750)
        backend.load('musicfm_3000',3000);backend.load('musicfm',750)
        self.assertEqual([x[2] for x in calls],['short_30','partial','short_30','long_120','partial'])
        self.assertEqual({x[0] for x in calls},{'/tmp/Assets/musicfm.mlmodelc'})
        self.assertTrue(all(x[1]=='cpuOnly' for x in calls));self.assertEqual(len(backend.models),1)
        partial=backend.load('musicfm',750);self.assertIsNot(partial,backend.load('musicfm',739))
        backend.load('muq_750',750);backend.load('head');backend.release();self.assertFalse(backend.models)


if __name__=='__main__':unittest.main()
