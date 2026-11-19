import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import validate_encoder_pipeline as m


def fingerprint(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


class EncoderPipelineTests(unittest.TestCase):
    def test_full_windows_replace_all_four_streams_and_partial_tails_remain_exact(self):
        settings=dict(sample_rate=1000,head_window_seconds=420)
        audio=np.arange(125000,dtype=np.float32);base=np.zeros((3123,4096),dtype=np.float32)
        base[3000:]=np.arange(4096,dtype=np.float32)[None,:]
        original=base.copy();calls=[]
        def encode(role,wave,frames):
            self.assertEqual(len(wave),frames//25*1000)
            calls.append((role,float(wave[0]),frames))
            return np.full((frames,1024),(1 if role=='musicfm' else 2)+frames/750,dtype=np.float32)
        new,events=m.overlay_chunk(audio,base,4.,encode,settings)
        self.assertEqual(len(calls),10);self.assertEqual(len(events),14)
        for column,value in zip((0,1024,2048,3072),(2,3,5,6)):
            np.testing.assert_array_equal(new[:3000,column:column+1024],value)
        np.testing.assert_array_equal(new[3000:],original[3000:]);np.testing.assert_array_equal(base,original)
        tails=[e for e in events if e['route']=='legacy_partial_cache_unchanged']
        self.assertEqual(len(tails),4);self.assertTrue(all(e['start']==124 and e['end']==129 for e in tails))
        self.assertTrue(all(e['retained_fused_frames']==123 for e in tails))

    def test_exact_full_final_window_accepts_only_bounded_stride_trim(self):
        audio=np.zeros(120000,dtype=np.float32);settings=dict(sample_rate=1000)
        def encode(role,wave,frames):return np.ones((frames,1024),dtype=np.float32)
        for n in (3000,2997):
            values,events=m.overlay_chunk(audio,np.zeros((n,4096),dtype=np.float32),0.,encode,settings)
            self.assertTrue(np.all(values==1));self.assertTrue(all(e['route']=='new_static' for e in events))
        with self.assertRaises(ValueError):m.overlay_chunk(audio,np.zeros((2994,4096),dtype=np.float32),0.,encode,settings)

    def test_short_song_uses_legacy_features_without_padding_or_encoder_calls(self):
        audio=np.ones(5000);base=np.arange(123*4096,dtype=np.float32).reshape(123,4096)
        def forbidden(*args):raise AssertionError('Partial audio was sent to a static profile')
        result,events=m.overlay_chunk(audio,base,0.,forbidden,dict(sample_rate=1000))
        np.testing.assert_array_equal(result,base)
        self.assertTrue(all(e['route']=='legacy_partial_cache_unchanged' for e in events))

    def test_chunk_clock_and_chunk_inventory_are_checked(self):
        settings=dict(sample_rate=1000,head_window_seconds=120)
        audio=np.zeros(125000);one=np.broadcast_to(np.zeros((1,4096),dtype=np.float32),(3000,4096))
        two=np.zeros((123,4096),dtype=np.float32)
        def encode(role,wave,frames):return np.ones((frames,1024),dtype=np.float32)
        result=list(m.overlaid_chunks(audio,[(one,0.,'old'),(two,120.,'old')],'new',encode,settings))
        self.assertEqual([x[1] for x in result],[0.,120.]);self.assertTrue(all(x[2]=='new' for x in result))
        np.testing.assert_array_equal(result[1][0],two)
        for bad in ([(one,1.,'old'),(two,120.,'old')],[(one,0.,'old')],[(one,0.,'old'),(two,120.,'old'),(two,240.,'old')]):
            with self.assertRaises(ValueError):list(m.overlaid_chunks(audio,bad,'new',encode,settings))

    def test_nonfinite_or_wrong_shape_encoder_outputs_stop_instead_of_filling(self):
        audio=np.zeros(30000);base=np.zeros((750,4096),dtype=np.float32)
        for values in (np.full((750,1024),float('nan')),np.zeros((749,1024)),np.zeros((1,750,1024))):
            with self.assertRaises(ValueError):m.overlay_chunk(audio,base,0.,lambda *args:values,dict(sample_rate=1000))
        with self.assertRaises(ValueError):m.overlay_chunk(audio,base*float('nan'),0.,lambda *args:None,dict(sample_rate=1000))

    def test_all_original_weight_parity_checks_and_both_profiles_required(self):
        proof=dict(protocol=m.GRAPH_PROTOCOL,encoder='muq',compression_scheme='palette6',same_weight_rewrite_passed=True,
            plan_id='p',provenance_id='v',checks=[dict(frames=n,**{k:dict(passed=True) for k in ('native_wrapper','rewrite_vs_original','trace_vs_original','coreml_fp32_vs_original')},compression_vs_fp32=dict(passed=False)) for n in (750,3000)],
            fixtures=[dict(frames=n,resource=f'muq_{n}') for n in (750,3000)])
        m.validate_export(proof,'muq','p','v') # Lossy compression parity is not falsely called an approval.
        for field,value in [('encoder','musicfm'),('plan_id','other'),('same_weight_rewrite_passed',False),('compression_scheme','fp32')]:
            bad=dict(proof);bad[field]=value
            with self.assertRaises(ValueError):m.validate_export(bad,'muq','p','v')
        bad=copy.deepcopy(proof);bad['checks'][0]['rewrite_vs_original']['passed']=False
        with self.assertRaises(ValueError):m.validate_export(bad,'muq','p','v')
        bad=copy.deepcopy(proof);bad['fixtures'].pop()
        with self.assertRaises(ValueError):m.validate_export(bad,'muq','p','v')

    def test_quality_report_does_not_hide_a_rare_class_regression(self):
        baseline=dict(section_accuracy=.8,macro_f1_present=.75,per_class={label:dict(f1=.75) for label in m.LABELS})
        candidate=copy.deepcopy(baseline)
        self.assertTrue(m.quality_comparison(candidate,baseline)['no_observed_drop'])
        candidate['macro_f1_present']=.76;candidate['per_class']['bridge']['f1']=.7
        result=m.quality_comparison(candidate,baseline)
        self.assertFalse(result['no_observed_drop']);self.assertAlmostEqual(result['per_class_f1_delta']['bridge'],-.05)

    def test_compact_resume_cache_checks_identity_payload_arrays_and_loss(self):
        with tempfile.TemporaryDirectory() as d:
            info=Path(d)/'result.json';arrays=Path(d)/'result.npz'
            self.assertIsNone(m.cached_result(info,arrays,'id',read,fingerprint,sha))
            np.savez(arrays,audio=np.zeros((3,4096),dtype=np.float32),probabilities=np.zeros((2,8)))
            item=dict(identity='id',array_sha256=sha(arrays),nll_sum=1.,sections=[])
            item['payload_sha256']=fingerprint(item);info.write_text(json.dumps(item))
            self.assertEqual(m.cached_result(info,arrays,'id',read,fingerprint,sha),item)
            with self.assertRaises(ValueError):m.cached_result(info,arrays,'other',read,fingerprint,sha)
            changed=dict(item,nll_sum=2.);info.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):m.cached_result(info,arrays,'id',read,fingerprint,sha)
            info.write_text(json.dumps(item));arrays.write_bytes(b'changed')
            with self.assertRaises(ValueError):m.cached_result(info,arrays,'id',read,fingerprint,sha)


if __name__=='__main__':unittest.main()
