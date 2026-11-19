import copy
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import canonical_encoder_quality as m


class CanonicalQualityTests(unittest.TestCase):
    def test_partial_windows_are_actually_reencoded_without_reusing_old_columns(self):
        settings=dict(sample_rate=1000,head_window_seconds=420)
        wave=np.zeros(150000,dtype=np.float32);old=np.full((3750,4096),-99.,dtype=np.float32);calls=[]
        def encode(audio,seconds):
            calls.append((len(audio),seconds));n=len(audio)//1000*25
            return [np.full((n,1024),seconds+len(audio)/1000,dtype=np.float32),np.full((n,1024),seconds+len(audio)/1000+1,dtype=np.float32)]
        result=list(m.reencoded_chunks(wave,[(old,0.,'old')],encode,settings,'new'))
        self.assertEqual(len(calls),7);self.assertIn((30000,120),calls)
        values,offset,identity,events=result[0]
        self.assertEqual((offset,identity),(0.,'new'));self.assertEqual(values.shape,old.shape)
        np.testing.assert_array_equal(values[3000:,2048:3072],150.)
        self.assertFalse(np.any(values==-99.));np.testing.assert_array_equal(old,-99.)
        self.assertEqual(sum(e['route']=='canonical_partial_cpu' for e in events),2)
        with self.assertRaises(ValueError):list(m.reencoded_chunks(wave,[(old,1.,'old')],encode,settings,'new'))
        with self.assertRaises(ValueError):list(m.reencoded_chunks(wave,[(old[:-3],0.,'old')],encode,settings,'new'))

    def test_only_complete_canonical_export_and_native_long_control_are_accepted(self):
        checks=[dict(function=name,frames=n,passed=True,packed_vs_unified=[dict(passed=True),dict(passed=True)],vs_original_profile=dict(passed=True))
            for name,n in [('short_30',750),('long_120',3000),('partial',594),('partial',739),('partial',1500),('partial',2989)]]
        profile=dict(non_parameter_graph_unchanged=True,parameters=110,original_graph_sha256='a'*64,unified_graph_sha256='a'*64)
        r=dict(protocol=m.EXPORT_PROTOCOL,encoder='musicfm',canonical_profile=m.CANONICAL,candidate_for_quality_validation=True,native_cpu_passed=True,
            weights_updated=True,weights_unified=True,weights_requantized=False,retrained=False,evidence=dict(quality_sha256='q',repair_sha256='r'),
            profiles={name:copy.deepcopy(profile) for name in ('short_30','long_120','partial')},native_checks=checks)
        m.validate_candidate(r,'musicfm','q','r')
        for key,value in [('weights_updated',False),('native_cpu_passed',False),('weights_requantized',True),('encoder','muq')]:
            bad=copy.deepcopy(r);bad[key]=value
            with self.assertRaises(ValueError):m.validate_candidate(bad,'musicfm','q','r')
        bad=copy.deepcopy(r);bad['native_checks'][1]['vs_original_profile']=None
        with self.assertRaises(ValueError):m.validate_candidate(bad,'musicfm','q','r')
        bad=copy.deepcopy(r);bad['native_checks'][0]['packed_vs_unified']=[]
        with self.assertRaises(ValueError):m.validate_candidate(bad,'musicfm','q','r')

    def test_thirty_second_tail_of_long_stream_selects_partial_function(self):
        created=[]
        class Native:
            def __init__(self,path,compute_units,function_name=None):
                self.role=Path(path).stem;self.function=function_name;created.append(self)
            def get_spec(self):return SimpleNamespace(description=SimpleNamespace(input=[SimpleNamespace(name='input')]))
            def predict(self,inputs):
                value=next(iter(inputs.values()))
                if self.role=='log_mel':return dict(mel=np.zeros((1,128,(value.shape[1]-2048)//240),dtype=np.float32))
                if self.role.endswith('_conv'):return dict(conv=np.zeros((1,value.shape[-1]//4,1024),dtype=np.float32))
                return dict(embedding=np.zeros_like(inputs['conv_in']))
        ct=SimpleNamespace(models=SimpleNamespace(MLModel=Native,CompiledMLModel=Native),ComputeUnit=SimpleNamespace(CPU_ONLY='cpuOnly'))
        app=dict(models={k:k+'.mlpackage' for k in ('log_mel','musicfm_conv','muq_conv')},rotary_inv_freq=dict(musicfm=[],muq=[]))
        rotary=SimpleNamespace(rotary=lambda n,f:np.zeros((2,n,1,1,64),dtype=np.float32))
        with patch.dict(sys.modules,{'deployment.verify_encoders':rotary}):
            backend=m.Encoders(ct,Path('/tmp'),app,{},dict(musicfm=Path('/tmp/a/report.json'),muq=Path('/tmp/b/report.json')))
            audio=np.zeros(30*24000,dtype=np.float32)
            backend.encode_pair(audio,120)
            self.assertEqual([v.function for v in created if v.function],['partial','partial'])
            backend.encode_pair(audio,30)
            self.assertEqual([v.function for v in created if v.function][-2:],['short_30','short_30'])
            backend.release()


if __name__=='__main__':unittest.main()
