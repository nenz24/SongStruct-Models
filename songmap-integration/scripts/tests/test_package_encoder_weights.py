import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import package_encoder_weights as m


def toy(ct,frames,alter=False):
    from coremltools.converters.mil import Builder as mb
    @mb.program(input_specs=[mb.TensorSpec(shape=(1,16,frames))],opset_version=ct.target.iOS18)
    def program(x):
        weights=np.random.RandomState(2).normal(size=(32,16,1)).astype(np.float32)
        if alter:weights[0,0,0]+=10
        projected=mb.conv(x=x,weight=weights,pad_type='valid')
        a=mb.slice_by_index(x=projected,begin=[0,0,0],end=[1,16,frames])
        b=mb.slice_by_index(x=projected,begin=[0,16,0],end=[1,32,frames])
        return mb.mul(x=a,y=mb.sigmoid(x=b))
    model=ct.convert(program,source='milinternal',convert_to='mlprogram',compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,skip_model_load=True)
    import coremltools.optimize.coreml as optimize
    return optimize.palettize_weights(model,config=optimize.OptimizationConfig(global_config=optimize.OpPalettizerConfig(mode='uniform',nbits=6,weight_threshold=0)))


@unittest.skipUnless(importlib.util.find_spec('coremltools'),'Actual MIL sharing tests require coremltools')
class EncoderSharingTests(unittest.TestCase):
    def test_dynamic_partial_shape_contract_and_graph_are_preserved(self):
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        bound=ct.RangeDim(lower_bound=1,upper_bound=20,default=13)
        @mb.program(input_specs=[mb.TensorSpec(shape=(1,16,bound.symbol))],opset_version=ct.target.iOS18)
        def program(x):
            weight=np.random.RandomState(2).normal(size=(32,16,1)).astype(np.float32)
            left,right=mb.split(x=mb.conv(x=x,weight=weight),num_splits=2,axis=1)
            return mb.mul(x=left,y=mb.sigmoid(x=right))
        model=ct.convert(program,source='milinternal',convert_to='mlprogram',compute_precision=ct.precision.FLOAT32,
            inputs=[ct.TensorType(name='x',shape=(1,16,bound))],minimum_deployment_target=ct.target.iOS18,skip_model_load=True)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sources={}
            for name,frames in zip(m.FUNCTIONS,(7,11,13)):
                path=root/(name+'.mlpackage')
                (model if name=='partial' else toy(ct,frames)).save(str(path));sources[name]=path
            report=m.pack(ct,sources,root/'shared.mlpackage')
            self.assertTrue(all(report['checks']['partial'].values()))

    def test_actual_multifunction_deduplicates_without_changing_graph_weights_or_contract(self):
        import coremltools as ct
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sources={}
            for name,frames in zip(m.FUNCTIONS,(7,11,13)):
                path=root/(name+'.mlpackage');toy(ct,frames).save(str(path));sources[name]=path
            before={k:m.package_digest(v) for k,v in sources.items()}
            report=m.pack(ct,sources,root/'shared.mlpackage')
            self.assertTrue(report['smaller']);self.assertFalse(report['weights_requantized']);self.assertFalse(report['graphs_rewritten'])
            self.assertTrue(all(all(v.values()) for v in report['checks'].values()))
            self.assertEqual(report['source_digests'],before)
            self.assertEqual({k:m.package_digest(v) for k,v in sources.items()},before)
            source_weight_bytes=sum(m.size_bytes(v/'Data/com.apple.CoreML/weights') for v in sources.values())
            packed_weight_bytes=m.size_bytes(root/'shared.mlpackage/Data/com.apple.CoreML/weights')
            self.assertGreater(source_weight_bytes,0);self.assertLess(packed_weight_bytes,source_weight_bytes)
            with self.assertRaises(ValueError):m.pack(ct,sources,root/'shared.mlpackage')

    def test_graph_digest_detects_changed_constants_and_function_wiring(self):
        import coremltools as ct
        with tempfile.TemporaryDirectory() as d:
            paths=[Path(d)/(name+'.mlpackage') for name in ('original','changed')]
            toy(ct,7).save(str(paths[0]));toy(ct,7,alter=True).save(str(paths[1]))
            _,a=m.model_program(ct,paths[0]);_,b=m.model_program(ct,paths[1])
            self.assertNotEqual(m.function_digest(a.functions['main']),m.function_digest(b.functions['main']))
            before=m.function_digest(a.functions['main'])
            function=a.functions['main'];function.set_outputs([list(function.operations)[-2].outputs[0]])
            self.assertNotEqual(m.function_digest(function),before)
        with self.assertRaises(ValueError):m.pack(ct,{'short_30':Path('missing')},Path('unused'))


if __name__=='__main__':unittest.main()
