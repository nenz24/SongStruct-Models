from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from deployment.fixed_musicfm import fixed_shapes, assert_fixed_inputs, materialize_package


def spec(frames,flexible=False):
    features=[]
    for name,shape in fixed_shapes(frames).items():
        ranges=[NS(lowerBound=1,upperBound=3000)] if flexible else []
        array=NS(shape=shape,shapeRange=NS(sizeRanges=ranges),enumeratedShapes=NS(shapes=[]))
        features.append(NS(name=name,type=NS(multiArrayType=array)))
    op=NS(type='constexpr_blockwise_shift_scale',blocks=[],outputs=[])
    return NS(description=NS(input=features),mlProgram=NS(functions={
        'main':NS(block_specializations={'CoreML8':NS(operations=[op])})}))


class StaticMusicFMTests(unittest.TestCase):
    def test_fixed_inputs_reject_remaining_ranges(self):
        assert_fixed_inputs(spec(750),750)
        with self.assertRaisesRegex(ValueError,'remains flexible'):
            assert_fixed_inputs(spec(750,True),750)
        with self.assertRaisesRegex(ValueError,'contract differs'):
            assert_fixed_inputs(spec(750),2977)
        with self.assertRaises(ValueError):
            fixed_shapes(3001)

    def test_each_fixture_reloads_source_and_retains_compression(self):
        loads=[]
        materializations=[]
        metadata={'compression_scheme':'int8','precision':'fp32'}
        class Model:
            def __init__(self,path,**kwargs):
                self.path=path
                self.user_defined_metadata=dict(metadata)
                self.frames=materializations[-1]['main']['conv_in'][1] if 'materialized.mlpackage' in path else 750
                loads.append(path)
            def get_spec(self):
                return spec(self.frames)
            def predict(self,inputs):
                return {'embedding':np.ones((1,inputs['conv_in'].shape[1],1024),np.float32)}
            def save(self,path):
                Path(path).mkdir()
        def materialize(api,model,frames):
            mapping={'main':fixed_shapes(frames)}
            materializations.append(mapping)
            return NS(save=lambda path:Path(path).mkdir())
        ct=NS(models=NS(MLModel=Model,utils=NS(materialize_dynamic_shape_mlmodel=materialize)),
              ComputeUnit=NS(CPU_ONLY='cpu'))
        with tempfile.TemporaryDirectory() as folder, patch('deployment.fixed_musicfm.materialize_fp32_model',side_effect=materialize):
            for frames in (750,2977):
                result=materialize_package(ct,'original.mlpackage',frames,Path(folder)/f'{frames}.mlpackage',
                    {'conv_in':np.ones((1,frames,1024),np.float32),
                     'rotary':np.ones((2,frames,1,1,64),np.float32)})
                self.assertTrue(result['cpu_parity']['same_precision_strict_passed'])
                self.assertEqual(result['shape_policy'],'fixed_per_fixture')
                self.assertFalse(result['iphone_verified'])
        self.assertEqual(loads.count('original.mlpackage'),2)
        self.assertEqual(materializations[1]['main']['rotary'],(2,2977,1,1,64))


if __name__=='__main__':
    unittest.main()
