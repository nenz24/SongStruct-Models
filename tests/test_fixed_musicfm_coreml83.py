"""Real MIL conversion regression; does not need macOS prediction runtime."""
import importlib.util
import tempfile
from pathlib import Path
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('coremltools'),'coremltools not installed')
class CoreML83MaterializationTests(unittest.TestCase):
    def test_fp32_fixed_graph_preserves_compressed_ops_without_added_half_precision(self):
        import coremltools as ct
        if not ct.__version__.startswith('8.3'):
            self.skipTest('Regression targets the deployed coremltools 8.3 materializer')
        import coremltools.optimize.coreml as opt
        from coremltools.converters.mil.mil import Builder as mb, get_new_symbol
        from deployment.fixed_musicfm import materialize_fp32_model, assert_fixed_inputs
        from deployment.diagnose_musicfm import precision_inventory
        from deployment.compress_coreml import operation_counts
        rng=np.random.default_rng(42)
        weights1=rng.normal(size=(32,1024)).astype(np.float32)*.01
        weights2=rng.normal(size=(1024,32)).astype(np.float32)*.01
        for scheme in ('int8','palette6'):
            with self.subTest(scheme=scheme), tempfile.TemporaryDirectory() as folder:
                symbol=get_new_symbol()
                @mb.program(input_specs=[mb.TensorSpec(shape=(1,symbol,1024)),
                    mb.TensorSpec(shape=(2,symbol,1,1,64))],opset_version=ct.target.iOS18)
                def program(conv_in,rotary):
                    values=mb.linear(x=conv_in,weight=weights1)
                    values=mb.linear(x=values,weight=weights2)
                    positions=mb.reduce_mean(x=rotary,axes=[0,2,3,4],keep_dims=False)
                    positions=mb.expand_dims(x=positions,axes=[0,2])
                    return mb.add(x=values,y=positions,name='embedding')
                time=ct.RangeDim(lower_bound=1,upper_bound=32,default=3)
                source=ct.convert(program,source='milinternal',convert_to='mlprogram',
                    inputs=[ct.TensorType(name='conv_in',shape=(1,time,1024)),
                            ct.TensorType(name='rotary',shape=(2,time,1,1,64))],
                    compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.iOS18,
                    skip_model_load=True)
                if scheme=='int8':
                    source=opt.linear_quantize_weights(source,config=opt.OptimizationConfig(
                        global_config=opt.OpLinearQuantizerConfig(dtype=np.int8)))
                    compressed_op='constexpr_blockwise_shift_scale'
                else:
                    # Uniform avoids a test dependency on sklearn; exercises real 6-bit LUT storage.
                    source=opt.palettize_weights(source,config=opt.OptimizationConfig(
                        global_config=opt.OpPalettizerConfig(mode='uniform',nbits=6)))
                    compressed_op='constexpr_lut_to_dense'
                path=Path(folder)/'source.mlpackage'
                source.save(str(path))
                loaded=ct.models.MLModel(str(path),skip_model_load=True)
                original_types=precision_inventory(loaded.get_spec())['declared_tensor_dtype_counts']
                self.assertEqual(original_types.get('FLOAT16',0),0)
                # Demonstrate the precision regression in the exact public utility previously used.
                legacy=Path(folder)/'legacy.mlpackage'
                ct.models.utils.materialize_dynamic_shape_mlmodel(loaded,
                    {'main':{'conv_in':(1,3,1024),'rotary':(2,3,1,1,64)}},str(legacy))
                legacy_model=ct.models.MLModel(str(legacy),skip_model_load=True)
                legacy_types=precision_inventory(legacy_model.get_spec())['declared_tensor_dtype_counts']
                self.assertGreater(legacy_types.get('FLOAT16',0),0)
                for frames in (3,7):
                    loaded=ct.models.MLModel(str(path),skip_model_load=True)
                    fixed=materialize_fp32_model(ct,loaded,frames)
                    assert_fixed_inputs(fixed.get_spec(),frames)
                    dtypes=precision_inventory(fixed.get_spec())['declared_tensor_dtype_counts']
                    self.assertEqual(dtypes.get('FLOAT16',0),0)
                    self.assertGreater(dtypes.get('FLOAT32',0),0)
                    self.assertGreater(operation_counts(fixed.get_spec()).get(compressed_op,0),0)
                    self.assertEqual(fixed.get_spec().specificationVersion,source.get_spec().specificationVersion)


if __name__=='__main__':
    unittest.main()
